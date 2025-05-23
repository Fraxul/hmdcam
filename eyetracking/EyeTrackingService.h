#pragma once
#include "common/CANBus.h"
#include "common/FxRenderView.h"
#include "common/ScrollingBuffer.h"
#include "SingleEyeFitter/SingleEyeFitter.h"
#include "one_euro_filter.h"
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <glm/glm.hpp>
#include <opencv2/core.hpp>
#include "CuDLAStandaloneRunner.h"
#include "V4L2Camera.h"

#include <algorithm>
#include <memory>
#include <vector>

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

class EyeTrackingService {
public:
  EyeTrackingService();
  ~EyeTrackingService();

  bool loadCalibrationData();
  bool loadCalibrationData(cv::FileStorage&);

  void saveCalibrationData();
  void saveCalibrationData(cv::FileStorage&);

  void renderIMGUI();
  void renderSceneGizmos(FxRenderView* renderViews);

  void setInputFilename(size_t eyeIdx, const std::string& s) {
    assert(eyeIdx < 2);
    m_processingState[eyeIdx].m_inputFilename = s;
  }

  bool processFrame(); // Called from main thread

  void CANTransmitEyeAngles();

  void requestCapture();

  glm::vec2 getPitchYawAnglesForEye(size_t eyeIdx);

  cv::Mat& getDebugViewForEye(size_t eyeIdx);
  bool m_debugDrawOverlays = true; // Affects content returned by getDebugViewForEye

  enum CalibrationState {
    kWaitingForValidFrames,
    kCentering,
    kCalibrated
  };

  struct ProcessingState {
    boost::thread m_processingThread;

    bool m_processingThreadAlive = false;
    uint64_t m_lastCaptureTimestampNs = 0; // currentTimeNs

    uint32_t m_captureFileIndex = 0; // Set to non-zero to one-shot capture to a file

    std::string m_inputFilename;
    // Ratelimiting for capture-open attempts
    uint64_t m_lastCaptureOpenAttemptTimeNs = 0; // currentTimeNs

    // Video capture object
    V4L2Camera m_capture;

    // Profiling stats
    float m_lastFramePreProcessingTimeMs = 0.0f;
    float m_lastFrameROITimeMs = 0.0f;
    float m_lastFrameSegmentationTimeMs = 0.0f;
    float m_lastFramePostProcessingTimeMs = 0.0f;

    // Calibration state machine data

    uint32_t m_contiguousValidFrameCounter = 0;
    uint32_t m_contiguousInvalidFrameCounter = 0;

    ScrollingBuffer<cv::RotatedRect> m_calibrationSamples {24};
    singleeyefitter::Circle3D<double> m_centerPupilCircle;

    glm::vec3 centerPupilNormal() const {
      // Swizzle coordinate system to make the pitch/yaw angles a bit more palatable
      return glm::vec3(-m_centerPupilCircle.normal[0], m_centerPupilCircle.normal[1], -m_centerPupilCircle.normal[2]);
    }

    cv::RotatedRect m_centerCalibrationSample;
    float m_centerPitchDeg = 0.0f;
    float m_centerYawDeg = 0.0f;

    CalibrationState m_calibrationState = kWaitingForValidFrames;

    // Center offset of the camera capture, used for translating between the eye-fitter
    // coordinate system (zero at center) and the capture/image coordinate system (zero at left-top)
    cv::Point2f m_captureCenterOffset;

    // ROI scale output
    cv::Mat m_roiScaleMat;

    // Offset from segmentation ROI coordinates to capture mat coordinates. Always positive.
    cv::Point2i m_lastSegROIToCaptureMatOffset;

    // Pupil mask, filled by postprocessing the segmentation network output
    cv::Mat m_pupilMask;

    // Postprocessing output
    cv::RotatedRect m_pupilEllipse;
    bool m_eyeFitterOutputsValid = false;
    singleeyefitter::Circle3D<double> m_fitPupilCircle;

    glm::vec3 fitPupilNormal() const {
      // Swizzle coordinate system to make the pitch/yaw angles a bit more palatable
      return glm::vec3(-m_fitPupilCircle.normal[0], m_fitPupilCircle.normal[1], -m_fitPupilCircle.normal[2]);
    }

    glm::vec3 fitPupilNormalFiltered() const;

    float m_pupilRawPitchDeg = 0.0f, m_pupilRawYawDeg = 0.0f;
    float m_pupilFilteredPitchDeg = 0.0f, m_pupilFilteredYawDeg = 0.0f;

    // Filters for the rotation angles.
    // The initializers here are overwritten with the EyeTrackingService's config below.
    one_euro_filter<float, double> m_pitchFilter = {/*freq=*/ 120, /*minCutoff=*/ 1, /*beta=*/ 0.1, /*dcutoff=*/ 1};
    one_euro_filter<float, double> m_yawFilter = {/*freq=*/ 120, /*minCutoff=*/ 1, /*beta=*/ 0.1, /*dcutoff=*/ 1};

    std::vector<cv::RotatedRect> m_eyeFitterSamples;

    // Eye fitter
    singleeyefitter::EyeModelFitter m_eyeModelFitter;

    // Execution context for running the tracking model
    std::unique_ptr<CuDLAStandaloneRunner> m_segmentationExec;
    std::unique_ptr<CuDLAStandaloneRunner> m_roiExec;

    // Debug view support
    cv::Mat m_debugViewRGB; // RGB debug view, optionally with debug overlays drawn on it
    glm::vec2 m_debugBoundsCenter; // for rendering sector cutoff gizmo
    std::vector<cv::Point2f> m_debugTransformedContour;
  };


  // per-eye state
  ProcessingState m_processingState[2];

protected:


  void eyeProcessingThreadFn(size_t eyeIdx);

  void postprocessOneEye(size_t eyeIdx);
  bool postprocessOneEye_fitEllipse(size_t eyeIdx);


  // Calibration data and settings
  float m_focalLength = 6.0; // millimeters. seems only vaguely related to the actual lens focal length.
  float m_pixelPitchMicrons = 3.0; // Pixel size/pitch of the camera sensor, micrometers
  float m_eyeZ = 15.0; // millimeters
  float m_rollOffsetDeg[2] = {0.0, 0.0}; // per-eye roll correction angle, degrees

  float m_sectorCutoffAngleDeg = 45.0; // Cutoff angle for filtering top/bottom sectors of pupil contour

  float m_filterMinCutoff = 0.2;
  float m_filterDCutoff = 0.2;
  float m_filterBetaExponent = -0.8; // Filter beta is pow(10.0, m_filterBetaExponent)

  void applyCalibrationData();


  // Calibration derived data accessors
  const double pixelPitchMM() const { return m_pixelPitchMicrons / 1000.0; }
  const double pupilRadius() const { return 2.0 / pixelPitchMM(); }
  const double initialEyeZ() const { return m_eyeZ / pixelPitchMM(); }
  const double sefFocalLength() const { return m_focalLength / pixelPitchMM(); }


  // Segmentation model I/O sizes
  uint32_t m_segInputWidth = 0, m_segInputHeight = 0;
  uint32_t m_segInputRowStrideElements = 0;

  uint32_t m_segOutputRowPitchElements = 0;
  uint32_t m_segOutputPlanePitchElements = 0;

  // ROI model I/O sizes
  bool m_roiIOIsInt8 = false;
  size_t roiElementSize() const { return m_roiIOIsInt8 ? 1 : 2; }
  uint32_t m_roiInputWidth = 0, m_roiInputHeight = 0;
  uint32_t m_roiInputRowStrideElements = 0;

  uint32_t m_roiOutputWidth = 0, m_roiOutputHeight = 0;
  uint32_t m_roiOutputRowStrideElements = 0;


  uint32_t m_nextCaptureIndex = 0;
};

