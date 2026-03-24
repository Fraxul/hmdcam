#pragma once
#include "common/CANBus.h"
#include "common/FxRenderView.h"
#include "common/ScrollingBuffer.h"
#include "SingleEyeFitter/SingleEyeFitter.h"
#include "TrackingThreadBase.h"
#include "one_euro_filter.h"
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <glm/glm.hpp>
#include <opencv2/core.hpp>
#include "CuDLAStandaloneRunner.h"

#include <algorithm>
#include <memory>
#include <vector>

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

class EyeTrackingService;

class EyeTrackingService {
public:
  EyeTrackingService();
  ~EyeTrackingService();

  bool loadCalibrationData();
  bool loadCalibrationData(cv::FileStorage&);

  void saveCalibrationData();
  void saveCalibrationData(cv::FileStorage&);

  void renderIMGUI();

  void renderSceneGizmos_preUI(FxRenderView* renderViews);
  void renderSceneGizmos_postUI(FxRenderView* renderViews);

  void setInputDeviceOverride(size_t eyeIdx, const std::string& s) {
    assert(eyeIdx < 2);
    m_processingState[eyeIdx].m_cameraDeviceNameOverride = s;
  }

  bool processFrame(); // Called from main thread

  void requestCapture();

  glm::vec2 getPitchYawAnglesForEye(size_t eyeIdx);

  const char* getDebugPerfStatsForEye(size_t eyeIdx);
  cv::Mat& getDebugViewForEye(size_t eyeIdx);
  bool m_debugDrawOverlays = true; // Affects content returned by getDebugViewForEye

  bool m_debugShowFeedbackView = false; // Draw eye camera(s) over the scene in renderSceneGizmos(). Required for getDebugViewForEye() to return anything.
  float m_debugFeedbackBrightness = 1.0f;
  bool m_debugFreezeCapture = false;
  bool m_debugDisableProcessing = false;

  bool m_debugSaveBadFitImages = false;
  uint32_t m_debugSaveBadFitIntervalMs = 500; // Write at most one image per this many ms


  enum CalibrationState {
    kWaitingForValidFrames,
    kCentering,
    kCalibrated
  };

  class ProcessingState : public TrackingThreadBase {
  public:
    virtual ~ProcessingState();

    EyeTrackingService* m_service = nullptr; // Ref to containing service
    size_t m_eyeIdx = 0; // Which eye this is, for debug prints

    virtual void internalUpdateStateOnCaptureOpen();
    virtual void internalProcessOneCapture();
    void postprocessOneEye();
    bool postprocessOneEye_fitEllipse();

    cv::Rect m_captureCropRect;
    cv::Mat m_tempRGBDebugMat; // Temporary drawing target for debug feedback view.

    cv::Mat m_roiMaskMat;
    cv::Mat m_roiDilatedMaskMat;

    uint64_t m_lastDebugBadFitCaptureTimestampMs = 0; // currentRealTimeMs

    // Profiling stats
    float m_lastFrameTotalProcessingTimeMs = 0.0f;

    float m_lastFramePreProcessingTimeMs = 0.0f;
    float m_lastFrameROITimeMs = 0.0f;
    float m_lastFrameROIToSegmentationTimeMs = 0.0f;
    float m_lastFrameSegmentationTimeMs = 0.0f;
    float m_lastFramePostProcessingTimeMs = 0.0f;
    float m_lastFrameDebugViewTimeMs = 0.0f;

    // Calibration state machine data

    uint32_t m_contiguousValidFrameCounter = 0;
    uint32_t m_contiguousInvalidFrameCounter = 0;
    uint32_t m_lastInvalidFrameRunLength = 0; // Updated when we start seeing valid frames again while inside a run of invalid frames

    // Crosshair state tracking for runs of valid/invalid frames
    bool m_shouldShowCrosshair = true;

    ScrollingBuffer<cv::RotatedRect> m_calibrationSamples {60};
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
    RHISurface::ptr m_eyeTrackingDebugTexture;

    char m_debugPerfStatsBuffer[256];

    // Additional debugging data
    struct GraphData {
      float deltaSize = 0.0f;
      float deltaCenter = 0.0f;
      float deltaAngle = 0.0f;
    };
    bool m_freezeGraphData = false;
    ScrollingBuffer<GraphData> m_graphData { 120 /*samples*/ };
  };


  // per-eye state
  ProcessingState m_processingState[2];

protected:

  void CANTransmitEyeAngles(); // called in processFrame()

  // Calibration data and settings
  float m_focalLength = 6.0; // millimeters. seems only vaguely related to the actual lens focal length.
  float m_pixelPitchMicrons = 3.0; // Pixel size/pitch of the camera sensor, micrometers
  float m_eyeZ = 15.0; // millimeters
  float m_rollOffsetDeg[2] = {0.0, 0.0}; // per-eye roll correction angle, degrees

  float m_sectorCutoffAngleDeg = 45.0; // Cutoff angle for filtering top/bottom sectors of pupil contour

  float m_filterMinCutoff = 0.2;
  float m_filterDCutoff = 0.2;
  float m_filterBetaExponent = -0.8; // Filter beta is pow(10.0, m_filterBetaExponent)

  int m_hideCrosshairAfterFrameCount = 120; // Hide UI feedback crosshair after this many valid frames in a row.
  int m_showCrosshairAfterFrameCount = 16; // Show UI feedback crosshair after this many invalid frames in a row.

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
};

