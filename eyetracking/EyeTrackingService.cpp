#include "EyeTrackingService.h"
#include "CuDLAStandaloneRunner.h"
#include "VectorOps.h"
#include "common/FxThreading.h"
#include "common/Timing.h"
#include "common/mmfile.h"
#include "imgui.h"
#include "implot/implot.h"
#include "SingleEyeFitter/projection.h"
#include "stb/stb_image_write.h"
#include "rhi/RHIResources.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <math.h>

#if 0
#define FRAME_DEBUG_LOG printf
#else
#define FRAME_DEBUG_LOG(...)
#endif

static const char* calibrationFilename = "eyetracking-calibration.yml";

static const char* kCaptureDirName = "eyetracking-captures";
static const char* kBadFitCaptureDirName = "eyetracking-bad-fit-captures";

static inline glm::vec2 toGlm(const cv::Point& p) { return glm::vec2(p.x, p.y); }
static inline glm::vec2 toGlm(const cv::Point2f& p) { return glm::vec2(p.x, p.y); }
// static inline cv::Point2f toCv(const glm::vec2& p) { return cv::Point2f(p[0], p[1]); }

static inline glm::vec2 vec2AtAngleDeg(float deg) { return glm::vec2(cosf(glm::radians(deg)), sinf(glm::radians(deg))); }
// static inline glm::vec2 vec2AtAngle(float rad) { return glm::vec2(cosf(rad), sinf(rad)); }

template <typename T> void anglesToVector(T roll, T pitch, T yaw, T* outVec) {
  // Mathematica: evaluated EulerMatrix[{roll, pitch, yaw}, {3, 1, 2}] . {0, 0, 1}
  // result: {Cos[yaw] Sin[pitch] Sin[roll] + Cos[roll] Sin[yaw], -Cos[roll] Cos[yaw] Sin[pitch] + Sin[roll] Sin[yaw], Cos[pitch] Cos[yaw]}
  outVec[0] = (cos(yaw)*sin(pitch)*sin(roll)) + (cos(roll)*sin(yaw));
  outVec[1] = (-cos(roll)*cos(yaw)*sin(pitch)) + (sin(roll)*sin(yaw));
  outVec[2] = (cos(pitch)*cos(yaw));
}

template <typename T> void vectorToAngles(const T* vec, T& outPitch, T& outYaw, bool toDegrees) {
  // Note: argument ordering is atan2(y, x)

  // Pitch is rotation around / flattening along the X axis, where the 2d plane is Z, Y
  outPitch = atan2(-vec[1], vec[2]);

  // Yaw is rotation around / flattening along the Y axis, where the 2d plane is Z, X
  //outYaw = atan2(vec[0], vec[2]); // simple, but doesn't round-trip
  outYaw = atan2(vec[0], sqrt((vec[2] * vec[2]) + (vec[1] * vec[1]))); // based on Mathematica's ToSphericalCoordinates[], round-trips with anglesToVector (when roll == 0)
  if (outYaw >= M_PI)
    outYaw -= (2.0 * M_PI);

  if (toDegrees) {
    outPitch = glm::degrees(outPitch);
    outYaw = glm::degrees(outYaw);
  }
}

template <typename T> T clamp(T value, T min_, T max_) {
  return std::min<T>(max_, std::max<T>(min_, value));
}


template <typename T> glm::vec2 boundsCenterFromPoints(const std::vector<cv::Point_<T> >& points) {
  glm::vec2 boundsMin = toGlm(points[0]);
  glm::vec2 boundsMax = toGlm(points[0]);
  for (size_t i = 1; i < points.size(); ++i) {
    const cv::Point2f& p = points[i];
    boundsMin = glm::min(toGlm(p), boundsMin);
    boundsMax = glm::max(toGlm(p), boundsMax);
  }
  return (boundsMin + boundsMax) * 0.5f;
}

// [-180.0, 180.0] -> [-18000, 18000] -- 0.01deg precision in int16_t
inline int16_t serializeAngle(float angleDeg) {
  return static_cast<int16_t>(glm::clamp(angleDeg, -180.0f, 180.0f) * 100.0);
}


EyeTrackingService::EyeTrackingService() {

  // Set some ProcessingState constants
  PER_EYE {
    m_processingState[eyeIdx].m_eyeIdx = eyeIdx;
    m_processingState[eyeIdx].m_service = this;
    m_processingState[eyeIdx].m_captureDirName = "eyetracking-captures";

    // Construct capture file suffix with eye index
    m_processingState[eyeIdx].m_captureFileSuffix = (eyeIdx  ? "_right" : "_left");
  }

  loadCalibrationData();

  // Load segmentation engine
  {
    mmfile fp("eyetracking/models/eyeseg-dla-standalone.engine");
    PER_EYE {
      m_processingState[eyeIdx].m_segmentationExec.reset(new CuDLAStandaloneRunner(0, reinterpret_cast<const uint8_t*>(fp.data()), fp.size()));
    }
  }

  // Load ROI engine
  {
    mmfile fp("eyetracking/models/roi-dla-standalone.engine");
    PER_EYE {
      m_processingState[eyeIdx].m_roiExec.reset(new CuDLAStandaloneRunner(0, reinterpret_cast<const uint8_t*>(fp.data()), fp.size()));
    }
  }

  // Get the input size
  {
    // Segmentation
    const cudlaModuleTensorDescriptor& desc = m_processingState[0].m_segmentationExec->inputTensorDescriptor(0);
    m_segInputWidth = desc.w;
    m_segInputHeight = desc.h;

    m_segInputRowStrideElements = desc.stride[1] / desc.stride[0];
    printf("Segmentation image dimensions: %ux%u. Row stride is %u elements\n", m_segInputWidth, m_segInputHeight, m_segInputRowStrideElements);
  }

  {
    // ROI
    const cudlaModuleTensorDescriptor& desc = m_processingState[0].m_roiExec->inputTensorDescriptor(0);
    m_roiInputWidth = desc.w;
    m_roiInputHeight = desc.h;

    m_roiInputRowStrideElements = desc.stride[1] / desc.stride[0];

    printf("ROI image dimensions: %ux%u. Row stride is %u elements\n", m_roiInputWidth, m_roiInputHeight, m_roiInputRowStrideElements);
  }

  // Get the output size
  {
    // Segmentation
    const cudlaModuleTensorDescriptor& desc = m_processingState[0].m_segmentationExec->outputTensorDescriptor(0);

    assert(desc.dataType == CUDLA_DATA_TYPE_HALF);

    // Ensure output width and height match
    assert(m_segInputWidth == desc.w);
    assert(m_segInputHeight == desc.h);

    // Output should have 1 channel and batch size=1
    assert(1 == desc.c);
    assert(1 == desc.n);

    m_segOutputRowPitchElements = desc.stride[1] / desc.stride[0];
    m_segOutputPlanePitchElements = desc.stride[2] / desc.stride[0];

    printf("Segmentation output row pitch is %u elements, plane pitch is %u elements\n", m_segOutputRowPitchElements, m_segOutputPlanePitchElements);
  }

  {
    // ROI
    const cudlaModuleTensorDescriptor& desc = m_processingState[0].m_roiExec->outputTensorDescriptor(0);

    // Only support same input and output types
    assert(m_processingState[0].m_roiExec->inputTensorDescriptor(0).dataType == desc.dataType);

    if (desc.dataType == CUDLA_DATA_TYPE_INT8) {
      m_roiIOIsInt8 = true;
    } else if (desc.dataType == CUDLA_DATA_TYPE_HALF) {
      m_roiIOIsInt8 = false;
    } else {
      assert(false && "ROI engine: unsupported I/O type (must be kHALF or kINT8)");
    }

    // Output W/H depends on network config
    m_roiOutputWidth = desc.w;
    m_roiOutputHeight = desc.h;
    // Output should have 1 channel and batch size=1
    assert(1 == desc.c);
    assert(1 == desc.n);

    m_roiOutputRowStrideElements = desc.stride[1] / desc.stride[0];

    printf("ROI Output is %ux%u. Row pitch is %u elements\n", m_roiOutputWidth, m_roiOutputHeight, m_roiOutputRowStrideElements);
  }

  PER_EYE {
    // Pre-create the pupil mask mat
    ProcessingState& ps = m_processingState[eyeIdx];
    ps.m_pupilMask.create(m_segInputHeight, m_segInputWidth, CV_8UC1);
  }

  applyCalibrationData();
}

EyeTrackingService::~EyeTrackingService() {
  PER_EYE {
    m_processingState[eyeIdx].shutdownThread();
  }

}

bool EyeTrackingService::loadCalibrationData() {

  cv::FileStorage fs(calibrationFilename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("Unable to open calibration data file %s\n", calibrationFilename);
    return false;
  }

  return loadCalibrationData(fs);
}

void EyeTrackingService::saveCalibrationData() {
  cv::FileStorage fs(calibrationFilename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  saveCalibrationData(fs);
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)
bool EyeTrackingService::loadCalibrationData(cv::FileStorage& fs) {
  try {
    readNode(fs, focalLength);
    readNode(fs, pixelPitchMicrons);
    readNode(fs, eyeZ);
    cv::read(fs["rollOffsetL"], m_rollOffsetDeg[0], m_rollOffsetDeg[0]);
    cv::read(fs["rollOffsetR"], m_rollOffsetDeg[1], m_rollOffsetDeg[1]);
    readNode(fs, filterMinCutoff);
    readNode(fs, filterBetaExponent);
    readNode(fs, filterDCutoff);
    readNode(fs, hideCrosshairAfterFrameCount);
    readNode(fs, showCrosshairAfterFrameCount);
    cv::read(fs["leftEyeCamera"], m_processingState[0].m_cameraDeviceName, m_processingState[0].m_cameraDeviceName);
    cv::read(fs["rightEyeCamera"], m_processingState[1].m_cameraDeviceName, m_processingState[1].m_cameraDeviceName);
    readNode(fs, blinkDetectMinFrames);
    readNode(fs, blinkDetectMaxFrames);
    readNode(fs, blinkCooldownFrames);

  } catch (const std::exception& ex) {
    printf("Unable to load calibration data: %s\n", ex.what());
    return false;
  }
  return true;
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void EyeTrackingService::saveCalibrationData(cv::FileStorage& fs) {
  writeNode(fs, focalLength);
  writeNode(fs, pixelPitchMicrons);
  writeNode(fs, eyeZ);
  fs.write("rollOffsetL", m_rollOffsetDeg[0]);
  fs.write("rollOffsetR", m_rollOffsetDeg[1]);
  writeNode(fs, filterMinCutoff);
  writeNode(fs, filterBetaExponent);
  writeNode(fs, filterDCutoff);
  writeNode(fs, hideCrosshairAfterFrameCount);
  writeNode(fs, showCrosshairAfterFrameCount);
  fs.write("leftEyeCamera", m_processingState[0].m_cameraDeviceName);
  fs.write("rightEyeCamera", m_processingState[1].m_cameraDeviceName);
  writeNode(fs, blinkDetectMinFrames);
  writeNode(fs, blinkDetectMaxFrames);
  writeNode(fs, blinkCooldownFrames);

}
#undef writeNode


void EyeTrackingService::ProcessingState::postprocessOneEye() {
  const uint32_t kContiguousValidFrameCountThreshold = 100;
  const uint32_t kInvalidFrameCountThreshold = 2000; // Number of invalid frames in a row to throw away the calibration and start over

  // Trigger/flag set by the main thread
  if (m_shouldClearCalibration) {
    printf("EyeTrackingService::postprocessOneEye(%zu): calibration reset requested\n", m_eyeIdx);
    m_shouldClearCalibration = false;

    // Wipe calibration state and start over
    m_calibrationState = kWaitingForValidFrames;
    m_calibrationSamples.clear();
    m_contiguousValidFrameCounter = 0;
    m_contiguousInvalidFrameCounter = 0;
    m_lastInvalidFrameRunLength = 0;
  }

  bool fitEllipse = postprocessOneEye_fitEllipse();

  // Update counters for calibration state machine
  if (fitEllipse) {
    m_contiguousValidFrameCounter += 1;
    if (m_contiguousInvalidFrameCounter > 0) {
      m_lastInvalidFrameRunLength = m_contiguousInvalidFrameCounter;
    }
    m_contiguousInvalidFrameCounter = 0;
  } else {
    m_contiguousValidFrameCounter = 0;
    m_contiguousInvalidFrameCounter += 1;
  }

  if (m_shouldShowCrosshair) {
    // Currently showing, check if we should hide the crosshair
    if (m_contiguousValidFrameCounter > m_service->m_hideCrosshairAfterFrameCount)
      m_shouldShowCrosshair = false;
  } else {
    // Currently hidden, check if we should show the crosshair
    if (m_contiguousInvalidFrameCounter > m_service->m_showCrosshairAfterFrameCount)
      m_shouldShowCrosshair = true;
  }

  switch (m_calibrationState) {
    case kWaitingForValidFrames: {
      if (m_contiguousValidFrameCounter > kContiguousValidFrameCountThreshold) {
        printf("EyeTrackingService::postprocessOneEye(%zu): waiting -> centering\n", m_eyeIdx);
        m_calibrationSamples.clear();
        m_calibrationState = kCentering;
      }
    } break;

    case kCentering: {
      // Add valid samples to the scrolling buffer.
      // We throw away the first couple samples after a run of invalid samples to reduce noise
      if (fitEllipse && m_contiguousValidFrameCounter >= 3) {
        m_calibrationSamples.push_back(m_pupilEllipse);

        if (m_calibrationSamples.full()) {
          // Collected enough samples, see if they're consistent enough
          glm::vec2 mean(0.0f, 0.0f);
          for (size_t i = 0; i < m_calibrationSamples.size(); ++i) {
            mean.x += m_calibrationSamples[i].center.x;
            mean.y += m_calibrationSamples[i].center.y;
          }
          mean /= static_cast<float>(m_calibrationSamples.size());

          float maxDeviation = 0.0f;

          float minDeviation = FLT_MAX;
          size_t minDeviationIdx = 0;

          for (size_t i = 0; i < m_calibrationSamples.size(); ++i) {
            glm::vec2 p = glm::vec2(m_calibrationSamples[i].center.x, m_calibrationSamples[i].center.y);
            float lp = glm::length(p - mean);
            maxDeviation = std::max<float>(lp, maxDeviation);

            if (lp < minDeviation) {
              minDeviation = lp;
              minDeviationIdx = i;
            }
          }

          //printf("Center calibration: %zu samples, mean = {%.3f, %.3f}, deviation range = [%.3f, %.3f]\n",
          //  m_calibrationSamples.size(), mean.x, mean.y, minDeviation, maxDeviation);

          // TODO adjust threshold
          if (maxDeviation < 3.0f) {
            printf("Center calibration: deviation below threshold, accepting calibration. %zu samples, mean = {%.3f, %.3f}, deviation range = [%.3f, %.3f]\n",
              m_calibrationSamples.size(), mean.x, mean.y, minDeviation, maxDeviation);
            m_centerCalibrationSample = m_calibrationSamples[minDeviationIdx];


            // Clear the eye fitter model
            m_eyeModelFitter.reset();
            m_eyeFitterSamples.clear();

            // Move to 'calibrated' state.
            // We still need to rebuild the eye fitter model, but that'll happen over time as the user looks around.
            m_calibrationState = kCalibrated;

          }
        }
      }
    } break;

    case kCalibrated: {
      if (m_contiguousInvalidFrameCounter > kInvalidFrameCountThreshold) {
        printf("EyeTrackingService::postprocessOneEye(%zu): Could not fit pupil for %u frames, resetting calibration.\n", m_eyeIdx, m_contiguousInvalidFrameCounter);
        m_calibrationState = kWaitingForValidFrames;
      }

    } break;

    default:
      printf("EyeTrackingService::postprocessOneEye(%zu): invalid calibration state %u\n", m_eyeIdx, m_calibrationState);
      m_calibrationState = kWaitingForValidFrames;
      break;
  };

}


// std::sort support
namespace cv {
  bool operator<(const Point& left, const Point& right) {
    if (left.x < right.x)
      return true;
    else if (left.x > right.x)
      return false;
    else /*(left.x == right.x)*/
      return left.y < right.y;
  }
};


// Returns true if an ellipse was fit this frame, even if the full eye model fit/unproject wasn't successful.
bool EyeTrackingService::ProcessingState::postprocessOneEye_fitEllipse() {

  // Find contours in the pupil mask
  struct Contour {
    std::vector<cv::Point> points;
    float area;
    float perimeter;
  };
  std::vector<Contour> filteredContours;


  // Collect stats and filter contours
  {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContoursLinkRuns(m_pupilMask, contours);


    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx) {
      auto& points = contours[contourIdx];
      if (points.size() < 20)
        continue; // Not enough points

      Contour c;
      c.area = fabs(cv::contourArea(points));
      if (c.area < 1000.0f)
        continue; // Not enough area

      c.perimeter = cv::arcLength(points, /*is_closed=*/ true);

      c.points = std::move(points);
      filteredContours.push_back(std::move(c));
    }

    // Sort filtered contours by area descending
    if (filteredContours.size() > 1)
      std::sort(filteredContours.begin(), filteredContours.end(), [](const Contour& left, const Contour& right) { return left.area > right.area; } );
  }

  bool didFitEllipse = false;
  std::vector<cv::Point2f> transformedContour;

  for (size_t contourIdx = 0; contourIdx < filteredContours.size(); ++contourIdx) {
    Contour& contour = filteredContours[contourIdx];
    FRAME_DEBUG_LOG("Contour %zu/%zu: area = %f, perimeter = %f, %zu points\n", contourIdx, filteredContours.size(), contour.area, contour.perimeter, contour.points.size());

    // Circularity
    // 75% circularity seems to be a good threshold for valid ellipses even at extreme angles
    float circularity = (4.0f * M_PI * contour.area) / (contour.perimeter * contour.perimeter);
    if (fabs(1.0f - circularity) > 0.25f) {
      FRAME_DEBUG_LOG(" -- Failed on circularity test (ratio %.3f)\n", circularity);
      continue;
    }

    // Convexity
    std::vector<cv::Point> hull;
    cv::convexHull(contour.points, hull);
    float convexHullArea = fabs(cv::contourArea(hull));
    if (convexHullArea < 1.0f) {
      FRAME_DEBUG_LOG(" -- Failed to fit a convex hull\n");
      continue;
    }

    float convexity = contour.area / convexHullArea;
    if (fabs(1.0f - convexity) > 0.10f) {
      FRAME_DEBUG_LOG(" -- Failed on convexity test (ratio %.3f)\n", convexity);
      continue;
    }

    // Tests passed

    // Fit only to contour points that exist in the convex hull
    // Avoids jitter with edge cutouts for glints, but hopefully without skewing the fit
    // to try and include the straight edges generated during the convex hull operation.

    // Sort contour and hull points
    std::sort(hull.begin(), hull.end());
    std::sort(contour.points.begin(), contour.points.end());

    // Apply filtering
    std::vector<cv::Point> filteredPoints;
    filteredPoints.reserve(contour.points.size());

    std::set_intersection(contour.points.begin(), contour.points.end(), hull.begin(), hull.end(), std::back_inserter(filteredPoints));

    if (filteredPoints.size() < 10) {
      FRAME_DEBUG_LOG("  -- Insufficient points (%zu) after convex hull filtering\n", filteredPoints.size());
      continue; // Not enough points
    }

    // Replace contour points with filtered points
    filteredPoints.swap(contour.points);

    // Find center of points for sector angle filtering
    glm::vec2 boundsCenter = boundsCenterFromPoints(contour.points);

    // Save bounds center for sector cutoff gizmo drawing
    m_debugBoundsCenter = boundsCenter + glm::vec2(m_lastSegROIToCaptureMatOffset.x, m_lastSegROIToCaptureMatOffset.y);

    // Vertical vector and cutoff angle
    glm::vec2 verticalVec = vec2AtAngleDeg(m_service->m_rollOffsetDeg[m_eyeIdx] + 90.0f);
    float cosAngleCutoff = cosf(glm::radians(m_service->m_sectorCutoffAngleDeg));

    // Transform contour points from ROI into full image space prior to ellipse fitting.
    // (easier to transform points than the ellipse equation, given that the scale can be nonuniform)

    transformedContour.reserve(contour.points.size());
    for (size_t i = 0; i < contour.points.size(); ++i) {
      cv::Point2f np = contour.points[i];

      // Sector-angle cutoff filter
      float cosAngle = glm::abs(glm::dot(glm::normalize(toGlm(np) - boundsCenter), verticalVec));
      if (cosAngle > cosAngleCutoff)
        continue; // Skip point -- failed sector cutoff check

      // Add ROI offset
      np.x += static_cast<float>(m_lastSegROIToCaptureMatOffset.x);
      np.y += static_cast<float>(m_lastSegROIToCaptureMatOffset.y);

      transformedContour.push_back(np);
    }

    // Minimum of 6 points is required to fit an ellipse. Ensure that we have enough points after cutoff filtering
    if (transformedContour.size() < 6) {
      transformedContour.clear(); // discard failed points for next attempt
      continue;
    }

    cv::RotatedRect ell = cv::fitEllipse(transformedContour);

    // Update debug deltas if there was a previous ellipse fit
    if (!m_pupilEllipse.size.empty() && !m_freezeGraphData) {
      ProcessingState::GraphData gd;

      gd.deltaSize = (ell.size.area() / m_pupilEllipse.size.area()) * 100.0f; // percentage
      gd.deltaCenter = glm::length(toGlm(ell.center) - toGlm(m_pupilEllipse.center));
      gd.deltaAngle = ell.angle - m_pupilEllipse.angle;

      m_graphData.push_back(gd);
    }

    m_pupilEllipse = ell;
    m_debugTransformedContour = transformedContour;
    didFitEllipse = true;

    // Only need to fit one ellipse.
    break;
  }

  if (!didFitEllipse) {
    // Clear pupil ellipse, since we didn't find anything useful this round.
    m_pupilEllipse = cv::RotatedRect();
    m_eyeFitterOutputsValid = false;

    // No valid ellipse this frame.
    return false;
  }


  // See if we need to add more samples to the model
  if (m_eyeModelFitter.pupils.size() < 50) {
    bool isNovelSample = true;
    if (m_eyeModelFitter.pupils.size()) {
      float minDist = FLT_MAX;
      for (const auto& pupilSample : m_eyeModelFitter.pupils) {
        cv::Point2f delta = cv::Point2f(
          m_pupilEllipse.center.x - (pupilSample.observation.ellipse.centre[0] + m_captureCenterOffset.x),
          m_pupilEllipse.center.y - (pupilSample.observation.ellipse.centre[1] + m_captureCenterOffset.y));
        float dist = sqrtf((delta.x * delta.x) + (delta.y * delta.y));
        minDist = std::min<float>(minDist, dist);
      }
      // FRAME_DEBUG_LOG("Min sample distance = %.3f\n", minDist);
      isNovelSample = (minDist > 3.0f);
    }

    if (isNovelSample) {
      // Add this observation to the eye model fitter
      m_eyeFitterSamples.push_back(m_pupilEllipse);

      // pupil_inliers needs to be in camera space, so apply captureCenterOffset
      std::vector<cv::Point2f> pupil_inliers;
      pupil_inliers.resize(transformedContour.size());
      for (size_t i = 0; i < pupil_inliers.size(); ++i) {
        pupil_inliers[i] = transformedContour[i] - m_captureCenterOffset;
      }

      // The eyefitter works in a coordinate system where the center of the image is at (0, 0),
      // so we need to offset the ellipse center coordinate (via toEllipseWithOffset)

      m_eyeModelFitter.add_observation(
        /*image (unused)=*/cv::Mat(),
        /*pupil=*/ singleeyefitter::toEllipseWithOffset<double>(m_pupilEllipse, m_captureCenterOffset),
        /*inliers=*/ pupil_inliers);

      // Try and fit the model
      if (m_eyeModelFitter.pupils.size() > 20) {
        FRAME_DEBUG_LOG("Attempting eye model fit. m_eyeFitterSamples.size()=%zu m_eyeModelFitter.pupils.size()=%zu\n",
          m_eyeFitterSamples.size(), m_eyeModelFitter.pupils.size());

        if (m_eyeModelFitter.unproject_observations(m_service->pupilRadius(), m_service->initialEyeZ())) {
          m_eyeModelFitter.initialise_model();
          m_eyeModelFitter.refine_with_inliers();


          // Use the center calibration sample to find angle offsets
          if (m_eyeModelFitter.unproject_single_observation(m_centerPupilCircle, singleeyefitter::toEllipseWithOffset<double>(m_centerCalibrationSample, m_captureCenterOffset), m_service->pupilRadius())) {
            glm::vec3 pupil = centerPupilNormal();

            vectorToAngles(glm::value_ptr(pupil), m_centerPitchDeg, m_centerYawDeg, /*toDegrees=*/ true);

            printf("Center calibration sample pitch=%.3f yaw=%.3f (n=%.3f %.3f %.3f)\n",
              m_centerPitchDeg, m_centerYawDeg,
              pupil.x, pupil.y, pupil.z);
          } else {
            printf("Center calibration sample invalid!\n");
            // TODO: try and recover from this?
          }

        } else {
          FRAME_DEBUG_LOG("Eye model fit failed; unproject_observations() returned false.\n");
        }
      }
    }
  }

  // Apply model to ellipse to generate 3d fit
  // Skip a few frames after an invalid one to avoid jitter during blinking
  if (m_eyeModelFitter.hasEyeModel() && m_contiguousValidFrameCounter > 4) {
    m_eyeFitterOutputsValid = m_eyeModelFitter.unproject_single_observation(m_fitPupilCircle, singleeyefitter::toEllipseWithOffset<double>(m_pupilEllipse, m_captureCenterOffset), m_service->pupilRadius());
    if (m_eyeFitterOutputsValid) {
      // Original coordinate system:
      // +x is left
      // -y is up
      // -z is forward

      glm::vec3 pupil = fitPupilNormal();
      vectorToAngles(glm::value_ptr(pupil), m_pupilRawPitchDeg, m_pupilRawYawDeg, /*toDegrees=*/ true);

      FRAME_DEBUG_LOG("n=%.3f %.3f %.3f\n", pupil.x, pupil.y, pupil.z);
      FRAME_DEBUG_LOG("pitch = %.3f, yaw = %.3f\n", m_pupilRawPitchDeg, m_pupilRawYawDeg);

      // Add sample to filter
      double ts = static_cast<double>(currentTimeNs() / 1000ULL) / 1'000'000.0;
      m_pupilFilteredPitchDeg = m_pitchFilter(m_pupilRawPitchDeg, ts);
      m_pupilFilteredYawDeg = m_yawFilter(m_pupilRawYawDeg, ts);
    }
  } else {
    m_eyeFitterOutputsValid = false;
  }

  // Had a valid ellipse observation this frame
  return true;
}


void polyline(cv::Mat& img, const cv::Point2f* points, size_t startIdx, size_t endIdx, const cv::Scalar& color, bool loop = false) {
  for (size_t i = startIdx; i < endIdx; ++i) {
    cv::line(img, points[i], points[i + 1], color);
  }
  if (loop)
    cv::line(img, points[endIdx], points[startIdx], color);
}

void lineCenterDirectionLength(cv::Mat& img, const glm::vec2& center, const glm::vec2& direction, float length, const cv::Scalar& color, bool bidirectional = true) {
  glm::vec2 v1 = center + (direction * length);
  glm::vec2 v2 = bidirectional ? (center - (direction * length)) : center;
  cv::line(img, cv::Point(v1.x, v1.y), cv::Point(v2.x, v2.y), color);
}

cv::Point2f toImgCoord(const cv::Point2f& point, const cv::Point2f& centerOffset) {
  return point + centerOffset;
}
cv::Point toImgCoord(const cv::Point& point, const cv::Point2f& centerOffset) {
    return cv::Point(
        static_cast<int>(centerOffset.x) + point.x,
        static_cast<int>(centerOffset.y) + point.y);
}
cv::RotatedRect toImgCoord(const cv::RotatedRect& rect, const cv::Point2f& centerOffset) {
    return cv::RotatedRect(toImgCoord(rect.center, centerOffset),
        cv::Size2f(rect.size.width, rect.size.height), rect.angle);
}

EyeTrackingService::ProcessingState::~ProcessingState() {

}

void EyeTrackingService::ProcessingState::internalUpdateStateOnCaptureOpen() {
  // Compute capture mat crop rect based on the ROI network input size
  {
    uint32_t targetStreamWidth = (m_capture.streamWidth() / m_service->m_roiInputWidth) * m_service->m_roiInputWidth;
    uint32_t targetStreamHeight = (m_capture.streamHeight() / m_service->m_roiInputHeight) * m_service->m_roiInputHeight;
    uint32_t cropOffsetX = (m_capture.streamWidth() - targetStreamWidth) / 2;
    uint32_t cropOffsetY = (m_capture.streamHeight() - targetStreamHeight) / 2;
    m_captureCropRect = cv::Rect(cropOffsetX, cropOffsetY, targetStreamWidth, targetStreamHeight);
    printf("EyeTrackingService::ProcessingState::internalUpdateStateOnCaptureOpen(%zu): Capture dimensions are %ux%u, crop is (%ux%u @ %u,%u)\n", m_eyeIdx,
      m_capture.streamWidth(), m_capture.streamHeight(),
      targetStreamWidth, targetStreamHeight,
      cropOffsetX, cropOffsetY);

    // Ensure that the ROI input scaling is what we expect (for the vectorized 1/6 area resize)
    assert(m_service->m_roiInputWidth == (targetStreamWidth / 6));
    assert(m_service->m_roiInputHeight == (targetStreamHeight / 6));
  }

  // Update the capture center offset, now that we know the frame dimensions.
  m_captureCenterOffset = cv::Point2f(
    static_cast<float>(m_captureCropRect.width) / 2.0f,
    static_cast<float>(m_captureCropRect.height) / 2.0f);


  // Pre-create some mats
  m_roiMaskMat.create(m_service->m_roiOutputHeight, m_service->m_roiOutputWidth, CV_8UC1);
  m_roiDilatedMaskMat.create(m_service->m_roiOutputHeight, m_service->m_roiOutputWidth, CV_8UC1);
  m_roiScaleMat.create(m_service->m_roiInputHeight, m_service->m_roiInputWidth, CV_8UC1);
}


void EyeTrackingService::ProcessingState::internalProcessOneCapture() {
  PerfTimer perfTimer;

  // Extract crop region from the luma plane
  cv::Mat captureMat = cv::Mat(m_capture.lumaPlane(), m_captureCropRect);

  // Scale from capture mat to the input size for the ROI prediction network
  // This vectorized downsample runs about the same speed as cv::resize(..., cv::INTER_LINEAR),
  // but implements a full 6x6 pixel box filter (equivalent quality to the much slower cv::INTER_AREA)
  areaDownsample6x6(captureMat.ptr<uint8_t>(), captureMat.step, m_roiScaleMat.ptr<uint8_t>(), m_roiScaleMat.cols, m_roiScaleMat.rows, m_roiScaleMat.step);

  // Convert CV_U8 from the scale output to int8 or half for the ROI network input
  if (m_service->m_roiIOIsInt8) {
    int8_t* roiInputTensor = m_roiExec->inputTensorPtr<int8_t>(0);
    if (m_service->m_roiInputRowStrideElements == m_service->m_roiInputWidth) {
      // Tightly-packed input, convert entire block at once
        convertUnorm8ToDLAInt8(m_roiScaleMat.ptr<uint8_t>(), roiInputTensor, m_service->m_roiInputWidth * m_service->m_roiInputHeight);
    } else {
      // Padded input, convert a row at a time
      for (size_t row = 0; row < m_service->m_roiInputHeight; ++row) {
        convertUnorm8ToDLAInt8(m_roiScaleMat.ptr<uint8_t>(row), roiInputTensor + (m_service->m_roiInputRowStrideElements * row), m_service->m_roiInputWidth);
      }
    }
  } else {
    _Float16* roiInputTensor = m_roiExec->inputTensorPtr<_Float16>(0);
    if (m_service->m_roiInputRowStrideElements == m_service->m_roiInputWidth) {
      // Tightly-packed input, convert entire block at once
      convertUnorm8ToSnormFp16(m_roiScaleMat.ptr<uint8_t>(), roiInputTensor, m_service->m_roiInputWidth * m_service->m_roiInputHeight);
    } else {
      // Padded input, convert a row at a time
      for (size_t row = 0; row < m_service->m_roiInputHeight; ++row) {
        convertUnorm8ToSnormFp16(m_roiScaleMat.ptr<uint8_t>(row), roiInputTensor + (m_service->m_roiInputRowStrideElements * row), m_service->m_roiInputWidth);
      }
    }
  }

  m_lastFramePreProcessingTimeMs = perfTimer.checkpoint();

  // Run ROI network
  m_roiExec->runInference();

  m_lastFrameROITimeMs = perfTimer.checkpoint();

  // Process ROI network output into binary mask
  float roiOutput[2]; // 0...1 coordinate range
  float roiSampleThreshold;

  if (m_service->m_roiIOIsInt8) {
    assert(false && "TODO: ROI mask processing for int8 i/o");
  } else {
    _Float16* roiBasePtr = m_roiExec->outputTensorPtr<_Float16>(0);

    // Weighted sampling:

    // Find the max value
    _Float16 maxValue = 0.0f;
    for (uint32_t y = 0; y < m_service->m_roiOutputHeight; ++y) {
      _Float16* roiRowPtr = roiBasePtr + (y * m_service->m_roiOutputRowStrideElements);
      _Float16 rowMax = fp16VectorMax(roiRowPtr, m_service->m_roiOutputWidth);
      maxValue = std::max<_Float16>(maxValue, rowMax);
    }

    // Convert sample points >= 0.9x max value into a binary mask

    _Float16 roiSampleThreshold_fp16 = maxValue * 0.9f16;
    roiSampleThreshold = roiSampleThreshold_fp16;

    for (uint32_t y = 0; y < m_service->m_roiOutputHeight; ++y) {
      _Float16* roiRowPtr = roiBasePtr + (y * m_service->m_roiOutputRowStrideElements);
      fp16ThresholdToU8Mask(roiRowPtr, roiSampleThreshold_fp16, m_roiMaskMat.ptr<uint8_t>(y), m_service->m_roiOutputWidth);
    }
  }

  // Cleanup m_roiMaskMat, generate contours, find the best match
  {
    // Run dilate-erode operation to close borders
    cv::dilate(/*src=*/ m_roiMaskMat, /*dst=*/ m_roiDilatedMaskMat, /*kernel (default)=*/ cv::Mat());
    cv::erode(/*src=*/ m_roiDilatedMaskMat, /*dst=*/ m_roiMaskMat, /*kernel (default)=*/ cv::Mat());

    // Collect and filter contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContoursLinkRuns(m_roiMaskMat, contours);

    struct Contour {
      std::vector<cv::Point> points;
      glm::vec2 normalizedBoundsCenter;
      float distanceToCenter;
      float area;
    };
    std::vector<Contour> filteredContours;
    filteredContours.reserve(contours.size());

    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx) {
      Contour c;

      c.points = std::move(contours[contourIdx]);
      c.normalizedBoundsCenter = boundsCenterFromPoints(c.points) / glm::vec2(m_service->m_roiOutputWidth, m_service->m_roiOutputHeight);
      c.distanceToCenter = glm::length(c.normalizedBoundsCenter - glm::vec2(0.5f, 0.5f));
      c.area = fabs(cv::contourArea(c.points));

      filteredContours.push_back(std::move(c));
    }

    if (filteredContours.size() > 1) {
      // Sort filtered contours by distance to center ascending
      // std::sort(filteredContours.begin(), filteredContours.end(), [](const Contour& left, const Contour& right) { return left.distanceToCenter < right.distanceToCenter; } );
      // Sort filtered contours by area descending
      std::sort(filteredContours.begin(), filteredContours.end(), [](const Contour& left, const Contour& right) { return left.area > right.area; } );
    }

    if (!filteredContours.empty()) {
      // Use the bounds-center of the highest-ranked contour

      roiOutput[0] = filteredContours[0].normalizedBoundsCenter[0];
      roiOutput[1] = filteredContours[0].normalizedBoundsCenter[1];
    } else {
      // No samples? Default to center of image.
      roiOutput[0] = 0.5f;
      roiOutput[1] = 0.5f;
    }
  } // ROI center computation

  // Use center coordinates computed from the ROI network to compute the ROI aligned inside the original capture mat
  // ROI Rect needs to be fixed size of the segmentation network input (no edge clipping allowed)

  FRAME_DEBUG_LOG("ROI computed center (0...1): (%f, %f)\n", roiOutput[0], roiOutput[1]);

  // Rescale to 0...1f and multiply by the actual source w/h
  cv::Point2i roiCenter_captureRelative = cv::Point2i(
    clamp<int32_t>(roiOutput[0] * static_cast<float>(captureMat.cols - 1), 0, captureMat.cols - 1),
    clamp<int32_t>(roiOutput[1] * static_cast<float>(captureMat.rows - 1), 0, captureMat.rows - 1)
  );

  // Clip the capture-relative ROI center to the capture dimensions inset by half of the segmentation network input size. This should ensure that the
  // segmentation ROI rect fits entirely within the capture region.

  roiCenter_captureRelative.x = clamp<int32_t>(roiCenter_captureRelative.x, (m_service->m_segInputWidth / 2), (captureMat.cols - 1) - (m_service->m_segInputWidth / 2));
  roiCenter_captureRelative.y = clamp<int32_t>(roiCenter_captureRelative.y, (m_service->m_segInputHeight / 2), (captureMat.rows - 1) - (m_service->m_segInputHeight / 2));

  // Build segmentation input crop rect in capture mat coordinates
  cv::Point2i segROIRect_tl = (roiCenter_captureRelative - cv::Point2i(m_service->m_segInputWidth / 2, m_service->m_segInputHeight / 2));
  // Just in case, clamp the top left corner so it doesn't go negative
  segROIRect_tl.x = std::max<int32_t>(segROIRect_tl.x, 0);
  segROIRect_tl.y = std::max<int32_t>(segROIRect_tl.y, 0);

  cv::Rect segROIRect = cv::Rect(segROIRect_tl, cv::Size(m_service->m_segInputWidth, m_service->m_segInputHeight));
  m_lastSegROIToCaptureMatOffset = segROIRect_tl; // save for eyefitter processing

  cv::Mat segROIMat = cv::Mat(captureMat, segROIRect);

  // Convert u8 pixels in ROI window to snorm fp16 to populate m_segInputTensor
  // The ROI input is known not to be contiguous, so we do it row-by-row.
  for (size_t y = 0; y < m_service->m_segInputHeight; ++y) {
    convertUnorm8ToSnormFp16(segROIMat.ptr<uint8_t>(y, 0), m_segmentationExec->inputTensorPtr<_Float16>(0) + (y * m_service->m_segInputRowStrideElements), m_service->m_segInputWidth);
  }

  m_lastFrameROIToSegmentationTimeMs = perfTimer.checkpoint();

  // Launch segmentation network.
  m_segmentationExec->asyncStartInference();


  // Cache this flag since it may be written async on the main thread.
  bool populateDebugView = m_service->m_debugShowFeedbackView;

  if (populateDebugView) {
    // While the DLA is running the segmentation network (~4+ ms), convert the capture buffer to RGBA in preparation for debug drawing
    m_tempRGBDebugMat.create(captureMat.size(), CV_8UC4);
    for (size_t y = 0; y < captureMat.rows; ++y) {
      convertGrayToRGBA(captureMat.ptr<uint8_t>(y), m_tempRGBDebugMat.ptr<uint8_t>(y), captureMat.cols);
    }
  }

  // Wait for segmentation network to finish
  m_segmentationExec->asyncFinishInference();

  m_lastFrameSegmentationTimeMs = perfTimer.checkpoint();

  // Postprocess network results: run threshold operation to create binary mask
  for (size_t y = 0; y < m_service->m_segInputHeight; ++y) {
    fp16ThresholdToU8Mask(m_segmentationExec->outputTensorPtr<_Float16>(0) + (y * m_service->m_segOutputRowPitchElements), 0.5f16, m_pupilMask.ptr<uint8_t>(y), m_service->m_segInputWidth);
  }

  // TODO: May still need to run closure operation to fill holes in the mask -- 3x3 dilation followed by 3x3 erosion


  // Eye-fitter postprocessing

  postprocessOneEye();

  m_lastFramePostProcessingTimeMs = perfTimer.checkpoint();
  if (m_lastFramePostProcessingTimeMs > 0.25f) {
    FRAME_DEBUG_LOG("Eye %zu CPU postprocess took %.3fms\n", m_eyeIdx, m_lastFramePostProcessingTimeMs);
  }

  if (m_contiguousInvalidFrameCounter && m_calibrationState == kCalibrated) {
    // Invalid fit. See if we should save this frame for later inclusion in training data.
    if (m_service->m_debugSaveBadFitImages && (m_lastDebugBadFitCaptureTimestampMs + m_service->m_debugSaveBadFitIntervalMs) < currentRealTimeMs()) {
      uint64_t timestamp = currentRealTimeMs();
      char fnbuf[256];
      snprintf(fnbuf, 255, "%s/%lu_%zu.png", kBadFitCaptureDirName, timestamp, m_eyeIdx);
      fnbuf[255] = '\0';

      // Async save full original frame
      FxThreading::runTaskAsync([eyeIdx = m_eyeIdx, filename = std::string(fnbuf), fullCap = m_capture.lumaPlane().clone()]() {
        if (mkdir(kBadFitCaptureDirName, 0777) < 0) {
          // EEXIST is ok, anything else is fatal
          if (errno != EEXIST) {
            printf("EyeTrackingService::eyeProcessingThreadFn(%zu): can't create bad-fit capture directory \"%s\": %s\n", eyeIdx, kBadFitCaptureDirName, strerror(errno));
            return;
          }
        }

        if (stbi_write_png(filename.c_str(), fullCap.cols, fullCap.rows, /*components=*/ fullCap.channels(), fullCap.ptr(), /*rowBytes=*/ fullCap.step)) {
          printf("EyeTrackingService::eyeProcessingThreadFn(%zu): wrote bad-fit capture to file %s\n", eyeIdx, filename.c_str());
        } else {
          printf("EyeTrackingService::eyeProcessingThreadFn(%zu): failed to write bad-fit capture to file %s\n", eyeIdx, filename.c_str());
        }
      });

      m_lastDebugBadFitCaptureTimestampMs = timestamp;
    }
  }


  // Update debug view
  if (populateDebugView) {
    if (m_service->m_debugDrawOverlays) {
#if 1
      {
        // Draw ROI scale view
        const int inset = 128; // Inset into the dest mat so it's easier to see

        cv::rectangle(m_tempRGBDebugMat, cv::Point(inset - 1, inset - 1), cv::Point(inset + 1 + m_roiScaleMat.cols, inset + 1 + m_roiScaleMat.rows), cv::Scalar(0, 255, 0));
        for (size_t row = 0; row < m_roiScaleMat.rows; ++row) {
          for (size_t col = 0; col < m_roiScaleMat.cols; ++col) {
            uint8_t roiVal = m_roiScaleMat.at<uint8_t>(row, col);

            uint8_t* dp = m_tempRGBDebugMat.ptr<uint8_t>(row + inset, col + inset);
            dp[0] = roiVal;
            dp[1] = roiVal;
            dp[2] = roiVal;
          }
        }

        // Draw ROI output
        if (!m_service->m_roiIOIsInt8) {
          _Float16* roiBasePtr = m_roiExec->outputTensorPtr<_Float16>(0);
          const int scale = 4;

          const int xOff = inset;
          const int yOff = inset + 8 + m_roiScaleMat.rows;

          cv::rectangle(m_tempRGBDebugMat, cv::Point(xOff - 1, yOff - 1), cv::Point(xOff + 1 + (m_service->m_roiOutputWidth * scale), yOff + 1 + (m_service->m_roiOutputHeight * scale)), cv::Scalar(0, 255, 255));
          for (uint32_t y = 0; y < m_service->m_roiOutputHeight; ++y) {
            _Float16* roiRowPtr = roiBasePtr + (y * m_service->m_roiOutputRowStrideElements);
            for (uint32_t x = 0; x < m_service->m_roiOutputWidth; ++x) {
              float fRoiVal = static_cast<float>(roiRowPtr[x]);
              uint8_t roiVal = static_cast<uint8_t>(fRoiVal * 255.0f);

              for (size_t r = 0; r < scale; ++r) {
                for (size_t c = 0; c < scale; ++c) {
                  uint8_t* dp = m_tempRGBDebugMat.ptr<uint8_t>(yOff + (y * scale) + r, xOff + (x * scale) + c);
                  if (fRoiVal > roiSampleThreshold) {
                    // ROI sample that was included in the center computation
                    dp[0] = 0; // R
                    dp[1] = roiVal; // G
                    dp[2] = 0; // B
                  } else {
                    // ROI sample that failed threshold test and was excluded
                    dp[0] = roiVal; // R
                    dp[1] = 0; // G
                    dp[2] = 0; // B
                  }
                }
              }
            }
          }
        }

      }

#endif

      // Segmentation ROI view of the RGB debug mat
      cv::Mat debugROIViewRGB = cv::Mat(m_tempRGBDebugMat, segROIRect);

#if 1
      // Draw segmentation mask colors
      for (size_t row = 0; row < debugROIViewRGB.rows; ++row) {
        uint8_t* pupilRowPtr = m_pupilMask.ptr<uint8_t>(row);
        for (size_t col = 0; col < debugROIViewRGB.cols; ++col) {
          if (pupilRowPtr[col])
            debugROIViewRGB.ptr<uint8_t>(row, col)[/*red channel=*/0] = 0xcc;
        }
      }
#endif

      // Now operating on the full view; coordinates that are relative to the ROI will need to be translated.

#if 0
      // Draw eye-fitter sample ellipses
      for (const auto& el : m_eyeFitterSamples) {
        cv::ellipse(m_tempRGBDebugMat, el, cv::Scalar(0x3f, 0, 0x3f), /*thickness=*/ 2);
      }
#endif

      // Draw pupil ellipse, if present
      if (!m_pupilEllipse.size.empty()) {
        cv::ellipse(m_tempRGBDebugMat, m_pupilEllipse, cv::Scalar(0xff, 0, 0xff), /*thickness=*/ 2);

        // Draw sector gizmo
        glm::vec2 verticalVec = vec2AtAngleDeg(m_service->m_rollOffsetDeg[m_eyeIdx] + 90.0f);
        glm::vec2 sector1Vec = vec2AtAngleDeg((m_service->m_rollOffsetDeg[m_eyeIdx] + 90.0f) + m_service->m_sectorCutoffAngleDeg);
        glm::vec2 sector2Vec = vec2AtAngleDeg((m_service->m_rollOffsetDeg[m_eyeIdx] + 90.0f) - m_service->m_sectorCutoffAngleDeg);

        lineCenterDirectionLength(m_tempRGBDebugMat, m_debugBoundsCenter, verticalVec, 80.0f, cv::Scalar(0, 0, 255), /*bidirectional=*/ true);
        lineCenterDirectionLength(m_tempRGBDebugMat, m_debugBoundsCenter, sector1Vec,  80.0f, cv::Scalar(255, 0, 0), /*bidirectional=*/ true);
        lineCenterDirectionLength(m_tempRGBDebugMat, m_debugBoundsCenter, sector2Vec,  80.0f, cv::Scalar(255, 0, 0), /*bidirectional=*/ true);
      }

      if (m_eyeFitterOutputsValid) {
        // Try-catch block avoids cv drawing functions crashing the app if we pass NaNs or something
        try {
          singleeyefitter::Conic<double> pupil_conic = singleeyefitter::project(m_fitPupilCircle, m_eyeModelFitter.focal_length);
          singleeyefitter::Ellipse2D<double> eye_ellipse = singleeyefitter::project(m_eyeModelFitter.eye, m_eyeModelFitter.focal_length);

          cv::RotatedRect pupilEllipseImg = toImgCoord(toRotatedRect(singleeyefitter::Ellipse2D<double>(pupil_conic)), m_captureCenterOffset);
          // pupil ellipse was already drawn above
          //cv::ellipse(m_tempRGBDebugMat, pupilEllipseImg, cv::Scalar(60, 60, 0), /*thickness=*/ 2);

          cv::RotatedRect eyeEllipseImg = toImgCoord(toRotatedRect(eye_ellipse), m_captureCenterOffset);
          cv::ellipse(m_tempRGBDebugMat, eyeEllipseImg, cv::Scalar(60, 60, 0), /*thickness=*/ 2);

          // order is _bottomLeft_, _topLeft_, topRight, bottomRight
          cv::Point2f rectPoints[4];
          eyeEllipseImg.points(rectPoints);

          // Draw crosshairs through the eye-ellipse
          cv::line(m_tempRGBDebugMat,
            (rectPoints[0] + rectPoints[1]) * 0.5f,
            (rectPoints[2] + rectPoints[3]) * 0.5f,
            cv::Scalar(60, 60, 0), /*thickness=*/2);

          cv::line(m_tempRGBDebugMat,
            (rectPoints[1] + rectPoints[2]) * 0.5f,
            (rectPoints[0] + rectPoints[3]) * 0.5f,
            cv::Scalar(60, 60, 0), /*thickness=*/2);

          // Draw a small marker on the eye center point
          cv::circle(m_tempRGBDebugMat, eyeEllipseImg.center, /*r=*/ 3, cv::Scalar(0, 0, 255), /*thickness=*/ -1);

          // Line from the eye center through the pupil center
          cv::line(m_tempRGBDebugMat, eyeEllipseImg.center, pupilEllipseImg.center, cv::Scalar(0, 255, 0), /*thickness=*/ 1);

        } catch (...) {}

#if 0
        char buf[64];
        snprintf(buf, 64, "n=%.3f %.3f %.3f", m_fitPupilCircle.normal[0], m_fitPupilCircle.normal[1], m_fitPupilCircle.normal[2]);
        cv::putText(m_tempRGBDebugMat, buf, cv::Point2f(/*x=*/ 5, /*y=*/ m_tempRGBDebugMat.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
#endif

        //FRAME_DEBUG_LOG("Ellipse: center=%.3f %.3f\n width=%.3f height=%.3f\n",
        //    m_pupilEllipse.center.x, m_pupilEllipse.center.y,
        //    m_pupilEllipse.size.width, m_pupilEllipse.size.height);

      }

      // Draw the ROI centroid
      cv::circle(m_tempRGBDebugMat, roiCenter_captureRelative, /*r=*/ 3, cv::Scalar(192, 0, 192));

      // Draw the ROI rectangle
      cv::rectangle(m_tempRGBDebugMat, segROIRect.tl(), segROIRect.br(), cv::Scalar(0, 255, 0));

    } // Debug overlay drawing

    // Swap debug mat with the one in processing state
    // (tries to avoid the main thread getting a partially-drawn debug mat)
    cv::swap(m_debugViewRGB, m_tempRGBDebugMat);

  } // Debug surface population

  m_lastFrameDebugViewTimeMs = perfTimer.checkpoint();

  m_lastFrameTotalProcessingTimeMs = perfTimer.totalElapsedTime();
}

bool EyeTrackingService::processFrame() {
  if (m_debugDisableProcessing)
    return false;

  FRAME_DEBUG_LOG("\x1b[2J\x1b[H"); // Clear terminal per frame

  // Processing thread maintenance:
  // Start threads if they're not running, clean up after exited threads

  PER_EYE {

    // Copy some flags from the service to the tracking thread base
    m_processingState[eyeIdx].m_debugFreezeCapture = m_debugFreezeCapture;

    m_processingState[eyeIdx].processFrameHook();
  }

  if (!(m_processingState[0].processingThreadAlive() || m_processingState[1].processingThreadAlive())) {
    // printf("EyeTrackingService: no processing threads alive\n");
    return false;
  }


  // Blink tracking
  bool triggerBlink = false;
  {
    bool havePupilLock = false;
    ++m_frameCounterSinceLastBlink;

    PER_EYE {
      if (m_processingState[eyeIdx].m_calibrationState == kCalibrated) {
        havePupilLock |= m_processingState[eyeIdx].m_eyeFitterOutputsValid;
      }
    }
    if (havePupilLock) {
      // Check for rising edge of pupil lock signal (where m_lastPupilLockFailureLengthFrames is nonzero).
      if (m_lastPupilLockFailureLengthFrames >= m_blinkDetectMinFrames && m_lastPupilLockFailureLengthFrames <= m_blinkDetectMaxFrames) {
        // Reacquired pupil lock and blink conditions match. Trigger blink if not within the cooldown period.
        if (m_frameCounterSinceLastBlink > m_blinkCooldownFrames) {
          triggerBlink = true;
          m_frameCounterSinceLastBlink = 0;
        }
      }
      m_lastPupilLockFailureLengthFrames = 0;
    } else {
      ++m_lastPupilLockFailureLengthFrames;
    }
  }

  // CANbus message transmission
  // TODO: This should transmit refined gaze vector from both eyes, once we implement dual-eye tracking.
  constexpr uint16_t kPortID = 201;

  uint8_t state = 0; // kStateInvalid
  if (m_processingState[0].m_calibrationState == kCalibrated) {
    if (m_processingState[0].m_eyeFitterOutputsValid) {
      state = 2; // kStatePupilLock
    } else {
      state = 1; // kStateCalibrated
    }
  }

  SerializationBuffer buf;
  buf.reserve(8);

  glm::vec2 angles = getPitchYawAnglesForEye(0);
  buf.put_u8(state);
  buf.put_i16_le(serializeAngle(angles[0])); // pitch
  buf.put_i16_le(serializeAngle(angles[1])); // yaw
  buf.put_u8((triggerBlink && m_enableBlink) ? 1 : 0); // blink trigger flag

  canbus()->transmitMessage(kPortID, buf);

  return true;
}

cv::Mat& EyeTrackingService::getDebugViewForEye(size_t eyeIdx) {
  assert(eyeIdx < 2);
  return m_processingState[eyeIdx].m_debugViewRGB;
}

void EyeTrackingService::applyCalibrationData() {
  PER_EYE {
    ProcessingState& ps = m_processingState[eyeIdx];

    // Experimentally, roughly 1800.0 is a good ps.m_eyeModelFitter.focal_length value for the OV9281 sensor in 640x480 mode with 60 degree lens.
    ps.m_eyeModelFitter.focal_length = sefFocalLength();

    ps.m_pitchFilter.mincutoff = m_filterMinCutoff;
    ps.m_pitchFilter.beta = powf(10.0f, m_filterBetaExponent);
    ps.m_pitchFilter.dcutoff = m_filterDCutoff;

    ps.m_yawFilter.mincutoff = m_filterMinCutoff;
    ps.m_yawFilter.beta = powf(10.0f, m_filterBetaExponent);
    ps.m_yawFilter.dcutoff = m_filterDCutoff;
  }
}

void EyeTrackingService::renderIMGUI() {
  ImGui::PushID(this);

  ImGui::Checkbox("Auto-save bad fit images", &m_debugSaveBadFitImages);
  ImGui::Checkbox("ET capture freeze", &m_debugFreezeCapture);
  ImGui::Checkbox("Disable Processing", &m_debugDisableProcessing);
  ImGui::Checkbox("ET camera feedback view", &m_debugShowFeedbackView);
  if (m_debugShowFeedbackView) {
    ImGui::DragFloat("FB brightness", &m_debugFeedbackBrightness, /*speed=*/ 0.05f, /*min=*/ 0.0f, /*max=*/ 1.0f, "%.2f");
    ImGui::Checkbox("Draw debug overlays", &m_debugDrawOverlays);
  }

  if (ImGui::Button("Recalibrate")) {
    debugClearCalibration();
  }

  if (ImGui::CollapsingHeader("Data graphs")) {
    int plotFlags = ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;

    PER_EYE {
      ProcessingState& ps = m_processingState[eyeIdx];
      if (!ps.processingThreadAlive())
        continue;

      ImGui::PushID(eyeIdx);
      ImGui::Text("Eye %zu", eyeIdx);

      ImGui::Text("Last invalid run: %u frames", ps.m_lastInvalidFrameRunLength);

      ImGui::Checkbox("Freeze graph", &ps.m_freezeGraphData);

      if (ImPlot::BeginPlot("##EllipseData1", ImVec2(-1,150), /*flags=*/ plotFlags)) {
        ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit); // | ImPlotAxisFlags_LockMin);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, ps.m_graphData.size(), ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -10.0f, 10.0f, ImPlotCond_Always);
        ImPlot::SetupFinish();

        ImPlot::PlotLine("Delta Center", &ps.m_graphData.data()[0].deltaCenter, ps.m_graphData.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, ps.m_graphData.offset(), sizeof(ProcessingState::GraphData));
        ImPlot::EndPlot();
      }

      if (ImPlot::BeginPlot("##EllipseData2", ImVec2(-1,150), /*flags=*/ plotFlags)) {
        ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit); // | ImPlotAxisFlags_LockMin);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, ps.m_graphData.size(), ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 90.0f, 110.0f, ImPlotCond_Once);
        ImPlot::SetupFinish();

        ImPlot::PlotLine("Delta Size", &ps.m_graphData.data()[0].deltaSize, ps.m_graphData.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, ps.m_graphData.offset(), sizeof(ProcessingState::GraphData));
        ImPlot::EndPlot();
      }

      if (ImPlot::BeginPlot("##EllipseData3", ImVec2(-1,150), /*flags=*/ plotFlags)) {
        ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit); // | ImPlotAxisFlags_LockMin);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, ps.m_graphData.size(), ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -45.0f, 45.0f, ImPlotCond_Always);
        ImPlot::SetupFinish();

        ImPlot::PlotLine("Delta Angle", &ps.m_graphData.data()[0].deltaAngle, ps.m_graphData.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, ps.m_graphData.offset(), sizeof(ProcessingState::GraphData));
        ImPlot::EndPlot();
      }

      ImGui::PopID();
    }
  }


  bool dirty = false;

  // Roll angles don't require recalibrating
  ImGui::DragFloat("L Roll angle (deg)", &m_rollOffsetDeg[0], /*speed=*/ 0.1f, /*min=*/ -30.0f, /*max=*/ 30.0f, "%.1f");
  ImGui::DragFloat("R Roll angle (deg)", &m_rollOffsetDeg[1], /*speed=*/ 0.1f, /*min=*/ -30.0f, /*max=*/ 30.0f, "%.1f");

  ImGui::Separator();
  ImGui::DragFloat("Sector cutoff angle (deg)", &m_sectorCutoffAngleDeg, /*speed=*/ 1.0f, /*min=*/ 0.0f, /*max=*/ 90.0f, "%.1f");

  // Filter settings also apply immediately, but still require calling applyCalibrationData
  ImGui::Separator();
  dirty |= ImGui::DragFloat("Filter min cutoff", &m_filterMinCutoff, /*speed=*/ 0.01f, /*min=*/ 0.0f, /*max=*/ 2.0f, "%.2f");
  dirty |= ImGui::DragFloat("Filter D cutoff", &m_filterDCutoff, /*speed=*/ 0.01f, /*min=*/ 0.0f, /*max=*/ 2.0f, "%.2f");
  dirty |= ImGui::DragFloat("Filter beta exponent", &m_filterBetaExponent, /*speed=*/ 0.1f, /*min=*/ -10.0f, /*max=*/ 10.0f, "%.1f");

  // All of these settings require recalibrating the model
  ImGui::Separator();
  dirty |= ImGui::DragFloat("Focal Length", &m_focalLength, /*speed=*/ 0.1, /*min=*/ 1.0f, /*max=*/ 20.0f, "%.1f");
  dirty |= ImGui::DragFloat("Distance to eye (mm)", &m_eyeZ, /*speed=*/ 0.5f, /*min=*/ 1.0f, /*max=*/ 50.0f, "%.1f");
  dirty |= ImGui::DragFloat("Sensor pixel pitch (um)", &m_pixelPitchMicrons, /*speed=*/ 0.1f, /*min=*/ 0.1f, /*max=*/ 10.0f, "%.1f");

  ImGui::Separator();
  // Misc settings
  ImGui::DragInt("Crosshair hide after valid frames", &m_hideCrosshairAfterFrameCount, /*v_speed=*/ 10, /*v_min=*/ 0, /*v_max=*/ 1000);
  ImGui::DragInt("Crosshair show after invalid frames", &m_showCrosshairAfterFrameCount, /*v_speed=*/ 1, /*v_min=*/ 0, /*v_max=*/ 100);

  ImGui::DragInt("Blink Detect Min Frames", &m_blinkDetectMinFrames, /*v_speed=*/ 1, /*v_min=*/ 0, /*v_max=*/ 100);
  ImGui::DragInt("Blink Detect Max Frames", &m_blinkDetectMaxFrames, /*v_speed=*/ 1, /*v_min=*/ 0, /*v_max=*/ 100);
  ImGui::DragInt("Blink Cooldown Frames", &m_blinkCooldownFrames, /*v_speed=*/ 10, /*v_min=*/ 0, /*v_max=*/ 1000);

  if (ImGui::Button("Save Settings")) {
    saveCalibrationData();
  }

  if (dirty) {
    applyCalibrationData();
  }


  ImGui::PopID();
}

glm::vec2 EyeTrackingService::getPitchYawAnglesForEye(size_t eyeIdx) {
  assert(eyeIdx == 0 || eyeIdx == 1);
  ProcessingState& ps = m_processingState[eyeIdx];

  if (ps.m_calibrationState != kCalibrated) {
    // Not calibrated, so we don't have valid angles to return.
    return glm::vec2(0.0f);
  }

  glm::vec2 angles = glm::vec2(
    ps.m_pupilFilteredPitchDeg - ps.m_centerPitchDeg,
    ps.m_pupilFilteredYawDeg - ps.m_centerYawDeg);

  // Apply roll correction
  glm::vec3 rollCorrectionVector;
  anglesToVector<float>(glm::radians(m_rollOffsetDeg[eyeIdx]), glm::radians(angles[0]), glm::radians(angles[1]), glm::value_ptr(rollCorrectionVector));
  vectorToAngles<float>(glm::value_ptr(rollCorrectionVector), angles[0], angles[1], /*toDegrees=*/ true);
  return angles;
}


void EyeTrackingService::requestCapture() {
  // Make sure the target directory exists
  if (mkdir(kCaptureDirName, 0777) < 0) {
    // EEXIST is ok, anything else is fatal
    if (errno != EEXIST) {
      printf("EyeTrackingService::requestCapture(): can't create capture directory \"%s\": %s\n", kCaptureDirName, strerror(errno));
      return;
    }
  }

  // Select a consistent capture index across all sessions.
  uint64_t captureIndex = currentRealTimeMs();

  // Trigger capture in any active sessions
  PER_EYE {
    ProcessingState& ps = m_processingState[eyeIdx];
    if (ps.processingThreadAlive()) {
      ps.m_captureFileIndex = captureIndex;
    }
  }

}

void EyeTrackingService::renderSceneGizmos_preUI(FxRenderView* renderViews) {
  if (m_debugDisableProcessing)
    return;

  // TODO needs to support dual eye!
  // TODO parameterize? or pull from external config
  const float feedbackViewDepth = 0.5f;

  // Render eyetracking debug view
  if (m_debugShowFeedbackView) {
    const float feedbackViewScale = 0.175f;

    ProcessingState& ps = m_processingState[0];
    const cv::Mat& debugView = getDebugViewForEye(0);

    if (debugView.cols && debugView.rows) {

      if (!ps.m_eyeTrackingDebugTexture || (ps.m_eyeTrackingDebugTexture->width() != debugView.cols) || (ps.m_eyeTrackingDebugTexture->height() != debugView.rows)) {
        ps.m_eyeTrackingDebugTexture = rhi()->newTexture2D(debugView.cols, debugView.rows, kSurfaceFormat_RGBA8);
      }

      // Eyetracking debug view is drawn in RGBA, so we just have to upload it.
      rhi()->loadTextureData(ps.m_eyeTrackingDebugTexture, kVertexElementTypeUByte4N, debugView.ptr());


      rhi()->bindBlendState(disabledBlendState);
      rhi()->bindDepthStencilState(disabledDepthStencilState);
      rhi()->bindRenderPipeline(uiLayerStereoPipeline);
      rhi()->loadTexture(ksImageTex, ps.m_eyeTrackingDebugTexture, linearClampSampler);
      // rhi()->setViewports(eyeViewports, 2); // should already be set

      UILayerStereoUniformBlock ub;
      glm::mat4 modelMatrix = glm::translate(glm::vec3(0.0f, 0.0f, -feedbackViewDepth)) * glm::scale(glm::vec3(feedbackViewScale * (static_cast<float>(ps.m_eyeTrackingDebugTexture->width()) / static_cast<float>(ps.m_eyeTrackingDebugTexture->height())), -feedbackViewScale, feedbackViewScale));
      ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
      ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;
      ub.tint = glm::vec4(m_debugFeedbackBrightness, m_debugFeedbackBrightness, m_debugFeedbackBrightness, 1.0f);

      rhi()->loadUniformBlockImmediate(ksUILayerStereoUniformBlock, &ub, sizeof(ub));
      rhi()->drawNDCQuad();
    }
  }
}

void EyeTrackingService::renderSceneGizmos_postUI(FxRenderView* renderViews) {
  if (m_debugDisableProcessing)
    return;

  // TODO needs to support dual eye!

  const float crosshairDepth = 0.4f;

  // Eye crosshair
  if (m_processingState[0].m_calibrationState == kCalibrated && m_processingState[0].m_shouldShowCrosshair) {
    CrosshairUniformBlock ub;

    glm::vec2 measuredPoint = getPitchYawAnglesForEye(0);

    glm::mat4 modelMatrix =
        glm::eulerAngleXY(
          glm::radians(measuredPoint.x),
          glm::radians(measuredPoint.y))
      * glm::translate(glm::vec3(0.0f, 0.0f, -crosshairDepth))
      * glm::scale(glm::vec3(0.005f));

    ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
    ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;
    if (m_processingState[0].m_contiguousValidFrameCounter) {
      // lock is currently valid
      ub.color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f); // green
    } else {
      // lock is currently invalid
      ub.color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // red
    }
    ub.thickness = 0.5f;

    rhi()->bindRenderPipeline(crosshairPipeline);
    rhi()->bindDepthStencilState(disabledDepthStencilState);

    rhi()->loadUniformBlockImmediate(ksCrosshairUniformBlock, &ub, sizeof(ub));
    // rhi()->setViewports(eyeViewports, 2); // should already be set

    rhi()->bindStreamBuffer(0, ndcQuadVBO);
    rhi()->drawPrimitives(0, 4, /*instanceCount=*/ 2);
  }

  // Center crosshair
  if (m_processingState[0].m_calibrationState <= kCentering) {
    CrosshairUniformBlock ub;

    glm::mat4 modelMatrix =
        glm::translate(glm::vec3(0.0f, 0.0f, -crosshairDepth))
      * glm::scale(glm::vec3(0.00375f));

    ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
    ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;
    ub.color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    ub.thickness = 0.75f;

    rhi()->bindRenderPipeline(crosshairPipeline);
    rhi()->bindDepthStencilState(disabledDepthStencilState);

    rhi()->loadUniformBlockImmediate(ksCrosshairUniformBlock, &ub, sizeof(ub));
    // rhi()->setViewports(eyeViewports, 2); // should already be set

    rhi()->bindStreamBuffer(0, ndcQuadVBO);
    rhi()->drawPrimitives(0, 4, /*instanceCount=*/ 2);
  }
}

const char* EyeTrackingService::getDebugPerfStatsForEye(size_t eyeIdx) {
  assert(eyeIdx <= 1);

  char* buf = m_processingState[eyeIdx].m_debugPerfStatsBuffer;
  constexpr size_t len = sizeof(m_processingState[eyeIdx].m_debugPerfStatsBuffer);
  if (m_processingState[eyeIdx].processingThreadAlive()) {
    snprintf(buf, len - 1,
      "Processing time: %.3fms (%.3fms pre, %.3fms ROI, %.3fms ROI-seg latency, %.3fms segmentation, %.3fms post, %.3fms debug view)",
      m_processingState[eyeIdx].m_lastFrameTotalProcessingTimeMs,
      m_processingState[eyeIdx].m_lastFramePreProcessingTimeMs,
      m_processingState[eyeIdx].m_lastFrameROITimeMs,
      m_processingState[eyeIdx].m_lastFrameROIToSegmentationTimeMs,
      m_processingState[eyeIdx].m_lastFrameSegmentationTimeMs,
      m_processingState[eyeIdx].m_lastFramePostProcessingTimeMs,
      m_processingState[eyeIdx].m_lastFrameDebugViewTimeMs);
  } else {
    snprintf(buf, len - 1,
      "Processing thread not running");
  }

  buf[len - 1] = '\0';
  return buf;
}

