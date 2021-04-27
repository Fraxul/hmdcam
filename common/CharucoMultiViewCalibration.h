#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include "common/ICameraProvider.h"
#include "rhi/RHISurface.h"
#include "rhi/RHIRenderTarget.h"

class CameraSystem;


// Handles tracking a Charuco target across multiple views for calibration purposes

class CharucoMultiViewCalibration {
public:
  // Camera stereo view IDs are optional -- if provided, images will be undistorted using the stereo distortion maps for the associated view.
  CharucoMultiViewCalibration(CameraSystem*, const std::vector<size_t>& cameraIds, const std::vector<size_t>& cameraStereoViewIds = std::vector<size_t>());

  // Returns true if a frame was actually captured (only if captureRequested is true)
  bool processFrame(bool captureRequested);
  void reset();


  bool m_undistortCapturedViews;
  bool m_enableFeedbackView;

  CameraSystem* m_cameraSystem;
  std::vector<size_t> m_cameraIds;
  std::vector<ssize_t> m_cameraStereoViewIds;

  size_t cameraCount() const { return m_cameraIds.size(); }

  CameraSystem* cameraSystem() const { return m_cameraSystem; }
  ICameraProvider* cameraProvider() const;

  // Per-camera
  std::vector<RHISurface::ptr> m_fullGreyTex;
  std::vector<RHIRenderTarget::ptr> m_fullGreyRT;
  std::vector<RHISurface::ptr> m_feedbackTex;
  std::vector<cv::Mat> m_feedbackView;

  // Per-camera per-frame
  std::vector<std::vector<std::vector<cv::Point2f> > > m_calibrationPoints; // Points in image space for the 2 views for the relevant corners each frame

  // Per-frame
  std::vector<std::vector<cv::Point3f> > m_objectPoints; // Points from the board definition for the relevant corners each frame
};
