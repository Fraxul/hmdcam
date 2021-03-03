#pragma once
#include <vector>
#include <string>
#include "rhi/RHI.h"
#include "common/ICameraProvider.h"
#include <opencv2/core.hpp>

class CameraSystem {
public:
  CameraSystem(ICameraProvider*);

  bool loadCalibrationData();
  void saveCalibrationData();


  struct Camera {
    RHISurface::ptr intrinsicDistortionMap;
    RHISurface::ptr mask;
    cv::Mat intrinsicMatrix; // From calibration
    cv::Mat distCoeffs;
    cv::Mat optimizedMatrix; // Computed by cv::getOptimalNewCameraMatrix from cameraIntrinsicMatrix and distCoeffs
    double fovX, fovY; // Values for the optimized camera matrix, in degrees

    bool haveIntrinsicCalibration() const { return (!(intrinsicMatrix.empty() || distCoeffs.empty())); }
    bool haveIntrinsicDistortionMap() const { return (!(optimizedMatrix.empty() || intrinsicDistortionMap.get() == nullptr)); }

  };

  struct View {
    bool isStereo;
    size_t cameraCount() const { return isStereo ? 2 : 1; }
    unsigned short cameraIndices[2]; // if (isStereo) [0] is left, [1] is right. Otherwise, only use [0].

    // TODO view translation and orientation

    // Stereo data, only valid if (isStereo)
    cv::Mat stereoRotation, stereoTranslation; // Calibrated
    cv::Mat stereoRectification[2], stereoProjection[2]; // Derived from stereoRotation/stereoTranslation via cv::stereoRectify
    cv::Mat stereoDisparityToDepth;
    cv::Rect stereoValidROI[2];
    RHISurface::ptr stereoDistortionMap[2]; // Combined intrinsic and stereo distortion
    double fovX, fovY; // Values for the stereo projection, in degrees

    bool haveStereoCalibration() const { return (!(stereoRotation.empty() || stereoTranslation.empty())); }
    bool haveStereoRectificationParameters() const { return (!(
      stereoRectification[0].empty() || stereoRectification[1].empty() ||
      stereoProjection[0].empty() || stereoProjection[1].empty() ||
      stereoDisparityToDepth.empty() ||
      stereoValidROI[0].empty() || stereoValidROI[1].empty()));
    }

  };

  size_t createMonoView(size_t cameraIndex);
  size_t createStereoView(size_t leftCameraIndex, size_t rightCameraIndex);

  size_t cameras() const { return m_cameras.size(); }
  Camera& cameraAtIndex(size_t cameraIndex) { return m_cameras[cameraIndex]; }

  size_t views() const { return m_views.size(); }
  View& viewAtIndex(size_t viewIndex) { return m_views[viewIndex]; }

  class CalibrationContext;


  CalibrationContext* calibrationContextForCamera(size_t cameraIdx);
  CalibrationContext* calibrationContextForView(size_t viewIdx);

  std::string calibrationFilename;

  class CalibrationContext {
  public:
    virtual ~CalibrationContext();
    virtual void processFrame() = 0;
    virtual void processUI() = 0; // run IMGUI code to draw UI and handle inputs
    virtual bool finished() = 0;

    virtual bool requiresStereoRendering() const = 0;
    virtual RHISurface::ptr overlaySurfaceAtIndex(size_t) = 0;
    virtual bool isCameraContext() { return false; }
    virtual bool isViewContext() { return false; }
    virtual size_t getCameraOrViewIndex() = 0;

    virtual bool involvesCamera(size_t cameraIdx) = 0;
    virtual size_t overlaySurfaceIndexForCamera(size_t cameraIdx) = 0;

  protected:
    CalibrationContext(CameraSystem*);
    CameraSystem* cameraSystem() const { return m_cameraSystem; }
    ICameraProvider* cameraProvider() const { return cameraSystem()->cameraProvider(); }

  private:
    CameraSystem* m_cameraSystem;
  };

  class CalibrationContextStateMachineBase : public CalibrationContext {
  public:
    virtual ~CalibrationContextStateMachineBase();

    virtual void processFrame();
    virtual void processUI();
    virtual bool finished();

  protected:
    CalibrationContextStateMachineBase(CameraSystem*);

    bool inCalibrationPreviewMode() const { return m_inCalibrationPreviewMode; }
    bool captureRequested() const { return m_captureRequested; }
    void acknowledgeCaptureRequest() { m_captureRequested = false; }
    bool shouldSaveCalibrationImages() const { return m_saveCalibrationImages; }

    virtual void processFrameCaptureMode() = 0;
    virtual void renderStatusUI() = 0;
    virtual bool cookCalibrationDataForPreview() = 0;
    virtual void didAcceptCalibrationPreview() = 0;
    virtual void didRejectCalibrationPreview() = 0;
    virtual void didCancelCalibrationSession() = 0;

  private:
    bool m_captureRequested;
    bool m_finishRequested;
    bool m_cancelRequested;
    bool m_saveCalibrationImages;

    bool m_inCalibrationPreviewMode;
    bool m_calibrationPreviewAccepted;
    bool m_calibrationPreviewRejected;

    bool m_calibrationFinished;

  };


  friend class CalibrationContext;

  class IntrinsicCalibrationContext : public CalibrationContextStateMachineBase {
  public:
    IntrinsicCalibrationContext(CameraSystem*, size_t cameraIdx);
    virtual ~IntrinsicCalibrationContext();

    virtual bool requiresStereoRendering() const;
    virtual RHISurface::ptr overlaySurfaceAtIndex(size_t);

    virtual bool isCameraContext() { return true; }
    virtual size_t getCameraOrViewIndex() { return m_cameraIdx; }

    virtual bool involvesCamera(size_t cameraIdx) { return cameraIdx == m_cameraIdx; }
    virtual size_t overlaySurfaceIndexForCamera(size_t cameraIdx) { return cameraIdx == m_cameraIdx ? 0 : -1; }

  protected:
    virtual void renderStatusUI();
    virtual void processFrameCaptureMode();
    virtual bool cookCalibrationDataForPreview();
    virtual void didAcceptCalibrationPreview();
    virtual void didRejectCalibrationPreview();
    virtual void didCancelCalibrationSession();

    size_t m_cameraIdx;
    RHISurface::ptr m_fullGreyTex;
    RHIRenderTarget::ptr m_fullGreyRT;
    RHISurface::ptr m_feedbackTex;
    cv::Mat m_feedbackView;

    std::vector<cv::Mat> m_allCharucoCorners;
    std::vector<cv::Mat> m_allCharucoIds;

    // Cached data of previous calibration to be restored if the context is cancelled.
    cv::Mat m_previousIntrinsicMatrix;
    cv::Mat m_previousDistCoeffs;

  };

  class StereoCalibrationContext : public CalibrationContextStateMachineBase {
  public:
    StereoCalibrationContext(CameraSystem*, size_t viewIdx);
    virtual ~StereoCalibrationContext();

    virtual bool requiresStereoRendering() const;
    virtual RHISurface::ptr overlaySurfaceAtIndex(size_t);

    virtual bool isViewContext() { return true; }
    virtual size_t getCameraOrViewIndex() { return m_viewIdx; }

    virtual bool involvesCamera(size_t cameraIdx) {
      View& v = cameraSystem()->viewAtIndex(m_viewIdx);
      return (cameraIdx == v.cameraIndices[0] || cameraIdx == v.cameraIndices[1]);
    }

    virtual size_t overlaySurfaceIndexForCamera(size_t cameraIdx) {
      View& v = cameraSystem()->viewAtIndex(m_viewIdx);
      if (cameraIdx == v.cameraIndices[0])
        return 0;
      else if (cameraIdx == v.cameraIndices[1])
        return 1;
      else
        return -1;
    }

  protected:
    virtual void renderStatusUI();
    virtual void processFrameCaptureMode();
    virtual bool cookCalibrationDataForPreview();
    virtual void didAcceptCalibrationPreview();
    virtual void didRejectCalibrationPreview();
    virtual void didCancelCalibrationSession();

    size_t m_viewIdx;

    RHISurface::ptr m_fullGreyTex[2];
    RHIRenderTarget::ptr m_fullGreyRT[2];
    RHISurface::ptr m_feedbackTex[2];
    cv::Mat m_feedbackView[2];

    std::vector<std::vector<cv::Point3f> > m_objectPoints; // Points from the board definition for the relevant corners each frame
    std::vector<std::vector<cv::Point2f> > m_calibrationPoints[2]; // Points in image space for the 2 views for the relevant corners each frame

    // Cached data of previous calibration to be restored if the context is cancelled.
    View m_previousViewData;

  };

  ICameraProvider* cameraProvider() const { return m_cameraProvider; }

protected:
  ICameraProvider* m_cameraProvider;

  std::vector<Camera> m_cameras;
  std::vector<View> m_views;


  void updateCameraIntrinsicDistortionParameters(size_t cameraIdx); // Generate camera-specific derived data after calibration
  void updateViewStereoDistortionParameters(size_t viewIdx); // Generate view-specific derived data after calibration
  RHISurface::ptr generateGPUDistortionMap(cv::Mat map1, cv::Mat map2);
  cv::Mat captureGreyscale(size_t cameraIdx, RHISurface::ptr tex, RHIRenderTarget::ptr rt, bool undistort);

private:
  // noncopyable
  CameraSystem(const CameraSystem&);
  CameraSystem& operator=(const CameraSystem&);
};

