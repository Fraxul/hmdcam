#pragma once
#include <vector>
#include <string>
#include "rhi/RHI.h"
#include "common/ICameraProvider.h"
#include <opencv2/core.hpp>
#include "glm/gtx/euler_angles.hpp"
#include "glm/gtx/transform.hpp"

class CameraSystem;
class CharucoMultiViewCalibration;
class DepthMapGenerator;

std::vector<glm::vec3> getTriangulatedPointsForView(CameraSystem* cameraSystem, size_t viewIdx, const std::vector<std::vector<cv::Point2f> >& leftCalibrationPoints, const std::vector<std::vector<cv::Point2f> >& rightCalibrationPoints);
std::vector<glm::vec3> transformBoardPointsForView(const glm::mat4& transform);

// computes a linear transform that maps the points in pVec2 to their positions in pVec1 -- translation and rotation only
// return value is RMS error distance between (outTransform * pVec2) points and pVec1 points
float computePointSetLinearTransform(const std::vector<glm::vec3>& pVec1, const std::vector<glm::vec3>& pVec2, glm::mat4& outTransform);

class CameraSystem {
public:
  CameraSystem(ICameraProvider*);

  bool loadCalibrationData();
  void saveCalibrationData();


  struct Camera {
    Camera() : fovX(0), fovY(0) {}

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
    View() : depthMapGenerator(NULL), isStereo(false), viewTranslation(0.0f), viewRotation(0.0f), fovX(0), fovY(0) {}

    DepthMapGenerator* depthMapGenerator;

    bool isStereo;
    size_t cameraCount() const { return isStereo ? 2 : 1; }
    unsigned short cameraIndices[2]; // if (isStereo) [0] is left, [1] is right. Otherwise, only use [0].

    glm::vec3 viewTranslation;
    glm::vec3 viewRotation; // euler, degrees

    // local transform only. for absolute view-to-world, see CameraSystem::viewWorldTransform
    glm::mat4 viewLocalTransform() const {
      glm::mat4 res = glm::eulerAngleXYZ(glm::radians(viewRotation[0]), glm::radians(viewRotation[1]), glm::radians(viewRotation[2]));
      for (size_t i = 0; i < 3; ++i) res[3][i] = viewTranslation[i];
      return res;
    }

    // Stereo data, only valid if (isStereo)
    cv::Mat stereoRotation, stereoTranslation;  // Calibrated -- from cv::stereoCalibrate
    cv::Mat essentialMatrix, fundamentalMatrix; // Calibrated -- from cv::stereoCalibrate
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
    bool isVerticalStereo() const { return fabs(stereoTranslation.at<double>(0, 1)) > fabs(stereoTranslation.at<double>(0, 0)); }

  };

  size_t createMonoView(size_t cameraIndex);
  size_t createStereoView(size_t leftCameraIndex, size_t rightCameraIndex);

  size_t cameras() const { return m_cameras.size(); }
  Camera& cameraAtIndex(size_t cameraIndex) { return m_cameras[cameraIndex]; }
  const Camera& cameraAtIndex(size_t cameraIndex) const { return m_cameras[cameraIndex]; }

  size_t views() const { return m_views.size(); }
  View& viewAtIndex(size_t viewIndex) { return m_views[viewIndex]; }
  const View& viewAtIndex(size_t viewIndex) const { return m_views[viewIndex]; }

  glm::mat4 viewWorldTransform(size_t viewIdx) const;

  class CalibrationContext;


  CalibrationContext* calibrationContextForCamera(size_t cameraIdx);
  CalibrationContext* calibrationContextForView(size_t viewIdx);
  CalibrationContext* calibrationContextForStereoViewOffset(size_t referenceViewIdx, size_t viewIdx);

  std::string calibrationFilename;

  class CalibrationContext {
  public:

    enum OverlayDistortionSpace {
      kDistortionSpaceUncorrected,
      kDistortionSpaceIntrinsic,
      kDistortionSpaceView
    };

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
    virtual OverlayDistortionSpace overlayDistortionSpace() const = 0;
    virtual RHISurface::ptr previewDistortionMapForCamera(size_t cameraIdx) const = 0;

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
    virtual void processFramePreviewMode() = 0;
    virtual void renderStatusUI() = 0;
    virtual bool cookCalibrationDataForPreview() = 0;
    virtual void didAcceptCalibrationPreview() = 0;
    virtual void didRejectCalibrationPreview() = 0;
    virtual void didCancelCalibrationSession() = 0;

  private:
    bool m_captureRequested;
    bool m_previewRequested;
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
    virtual OverlayDistortionSpace overlayDistortionSpace() const;
    virtual RHISurface::ptr previewDistortionMapForCamera(size_t cameraIdx) const;

  protected:
    virtual void renderStatusUI();
    virtual void processFrameCaptureMode();
    virtual void processFramePreviewMode();
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

    // Preview data
    void asyncUpdateIncrementalCalibration();

    cv::Mat m_perViewErrors;
    double m_feedbackRmsError;
    double m_feedbackFovX, m_feedbackFovY; // degrees
    cv::Point2d m_feedbackPrincipalPoint;
    bool m_incrementalUpdateInProgress;
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
    virtual OverlayDistortionSpace overlayDistortionSpace() const { return kDistortionSpaceUncorrected; }
    virtual RHISurface::ptr previewDistortionMapForCamera(size_t cameraIdx) const {
      // View& v = cameraSystem()->viewAtIndex(m_viewIdx);
      // return cameraSystem()->cameraAtIndex((cameraIdx == v.cameraIndices[0]) ? 0 : 1).intrinsicDistortionMap;
      return RHISurface::ptr();
    }

  protected:
    virtual void renderStatusUI();
    virtual void processFrameCaptureMode();
    virtual void processFramePreviewMode();
    virtual bool cookCalibrationDataForPreview();
    virtual void didAcceptCalibrationPreview();
    virtual void didRejectCalibrationPreview();
    virtual void didCancelCalibrationSession();

    void internalUpdateCaptureState();

    size_t m_viewIdx;

    CharucoMultiViewCalibration* m_calibState;

    // Cached data of previous calibration to be restored if the context is cancelled.
    View m_previousViewData;

    // Feedback data during capture
    glm::vec3 m_feedbackTx, m_feedbackRx;
    double m_feedbackRmsError;
    cv::Mat m_perViewErrors;
    cv::Rect m_feedbackValidROI[2];

  };


  class StereoViewOffsetCalibrationContext : public CalibrationContextStateMachineBase {
  public:
    StereoViewOffsetCalibrationContext(CameraSystem*, size_t referenceViewIdx, size_t viewIdx);
    virtual ~StereoViewOffsetCalibrationContext();

    virtual bool requiresStereoRendering() const;
    virtual RHISurface::ptr overlaySurfaceAtIndex(size_t);

    virtual bool isViewContext() { return true; }
    virtual size_t getCameraOrViewIndex() { return m_viewIdx; }

    virtual bool involvesCamera(size_t cameraIdx) {
      const View& rv = cameraSystem()->viewAtIndex(m_referenceViewIdx);
      const View& v = cameraSystem()->viewAtIndex(m_viewIdx);
      return (cameraIdx == v.cameraIndices[0] || cameraIdx == v.cameraIndices[1] ||
              cameraIdx == rv.cameraIndices[0] || cameraIdx == rv.cameraIndices[1]);
    }

    virtual size_t overlaySurfaceIndexForCamera(size_t cameraIdx) {
      View& rv = cameraSystem()->viewAtIndex(m_referenceViewIdx);
      if (cameraIdx == rv.cameraIndices[0])
        return 0;
      else if (cameraIdx == rv.cameraIndices[1])
        return 1;

      View& v = cameraSystem()->viewAtIndex(m_viewIdx);
      if (cameraIdx == v.cameraIndices[0])
        return 2;
      else if (cameraIdx == v.cameraIndices[1])
        return 3;

      return -1;
    }
    virtual OverlayDistortionSpace overlayDistortionSpace() const { return kDistortionSpaceView; }
    virtual RHISurface::ptr previewDistortionMapForCamera(size_t cameraIdx) const {
      View& rv = cameraSystem()->viewAtIndex(m_referenceViewIdx);
      if (cameraIdx == rv.cameraIndices[0])
        return rv.stereoDistortionMap[0];
      else if (cameraIdx == rv.cameraIndices[1])
        return rv.stereoDistortionMap[1];

      View& v = cameraSystem()->viewAtIndex(m_viewIdx);
      if (cameraIdx == v.cameraIndices[0])
        return v.stereoDistortionMap[0];
      else if (cameraIdx == v.cameraIndices[1])
        return v.stereoDistortionMap[1];

      return RHISurface::ptr();
    }

  protected:
    virtual void renderStatusUI();
    virtual void processFrameCaptureMode();
    virtual void processFramePreviewMode();
    virtual bool cookCalibrationDataForPreview();
    virtual void didAcceptCalibrationPreview();
    virtual void didRejectCalibrationPreview();
    virtual void didCancelCalibrationSession();

    size_t m_referenceViewIdx, m_viewIdx;

    CharucoMultiViewCalibration* m_refCalibState;
    CharucoMultiViewCalibration* m_tgtCalibState;

    std::vector<glm::vec3> m_refPoints, m_tgtPoints;

    glm::mat4 m_tgt2ref;
    float m_rmsError;

    // Cached data of previous calibration to be restored if the context is cancelled.
    glm::vec3 m_previousViewTranslation, m_previousViewRotation;

    bool m_useLinearRemap;
  };

  ICameraProvider* cameraProvider() const { return m_cameraProvider; }

  cv::Mat captureGreyscale(size_t cameraIdx, RHISurface::ptr tex, RHIRenderTarget::ptr rt, RHISurface::ptr distortionMap = RHISurface::ptr());
protected:
  ICameraProvider* m_cameraProvider;

  std::vector<Camera> m_cameras;
  std::vector<View> m_views;


  void updateCameraIntrinsicDistortionParameters(size_t cameraIdx); // Generate camera-specific derived data after calibration
  void updateViewStereoDistortionParameters(size_t viewIdx); // Generate view-specific derived data after calibration
  RHISurface::ptr generateGPUDistortionMap(cv::Mat map1, cv::Mat map2);

private:
  // noncopyable
  CameraSystem(const CameraSystem&);
  CameraSystem& operator=(const CameraSystem&);
};

