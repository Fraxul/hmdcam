#pragma once
#include "common/FxRenderView.h"
#include "common/DepthMapSHM.h"
#include "common/SHMSegment.h"
#include "rhi/RHISurface.h"
#include "rhi/RHIBuffer.h"
#include <glm/glm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

class CameraSystem;
class DepthMapGenerator;

enum DepthMapGeneratorBackend {
  kDepthBackendNone,
  kDepthBackendDGPU,
  kDepthBackendDepthAI
};

DepthMapGeneratorBackend depthBackendStringToEnum(const char* backendStr);
DepthMapGenerator* createDepthMapGenerator(DepthMapGeneratorBackend);

class DepthMapGenerator {
public:
  DepthMapGenerator(DepthMapGeneratorBackend);
  ~DepthMapGenerator();

  void initWithCameraSystem(CameraSystem*);
  void processFrame();
  void renderDisparityDepthMapStereo(size_t viewIdx, const FxRenderView& leftRenderView, const FxRenderView& rightRenderView, const glm::mat4& modelMatrix = glm::mat4(1.0f));
  void renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView, const glm::mat4& modelMatrix = glm::mat4(1.0f));
  void renderIMGUI();

  RHISurface::ptr disparitySurface(size_t viewIdx) const { return m_viewData[viewIdx]->m_disparityTexture; }
  RHISurface::ptr leftGrayscale(size_t viewIdx) const { return m_viewData[viewIdx]->m_leftGray; }
  RHISurface::ptr rightGrayscale(size_t viewIdx) const { return m_viewData[viewIdx]->m_rightGray; }

  // controls availability of leftGrayscale and rightGrayscale
  void setPopulateDebugTextures(bool value) { m_populateDebugTextures = value; }

  float m_disparityPrescale;

  bool loadSettings();
  void saveSettings();

  DepthMapGeneratorBackend backend() const { return m_backend; }

protected:

  DepthMapGeneratorBackend m_backend;

  virtual void internalLoadSettings(cv::FileStorage&) = 0;
  virtual void internalSaveSettings(cv::FileStorage&) = 0;
  virtual void internalRenderIMGUI() = 0;
  virtual void internalProcessFrame() = 0;

  uint32_t m_algoDownsampleX = 1; // optionally overridden in backend
  uint32_t m_algoDownsampleY = 1;

  uint32_t internalWidth() const { return m_internalWidth; }
  uint32_t internalHeight() const { return m_internalHeight; }

  void internalGenerateDisparityMips();

  CameraSystem* m_cameraSystem = NULL;

  struct ViewData {
    ViewData() {}

    bool m_isStereoView = false;
    bool m_isVerticalStereo = false;
    size_t m_leftCameraIndex = 0, m_rightCameraIndex = 0;

    glm::mat4 m_R1inv, m_Q, m_Qinv;

    RHISurface::ptr m_disparityTexture;
    RHISurface::ptr m_leftGray, m_rightGray;

    std::vector<RHIRenderTarget::ptr> m_disparityTextureMipTargets;
    float m_CameraDistanceMeters = 0.0f;

  private:
    ViewData(const ViewData&);
    ViewData& operator=(const ViewData&);
  };

  virtual ViewData* newEmptyViewData() = 0; // creates derived struct type
  virtual void internalUpdateViewData() = 0;

  unsigned int m_viewDataRevision = 0;
  std::vector<ViewData*> m_viewData;

  RHIBuffer::ptr m_geoDepthMapTexcoordBuffer;
  RHIBuffer::ptr m_geoDepthMapTristripIndexBuffer;
  RHIBuffer::ptr m_geoDepthMapLineIndexBuffer;
  size_t m_geoDepthMapTristripIndexCount, m_geoDepthMapLineIndexCount;

  // Render settings
  int m_trimLeft = 8, m_trimTop = 8;
  int m_trimRight = 8, m_trimBottom = 8;

  bool m_populateDebugTextures = false;

  void internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1, const glm::mat4& modelMatrix);

private:
  ViewData* viewDataAtIndex(size_t index) { return m_viewData[index]; }
  uint32_t m_internalWidth, m_internalHeight;

  DepthMapGenerator(const DepthMapGenerator&);
  DepthMapGenerator& operator=(const DepthMapGenerator&);
};
