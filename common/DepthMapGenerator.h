#pragma once
#include "common/CameraSystem.h"
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
  kDepthBackendDepthAI,
  kDepthBackendVPI
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
  void renderIMGUIPerformanceGraphs();

  RHISurface::ptr disparitySurface(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_disparityTexture : RHISurface::ptr(); }
  RHISurface::ptr leftGrayscale(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_leftGray : RHISurface::ptr(); }
  RHISurface::ptr rightGrayscale(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_rightGray : RHISurface::ptr(); }

  // controls availability of leftGrayscale and rightGrayscale
  void setPopulateDebugTextures(bool value) { m_populateDebugTextures = value; }

  bool loadSettings();
  void saveSettings();

  DepthMapGeneratorBackend backend() const { return m_backend; }

  uint32_t maxDisparity() const { return m_maxDisparity; } // maximum disparity value supported by the backend, pixel units
  float disparityPrescale() const { return m_disparityPrescale; } // multiplier to convert the raw values in disparitySurface to pixel units
  bool isFP16Disparity() const { return m_useFP16Disparity; }

  void setDebugDisparityCPUAccessEnabled(bool v) { m_debugDisparityCPUAccessEnabled = v; }
  bool debugDisparityCPUAccessEnabled() const { return m_debugDisparityCPUAccessEnabled; }

  float debugPeekDisparityTexel(size_t viewIdx, glm::ivec2 texelCoord) const;
  float debugPeekDisparityUV(size_t viewIdx, glm::vec2 uv) const;
  glm::vec3 debugPeekLocalPositionUV(size_t viewIdx, glm::vec2 uv) const;
  glm::vec3 debugPeekLocalPositionTexel(size_t viewIdx, glm::ivec2 texelCoord) const;
  float debugComputeDepthForDisparity(size_t viewIdx, float disparityPixels) const;

  float debugDisparityScale() const { return m_debugDisparityScale; }
  void setDebugDisparityScale(float v) { m_debugDisparityScale = v; }

  bool debugUseFixedDisparity() const { return m_debugUseFixedDisparity; }
  void setDebugUseFixedDisparity(bool v) { m_debugUseFixedDisparity = v; }

protected:

  DepthMapGeneratorBackend m_backend;

  virtual void internalLoadSettings(cv::FileStorage&) = 0;
  virtual void internalSaveSettings(cv::FileStorage&) = 0;
  virtual void internalRenderIMGUI() = 0;
  virtual void internalRenderIMGUIPerformanceGraphs() = 0;
  virtual void internalProcessFrame() = 0;

  // Data format controls that should be set in the backend
  uint32_t m_algoDownsampleX = 1;
  uint32_t m_algoDownsampleY = 1;
  uint32_t m_maxDisparity = 128;
  float m_disparityPrescale = 1.0f;
  bool m_useFP16Disparity = false;

  float m_debugDisparityScale = 1.0f;
  bool m_debugDisparityCPUAccessEnabled = false;

  uint32_t inputWidth() const { return m_cameraSystem->cameraProvider()->streamWidth(); }
  uint32_t inputHeight() const { return m_cameraSystem->cameraProvider()->streamHeight(); }

  uint32_t internalWidth() const { return m_internalWidth; }
  uint32_t internalHeight() const { return m_internalHeight; }

  void internalGenerateDisparityMips();

  CameraSystem* m_cameraSystem = NULL;

  struct ViewData {
    ViewData() {}
    virtual ~ViewData() {
      free(m_debugCPUDisparity);
    }

    bool m_isStereoView = false;
    bool m_isVerticalStereo = false;
    size_t m_leftCameraIndex = 0, m_rightCameraIndex = 0;

    glm::mat4 m_R1inv, m_Q, m_Qinv;


    void updateDisparityTexture(uint32_t w, uint32_t h, RHISurfaceFormat);

    RHISurface::ptr m_disparityTexture;
    RHISurface::ptr m_leftGray, m_rightGray;

    std::vector<RHIRenderTarget::ptr> m_disparityTextureMipTargets;
    float m_CameraDistanceMeters = 0.0f;

    void* m_debugCPUDisparity = nullptr;
    uint8_t m_debugCPUDisparityBytesPerPixel = 0;

    void ensureDebugCPUAccessEnabled(uint8_t disparityBytesPerPixel); // requires disparity texture to exist for array sizing

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

  RHIBuffer::ptr m_geoDepthMapPointTexcoordBuffer;
  RHIBuffer::ptr m_geoDepthMapPointTristripIndexBuffer;
  size_t m_geoDepthMapTristripIndexCount, m_geoDepthMapLineIndexCount, m_geoDepthMapPointTristripIndexCount;

  // Render settings
  int m_trimLeft = 8, m_trimTop = 8;
  int m_trimRight = 8, m_trimBottom = 8;
  bool m_splitDepthDiscontinuity = false;
  float m_maxDepthDiscontinuity = 1.0f;
  float m_minDepthCutoff = 0.050f;
  bool m_usePointRendering = true;
  float m_pointScale = 1.0f;

  bool m_populateDebugTextures = false;

  bool m_debugUseFixedDisparity = false;
  int m_debugFixedDisparityValue = 1;

  void internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1, const glm::mat4& modelMatrix);
  RHIRenderPipeline::ptr m_disparityDepthMapPipeline;
  RHIRenderPipeline::ptr m_disparityDepthMapPointsPipeline;
  RHIComputePipeline::ptr m_disparityMipComputePipeline;

private:
  ViewData* viewDataAtIndex(size_t index) const { return m_viewData[index]; }
  uint32_t m_internalWidth, m_internalHeight;

  DepthMapGenerator(const DepthMapGenerator&);
  DepthMapGenerator& operator=(const DepthMapGenerator&);
};
