#pragma once
#include "common/CameraSystem.h"
#include "common/FxRenderView.h"
#include "common/DepthMapSHM.h"
#include "common/SHMSegment.h"
#include "common/depthMeshAdaptive.h"
#include "common/fgsFilter.h"
#include "rhi/RHISurface.h"
#include "rhi/RHIBuffer.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/vk/RHIInteropSync.h"
#include "rhi/vk/RHIInteropSurfaceGL.h"
#include "rhi/vk/RHIInteropBufferGL.h"
#include <glm/glm.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/calib3d.hpp>
#include <npp.h>
#include <vector>

class CameraSystem;
class DepthMapGenerator;
class DebugServer;

enum DepthMapGeneratorBackend {
  kDepthBackendNone,
  kDepthBackendMock,
  kDepthBackendDGPU,
  kDepthBackendDepthAI,
  kDepthBackendOFA,
};

DepthMapGeneratorBackend depthBackendStringToEnum(const char* backendStr);
DepthMapGenerator* createDepthMapGenerator(DepthMapGeneratorBackend);

class DepthMapGenerator {
public:
  DepthMapGenerator(DepthMapGeneratorBackend);
  ~DepthMapGenerator();

  void initWithCameraSystem(CameraSystem*);
  void processFrame();
  void renderDisparityDepthMapStereo(size_t viewIdx, const FxRenderView& leftRenderView, const FxRenderView& rightRenderView);
  void renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView);
  void renderIMGUI();
  void renderIMGUIPerformanceGraphs();

  RHISurface::ptr disparitySurface(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_disparityTexture : RHISurface::ptr(); }
  RHISurface::ptr leftGrayscale(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_leftGray : RHISurface::ptr(); }
  RHISurface::ptr rightGrayscale(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_rightGray : RHISurface::ptr(); }
  RHISurface::ptr confidenceSurface(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_confidenceTexture : RHISurface::ptr(); }
  RHISurface::ptr debugResidualSurface(size_t viewIdx) const { return (viewIdx < m_viewData.size()) ? m_viewData[viewIdx]->m_debugResidual : RHISurface::ptr(); }

  // controls availability of leftGrayscale and rightGrayscale
  void setPopulateDebugTextures(bool value) { m_populateDebugTextures = value; }

  bool loadSettings();
  void saveSettings();

  DepthMapGeneratorBackend backend() const { return m_backend; }

  uint32_t maxDisparityPixels() const { return m_maxDisparityPixels; } // maximum disparity value supported by the backend -- pixel units
  uint32_t maxDisparityRaw() const { return m_maxDisparityPixels << m_disparitySubpixelBits; } // maximum disparity value supported by the backend -- raw/subpixel units
  float disparityPrescale() const { return 1.0f / static_cast<float>(1 << m_disparitySubpixelBits); } // multiplier to convert the raw values in disparitySurface to pixel units, accounting for the subpixel bits.

  void setDebugDisparityCPUAccessEnabled(bool v) { m_debugDisparityCPUAccessEnabled = v; }
  bool debugDisparityCPUAccessEnabled() const { return m_debugDisparityCPUAccessEnabled; }

  float debugPeekDisparityTexel(size_t viewIdx, glm::ivec2 texelCoord) const;
  float debugPeekDisparityUV(size_t viewIdx, glm::vec2 uv) const;
  glm::vec3 debugPeekLocalPositionUV(size_t viewIdx, glm::vec2 uv) const;
  glm::vec3 debugPeekLocalPositionTexel(size_t viewIdx, glm::ivec2 texelCoord) const;
  float debugComputeDepthForDisparity(size_t viewIdx, float disparityPixels) const;
  float debugPeekConfidenceTexel(size_t viewIdx, glm::ivec2 texelCoord) const;
  float debugPeekConfidenceUV(size_t viewIdx, glm::vec2 uv) const;
  uint8_t debugPeekResidualTexel(size_t viewIdx, glm::ivec2 texelCoord) const;
  uint8_t debugPeekResidualUV(size_t viewIdx, glm::vec2 uv) const;

  float debugDisparityScale() const { return m_debugDisparityScale; }
  void setDebugDisparityScale(float v) { m_debugDisparityScale = v; }

  bool debugUseFixedDisparity() const { return m_debugUseFixedDisparity; }
  void setDebugUseFixedDisparity(bool v) { m_debugUseFixedDisparity = v; }

protected:
  DepthMapGeneratorBackend m_backend;

  // CUDA and NPP stream context
  cv::cuda::Stream m_globalStream;
  NppStreamContext m_nppStreamContext;

  // CUDA-to-RHI sync object
  RHIInteropSync::ptr m_interopSync;

  virtual void internalLoadSettings(cv::FileStorage&) = 0;
  virtual void internalSaveSettings(cv::FileStorage&) = 0;
  virtual void internalRenderIMGUI() = 0;
  virtual void internalRenderIMGUIPerformanceGraphs() = 0;
  virtual void internalProcessFrame() = 0;
  virtual void internalPostInitWithCameraSystem(); // optional override, called after initWithCameraSystem()

  // Data format controls that should be set in the backend
  uint32_t m_algoInputWidth, m_algoInputHeight; // Defaults to m_internalWidth / m_internalHeight
  uint32_t m_algoDownsampleX = 1;
  uint32_t m_algoDownsampleY = 1;
  uint32_t m_maxDisparityPixels = 128; // pixel units
  uint32_t m_disparitySubpixelBits = 0; // subpixel bits in raw disparity

  float m_debugDisparityScale = 1.0f;
  bool m_debugDisparityCPUAccessEnabled = false;

  // Convenience accessors for the size of the input camera stream.
  uint32_t cameraStreamWidth() const { return m_cameraSystem->cameraProvider()->streamWidth(); }
  uint32_t cameraStreamHeight() const { return m_cameraSystem->cameraProvider()->streamHeight(); }

  // Input size of the L/R camera surfaces for the depth algorithm. Usually the same as internalWidth/internalHeight.
  // The debug L/R views will be this size.
  uint32_t algoInputWidth() const { return m_algoInputWidth; }
  uint32_t algoInputHeight() const { return m_algoInputHeight; }

  // Algorithm internal width/height. Processing happens at this resolution, disparity output is at this size.
  uint32_t internalWidth() const { return m_internalWidth; }
  uint32_t internalHeight() const { return m_internalHeight; }

  void internalFinalizeDisparityTexture();

  CameraSystem* m_cameraSystem = NULL;

  struct ViewData {
    ViewData() {}
    virtual ~ViewData();

    // Rebuild the CUtexObject wrappers around m_rectifiedLuma[]. Call after
    // any reallocation of those mats (i.e. after internalUpdateViewData()).
    void rebuildRectifiedLumaTextures();

    bool m_isStereoView = false;
    bool m_isVerticalStereo = false;
    size_t m_leftCameraIndex = 0, m_rightCameraIndex = 0;
    bool m_leftCameraStreamFailed = false;
    bool m_rightCameraStreamFailed = false;
    bool anyCameraStreamFailed() const { return m_leftCameraStreamFailed || m_rightCameraStreamFailed; }

    // Controls which disparity mat we're writing to.
    uint8_t m_currentMatWriteIndex = 0;
    // Called every frame before allowing the derived impl to write to currentDisparityMat
    void swapCurrentAndPreviousDisparity() { m_currentMatWriteIndex = 1 - m_currentMatWriteIndex; }

    glm::mat3 m_R1;
    glm::vec4 m_depthParameters; // Parameters extracted from the view's stereoDisparityToDepth matrix


    void updateDisparityTexture(DepthMapGenerator*, uint32_t w, uint32_t h, RHISurfaceFormat);

    // Current and previous frame mats
    cv::cuda::GpuMat& currentDisparityMat() { return m_disparityGpuMat[m_currentMatWriteIndex]; }
    cv::cuda::GpuMat& previousDisparityMat() { return m_disparityGpuMat[1 - m_currentMatWriteIndex]; }

    cv::cuda::GpuMat m_disparityGpuMat[2];
    cv::cuda::GpuMat m_disparityConfidence;
    cv::cuda::GpuMat m_disparityDebugResidual;

    cv::cuda::GpuMat m_disparityMedianFilterDestGpuMat;

    FGSFilterState m_fgsFilterState;
    cv::cuda::GpuMat m_fgsFilterInOutPacked;

    // Per-camera rectified luma at the backend's algoInput resolution
    // (1x or 2x the disparity resolution). Owned by the backend; the base
    // class wraps each in a CUtexObject for use in filtering.
    cv::cuda::GpuMat m_rectifiedLuma[2];
    CUtexObject m_rectifiedLumaTex[2] = {0, 0};

    RHIInteropSurfaceGL::ptr m_disparityTexture;
    RHIInteropSurfaceGL::ptr m_leftGray, m_rightGray, m_confidenceTexture, m_debugResidual;

    cv::Mat m_debugCPUDisparityInput[2]; // L/R inputs to stereo matching algorithm
    cv::Mat m_debugCPUDisparity;
    cv::Mat m_debugCPUConfidence;
    cv::Mat m_debugCPUDisparityResidual; // Generic debugging output from the disparity generation process for the debug server; CV_8U / uint8_t.

    // Adaptive-mesh path: variable-size triangle mesh built per-frame in CUDA from the
    // post-processed disparity. Buffers are GPU-private and CUDA-mapped during the build,
    // then drawn via glDrawElementsIndirect. Sized for the worst case (every cell is its
    // own quad) so they never need reallocation.
    RHIInteropBufferGL::ptr m_adaptiveVertexBuffer;
    RHIInteropBufferGL::ptr m_adaptiveIndexBuffer;
    RHIInteropBufferGL::ptr m_adaptiveIndirectArgsBuffer; // 2x DrawElementsIndirectCommand: [stereo, mono]
    DepthMeshAdaptiveScratch m_adaptiveScratch;

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

  // Processing settings
  bool m_useMedianFilter = true;
  bool m_useTemporalFilter = true;
  float m_temporalFilterStableThreshold = 8.0f;
  uint8_t m_temporalFilterAlpha = 48;

  // Stereo half-occlusion mask pre-pass: zeroes confidence on pixels in the
  // "shadow" to the left of foreground objects (no possible match in right
  // view) before any of the filters see them.
  bool m_useOcclusionMask = true;
  // 0 = derive from max disparity (cover the worst-case shadow width).
  uint32_t m_occlusionMaskSearchWindow = 0;
  // Slack in pixels on the geometric occlusion test; smaller is more
  // aggressive about marking near-edge pixels as occluded.
  float m_occlusionMaskHysteresis = 2.0f;
  // Pixels whose own confidence is >= this value are treated as having a
  // real right-view match and skipped by the mask. Tune with the cost-to-
  // confidence curve: should sit a bit below the value that high-quality
  // OFA matches typically produce.
  uint8_t m_occlusionMaskConfidenceCeiling = 192;
  // After marking a pixel occluded, also overwrite its disparity with the
  // nearest high-confidence disparity to the left (background fill). The
  // scan distance reuses the search window.
  bool m_occlusionMaskSmear = true;

  // Fast Global Smoother filter settings
  bool m_useFGSFilter = true;
  float m_fgsLambda = 8000.0f;
  float m_fgsSigmaColor = 1.0f;
  uint8_t m_fgsIterations = 3;
  float m_fgsLambdaAttenuation = 0.25f;

  // Render settings
  int m_trimLeft = 8, m_trimTop = 8;
  int m_trimRight = 8, m_trimBottom = 8;
  bool m_splitDepthDiscontinuity = false;
  float m_maxDepthDiscontinuity = 1.0f;
  float m_minDepthCutoff = 0.050f;
  bool m_usePointRendering = false;
  float m_pointScale = 1.0f;

  // Adaptive-mesh path: maximum (max - min) raw disparity within a block that still
  // counts as flat. Smaller = more subdivision (more triangles); larger = coarser merges
  // and more visible cracks at level transitions.
  int m_adaptiveFlatnessThreshold = 24;

  // Adaptive-mesh path: max |sampled - lerped| disparity at a level-transition edge that
  // still gets T-junction stitched. Steps larger than this stay as cracks so genuine
  // depth discontinuities remain visible.
  int m_adaptiveDepthDiscontinuityThreshold = 40;

  bool m_populateDebugTextures = false;

  bool m_debugUseFixedDisparity = false;
  int m_debugFixedDisparityValue = 1;

  bool internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1);
  RHIRenderPipeline::ptr m_disparityDepthMapPointsPipeline;
  RHIRenderPipeline::ptr m_disparityDepthMapAdaptivePipeline;

  // Profiling data
  CUevent m_finalizeDisparityStartEvent;
  CUevent m_finalizeDisparityFinishedEvent;
  float m_finalizeDisparityTimeMs = 0.0f;

private:
  ViewData* viewDataAtIndex(size_t index) const { return m_viewData[index]; }
  uint32_t m_internalWidth, m_internalHeight;

  DepthMapGenerator(const DepthMapGenerator&);
  DepthMapGenerator& operator=(const DepthMapGenerator&);

  friend class DebugServer;
};
