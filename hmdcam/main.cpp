#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <dlfcn.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/scoped_ptr.hpp>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/gl/GLCommon.h" // must be included before cudaEGL
#include <cudaEGL.h>
#include <cuda_egl_interop.h>

#include "ArgusCamera.h"
#include "common/CameraSystem.h"
#include "common/DepthMapGenerator.h"
#include "common/FxThreading.h"
#include "common/ScrollingBuffer.h"
#include "common/Timing.h"
#include "common/glmCvInterop.h"
#include "FocusAssistDebugOverlay.h"
#include "IDebugOverlay.h"
#include "InputListener.h"
#include "Render.h"
#include "RenderBackend.h"

#include "imgui_backend.h"
#include "implot/implot.h"

#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "rdma/RDMAContext.h"

//#define LATENCY_DEBUG

// Camera render parameters
float zoomFactor = 1.0f;
float stereoOffset = 0.0f;
bool useMask = true;
float panoClipOffset = 0.0f;
float panoClipScale = 1.0f;
float panoTxScale = 1.0f;
bool debugUseDistortion = true;
float uiScale = 0.2f;
float uiDepth = 0.4f;


uint64_t settingsDirtyFrame = 0; // 0 if not dirty, frame number otherwise.
const int kSettingsAutosaveIntervalSeconds = 10;
uint64_t settingsAutosaveIntervalFrames = 1000; // will be recomputed when we know the framerate based on kSettingsAutosaveIntervalSeconds

// Camera info/state
IArgusCamera* argusCamera;
CameraSystem* cameraSystem;


RDMAContext* rdmaContext;

#define readNode(node, settingName) cv::read(node[#settingName], settingName, settingName)
static const char* hmdcamSettingsFilename = "hmdcamSettings.yml";
bool loadSettings() {
  cv::FileStorage fs(hmdcamSettingsFilename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("Unable to open settings file %s\n", hmdcamSettingsFilename);
    return false;
  }

  try {
    readNode(fs, zoomFactor);
    readNode(fs, stereoOffset);
    readNode(fs, useMask);
    readNode(fs, panoClipOffset);
    readNode(fs, panoClipScale);
    readNode(fs, panoTxScale);
    readNode(fs, uiScale);
    readNode(fs, uiDepth);

    float ev = argusCamera->exposureCompensation();
    cv::read(fs["exposureCompensation"], ev, ev);
    argusCamera->setExposureCompensation(ev);

    glm::vec2 acRegionCenter = argusCamera->acRegionCenter();
    glm::vec2 acRegionSize = argusCamera->acRegionSize();
    cv::read(fs["acRegionCenterX"], acRegionCenter.x, acRegionCenter.x);
    cv::read(fs["acRegionCenterY"], acRegionCenter.y, acRegionCenter.y);
    cv::read(fs["acRegionSizeX"], acRegionSize.x, acRegionSize.x);
    cv::read(fs["acRegionSizeY"], acRegionSize.y, acRegionSize.y);
    argusCamera->setAcRegion(acRegionCenter, acRegionSize);

    // cv doesn't support int64_t, so we cast to double
    double captureDurationOffset = static_cast<double>(argusCamera->captureDurationOffset());
    readNode(fs, captureDurationOffset);
    argusCamera->setCaptureDurationOffset(captureDurationOffset);
  } catch (const std::exception& ex) {
    printf("Unable to load hmdcam settings: %s\n", ex.what());
    return false;
  }
  return true;
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, settingName)
void saveSettings() {
  try {
    cv::FileStorage fs(hmdcamSettingsFilename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);

    writeNode(fs, zoomFactor);
    writeNode(fs, stereoOffset);
    writeNode(fs, useMask);
    writeNode(fs, panoClipOffset);
    writeNode(fs, panoClipScale);
    writeNode(fs, panoTxScale);
    writeNode(fs, uiScale);
    writeNode(fs, uiDepth);

    fs.write("exposureCompensation", argusCamera->exposureCompensation());

    glm::vec2 acRegionCenter = argusCamera->acRegionCenter();
    glm::vec2 acRegionSize = argusCamera->acRegionSize();
    fs.write("acRegionCenterX", acRegionCenter.x);
    fs.write("acRegionCenterY", acRegionCenter.y);
    fs.write("acRegionSizeX", acRegionSize.x);
    fs.write("acRegionSizeY", acRegionSize.y);

    // cv doesn't support int64_t, so we cast to double
    double captureDurationOffset = static_cast<double>(argusCamera->captureDurationOffset());
    writeNode(fs, captureDurationOffset);
  } catch (const std::exception& ex) {
    printf("Unable to save hmdcam settings: %s\n", ex.what());
  }
  settingsDirtyFrame = 0;
}
#undef writeNode


// Profiling data
struct FrameTimingData {
  FrameTimingData() : captureTimeMs(0), submitTimeMs(0), captureLatencyMs(0), captureIntervalMs(0), captureIntervalAdjustmentMarker(0) {}

  float captureTimeMs;
  float submitTimeMs;

  float captureLatencyMs;
  float captureIntervalMs;

  float captureIntervalAdjustmentMarker;
};

ScrollingBuffer<FrameTimingData> s_timingDataBuffer(512);

// Inter-sensor timing data
struct SensorTimingData {
  SensorTimingData() { memset(timestampDelta, 0, sizeof(float) * 16); }

  float timestampDelta[16];
};
ScrollingBuffer<SensorTimingData> s_sensorTimingData(512);


bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;

  // Restore signal handlers so the program is still interruptable if clean shutdown gets stuck
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
}


const size_t DRAW_FLAGS_USE_MASK = 0x01;

void renderDrawCamera(size_t cameraIdx, size_t flags, RHISurface::ptr distortionMap, RHISurface::ptr overlayTexture /*can be null*/, glm::mat4 modelViewProjection, float minU = 0.0f, float maxU = 1.0f) {

  bool useClippedQuadUB = false;

  if (overlayTexture) {
    if (distortionMap) {
      rhi()->bindRenderPipeline(camUndistortOverlayPipeline);
      rhi()->loadTexture(ksDistortionMap, distortionMap);
    } else {
      rhi()->bindRenderPipeline(camOverlayPipeline);
    }

    rhi()->loadTexture(ksOverlayTex, overlayTexture, linearClampSampler);

  } else if (!distortionMap) {
    // no distortion or overlay
    rhi()->bindRenderPipeline(camTexturedQuadPipeline);
      
  } else {
    rhi()->bindRenderPipeline(camUndistortMaskPipeline);
    useClippedQuadUB = true;
    rhi()->loadTexture(ksDistortionMap, distortionMap);
    if (flags & DRAW_FLAGS_USE_MASK) {
      rhi()->loadTexture(ksMaskTex, cameraSystem->cameraAtIndex(cameraIdx).mask);
    } else {
      rhi()->loadTexture(ksMaskTex, disabledMaskTex, linearClampSampler);
    }
  }

  rhi()->loadTexture(ksImageTex, argusCamera->rgbTexture(cameraIdx), linearClampSampler);

  if (useClippedQuadUB) {
    NDCClippedQuadUniformBlock ub;
    ub.modelViewProjection = modelViewProjection;
    ub.minUV = glm::vec2(minU, 0.0f);
    ub.maxUV = glm::vec2(maxU, 1.0f);

    rhi()->loadUniformBlockImmediate(ksNDCClippedQuadUniformBlock, &ub, sizeof(NDCClippedQuadUniformBlock));
  } else {
    NDCQuadUniformBlock ub;
    ub.modelViewProjection = modelViewProjection;

    rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));
  }
  rhi()->drawNDCQuad();
}

int main(int argc, char* argv[]) {

  DepthMapGeneratorBackend depthBackend = kDepthBackendNone;
  ERenderBackend renderBackendType = kRenderBackendVKDirect;
  bool enableRDMA = true;
  bool debugInitOnly = false;
  bool debugMockCameras = false;
  bool debugNoRepeatingCapture = false;
  int rdmaInterval = 2;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth-backend")) {
      if (i == (argc - 1)) {
        printf("--depth-backend: requires argument\n");
        return 1;
      }
      depthBackend = depthBackendStringToEnum(argv[++i]);
    } else if (!strcmp(argv[i], "--render-backend")) {
      if (i == (argc - 1)) {
        printf("--render-backend: requires argument\n");
        return 1;
      }
      renderBackendType = renderBackendStringToEnum(argv[++i]);
    } else if (!strcmp(argv[i], "--disable-rdma")) {
      enableRDMA = false;
    } else if (!strcmp(argv[i], "--debug-init-only")) {
      debugInitOnly = true;
    } else if (!strcmp(argv[i], "--debug-no-repeating-capture")) {
      debugNoRepeatingCapture = true;
    } else if (!strcmp(argv[i], "--debug-mock-cameras")) {
      debugMockCameras = true;
    } else if (!strcmp(argv[i], "--rdma-interval")) {
      if (i == (argc - 1)) {
        printf("--rdma-interval: requires argument\n");
        return 1;
      }
      rdmaInterval = atoi(argv[++i]);
      if (rdmaInterval <= 0) {
        printf("--rdma-interval: invalid argument\n");
        return 1;
      }
    } else {
      printf("Unrecognized argument %s\n", argv[i]);
      return 1;
    }
  }

  DepthMapGenerator* depthMapGenerator = createDepthMapGenerator(depthBackend);

  if (depthBackend == kDepthBackendDepthAI) {
    // Set thread affinity.
    // On Tegra, we get a small but noticeable performance improvement by pinning the DepthAI backend to CPU0-1 and hmdcam to all other CPUs.
    // This must be done early in initialization so that all of the library worker threads spawned later inherit these settings.

    cpu_set_t cpuset;
    // Create affinity mask for all CPUs besides CPU0-1
    CPU_ZERO(&cpuset);
    for (size_t i = 2; i < CPU_SETSIZE; ++i)
      CPU_SET(i, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
      perror("pthread_setaffinity");
    }
  }

  int rdmaFrame = 0;
  if (enableRDMA) {
    rdmaContext = RDMAContext::createServerContext();

    if (!rdmaContext) {
      printf("RDMA server context initialization failed; RDMA service will be unavailable.\n");
    }
  }

  startInputListenerThread();
  if (!RenderInit(renderBackendType)) {
    printf("RenderInit() failed\n");
    return 1;
  }

  settingsAutosaveIntervalFrames = kSettingsAutosaveIntervalSeconds * static_cast<unsigned int>(renderBackend->refreshRateHz());

  FxThreading::detail::init();

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = NULL; // Disable INI file load/save
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Keyboard navigation (mapped from the InputListener media remote interface)
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Gamepad navigation (not used right now)
  ImGui_ImplInputListener_Init();
  ImGui_ImplFxRHI_Init();

  io.DisplaySize = ImVec2(512.0f, 512.0f); // Not the full size, but the size of our overlay RT
  io.DisplayFramebufferScale = ImVec2(2.0f, 2.0f); // Use HiDPI rendering

  // Open the cameras

  if (debugMockCameras) {
    argusCamera = new ArgusCameraMock(4, 1920, 1080, 90.0);
  } else {
    argusCamera = new ArgusCamera(renderBackend->eglDisplay(), renderBackend->eglContext(), renderBackend->refreshRateHz());
  }

  std::vector<RHIRect> debugSurfaceCameraRects;
  {
    // Size and allocate debug surface area based on camera count
    unsigned int debugColumns = 1, debugRows = 1;
    if (argusCamera->streamCount() > 1) {
      debugColumns = 2;
      debugRows = (argusCamera->streamCount() + 1) / 2; // round up
    }
    uint32_t dsW = debugColumns * argusCamera->streamWidth();
    uint32_t dsH = debugRows * argusCamera->streamHeight();
    printf("Debug stream: selected a %ux%u layout on a %ux%u surface for %zu cameras\n", debugColumns, debugRows, dsW, dsH, argusCamera->streamCount());

    for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
      unsigned int col = cameraIdx % debugColumns;
      unsigned int row = cameraIdx / debugColumns;
      RHIRect r = RHIRect::xywh(col * argusCamera->streamWidth(), row * argusCamera->streamHeight(), argusCamera->streamWidth(), argusCamera->streamHeight());
      printf("  [%zu] (%ux%u) +(%u, %u)\n", cameraIdx, r.width, r.height, r.x, r.y);
      debugSurfaceCameraRects.push_back(r);
    }
    RenderInitDebugSurface(dsW, dsH);
  }

  cameraSystem = new CameraSystem(argusCamera);
  // Load whatever calibration we have (may be nothing)
  cameraSystem->loadCalibrationData();

  if (cameraSystem->views() == 0) {
    printf("No calibration data, creating a stub\n");
    if (cameraSystem->cameras() == 1) {
      cameraSystem->createMonoView(0);
    } else {
      cameraSystem->createStereoView(0, 1);
    }
    cameraSystem->saveCalibrationData();
  }


  if (depthMapGenerator) {
    depthMapGenerator->initWithCameraSystem(cameraSystem);
    depthMapGenerator->loadSettings();
  }


  // Create the RDMA configuration buffer and camera data buffers
  RDMABuffer::ptr configBuf;
  std::vector<RDMABuffer::ptr> rdmaCameraLumaBuffers;
  std::vector<RDMABuffer::ptr> rdmaCameraChromaBuffers;
  CUDA_RESOURCE_DESC lumaResourceDescriptor;
  memset(&lumaResourceDescriptor, 0, sizeof(lumaResourceDescriptor));

  CUDA_RESOURCE_DESC chromaResourceDescriptor;
  memset(&chromaResourceDescriptor, 0, sizeof(chromaResourceDescriptor));

  if (rdmaContext) {
    CUeglColorFormat eglColorFormat;

    for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
      CUgraphicsResource rsrc = argusCamera->cudaGraphicsResource(cameraIdx);
      if (!rsrc) {
        printf("ArgusCamera failed to provide CUgraphicsResource for stream %zu\n", cameraIdx);
        break;
      }

      // Using the Runtime API here instead since it gives better information about multiplanar formats
      cudaEglFrame eglFrame;
      CUDA_CHECK(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, (cudaGraphicsResource_t) rsrc, /*cubemapIndex=*/ 0, /*mipLevel=*/ 0));
      assert(eglFrame.frameType == cudaEglFrameTypePitch);
      assert(eglFrame.planeCount == 2);

      if (cameraIdx == 0) {
        eglColorFormat = (CUeglColorFormat) eglFrame.eglColorFormat; // CUeglColorFormat and cudaEglColorFormat are interchangeable

        // Convert eglFrame to resource descriptors.
        // We don't fill the device pointers, we're just going to serialize the contents
        // of these descriptors to populate the config buffer.
        lumaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
        lumaResourceDescriptor.res.pitch2D.devPtr       = 0;
        lumaResourceDescriptor.res.pitch2D.format       = CU_AD_FORMAT_UNSIGNED_INT8; // eglFrame.planeDesc[0].channelDesc.f; // TODO
        lumaResourceDescriptor.res.pitch2D.numChannels  = eglFrame.planeDesc[0].numChannels;
        lumaResourceDescriptor.res.pitch2D.width        = eglFrame.planeDesc[0].width;
        lumaResourceDescriptor.res.pitch2D.height       = eglFrame.planeDesc[0].height;
        lumaResourceDescriptor.res.pitch2D.pitchInBytes = eglFrame.planeDesc[0].pitch;

        // TODO hardcoded assumptions about chroma format -- we should be able to get this from the eglColorFormat!
        chromaResourceDescriptor.res.pitch2D.devPtr       = 0;
        chromaResourceDescriptor.res.pitch2D.format       = CU_AD_FORMAT_UNSIGNED_INT8; // eglFrame.planeDesc[1].channelDesc.f // TODO
        chromaResourceDescriptor.res.pitch2D.numChannels  = 2; // eglFrame.planeDesc[1].numChannels; // TODO
        chromaResourceDescriptor.res.pitch2D.width        = eglFrame.planeDesc[1].width;
        chromaResourceDescriptor.res.pitch2D.height       = eglFrame.planeDesc[1].height;
        // pitchInBytes NOTE: "...in case of multiplanar *eglFrame, pitch of only first plane is to be considered by the application."
        // (accessing planeDesc[0] is intentional)
        chromaResourceDescriptor.res.pitch2D.pitchInBytes = eglFrame.planeDesc[0].pitch;

        printf("Stream [%zu]:   Luma: %zu x %zu NumChannels=%u ChannelDesc=0x%x (%d,%d,%d,%d) pitchInBytes=%zu\n", cameraIdx,
          lumaResourceDescriptor.res.pitch2D.width, lumaResourceDescriptor.res.pitch2D.height,
          lumaResourceDescriptor.res.pitch2D.numChannels,
          eglFrame.planeDesc[0].channelDesc.f, eglFrame.planeDesc[0].channelDesc.x, eglFrame.planeDesc[0].channelDesc.y,
          eglFrame.planeDesc[0].channelDesc.z, eglFrame.planeDesc[0].channelDesc.w,
          lumaResourceDescriptor.res.pitch2D.pitchInBytes);
        printf("Stream [%zu]: Chroma: %zu x %zu NumChannels=%u ChannelDesc=0x%x (%d,%d,%d,%d) pitchInBytes=%zu\n", cameraIdx,
          chromaResourceDescriptor.res.pitch2D.width, chromaResourceDescriptor.res.pitch2D.height,
          chromaResourceDescriptor.res.pitch2D.numChannels,
          eglFrame.planeDesc[1].channelDesc.f, eglFrame.planeDesc[1].channelDesc.x, eglFrame.planeDesc[1].channelDesc.y,
          eglFrame.planeDesc[1].channelDesc.z, eglFrame.planeDesc[1].channelDesc.w,
          chromaResourceDescriptor.res.pitch2D.pitchInBytes);
      }

      // TODO handle other type-sizes
      assert(lumaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_UNSIGNED_INT8 || lumaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_SIGNED_INT8);
      assert(chromaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_UNSIGNED_INT8 || chromaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_SIGNED_INT8);

      char bufferName[32];
      sprintf(bufferName, "camera%zu_luma", cameraIdx);
      rdmaCameraLumaBuffers.push_back(rdmaContext->newManagedBuffer(bufferName,
        lumaResourceDescriptor.res.pitch2D.height * lumaResourceDescriptor.res.pitch2D.pitchInBytes, kRDMABufferUsageWriteSource));

      sprintf(bufferName, "camera%zu_chroma", cameraIdx);
      rdmaCameraChromaBuffers.push_back(rdmaContext->newManagedBuffer(bufferName, 
        chromaResourceDescriptor.res.pitch2D.height * chromaResourceDescriptor.res.pitch2D.pitchInBytes, kRDMABufferUsageWriteSource));
    }

    if (rdmaCameraLumaBuffers.size() == argusCamera->streamCount() && rdmaCameraChromaBuffers.size() == argusCamera->streamCount()) {
      configBuf = rdmaContext->newManagedBuffer("config", 8192, kRDMABufferUsageWriteSource); // TODO this should be read-source once we implement read

      SerializationBuffer cfg;
      cfg.put_u32(argusCamera->streamCount());
      cfg.put_u32(argusCamera->streamWidth());
      cfg.put_u32(argusCamera->streamHeight());

      cfg.put_u32(eglColorFormat);

      cfg.put_u32(lumaResourceDescriptor.res.pitch2D.format);
      cfg.put_u32(lumaResourceDescriptor.res.pitch2D.numChannels);
      cfg.put_u32(lumaResourceDescriptor.res.pitch2D.width);
      cfg.put_u32(lumaResourceDescriptor.res.pitch2D.height);
      cfg.put_u32(lumaResourceDescriptor.res.pitch2D.pitchInBytes);

      cfg.put_u32(chromaResourceDescriptor.res.pitch2D.format);
      cfg.put_u32(chromaResourceDescriptor.res.pitch2D.numChannels);
      cfg.put_u32(chromaResourceDescriptor.res.pitch2D.width);
      cfg.put_u32(chromaResourceDescriptor.res.pitch2D.height);
      cfg.put_u32(chromaResourceDescriptor.res.pitch2D.pitchInBytes);

      memcpy(configBuf->data(), cfg.data(), cfg.size());
      rdmaContext->asyncFlushWriteBuffer(configBuf);
    } else {
      printf("RDMA camera buffer setup failed -- service will be unavailable.\n");
      rdmaCameraLumaBuffers.clear();
      rdmaCameraChromaBuffers.clear();

      // TODO this just leaks/orphans the context -- we need to implement context shutdown at some point
      rdmaContext = nullptr;
    }
  }


  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);


  // Load masks. 
#if 1
  // TODO move mask loading to CameraSystem
  for (size_t cameraIdx = 0; cameraIdx < cameraSystem->cameras(); ++cameraIdx) {
    cameraSystem->cameraAtIndex(cameraIdx).mask = disabledMaskTex;
  }

#else
  {
    for (size_t cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
      cameraMask[cameraIdx] = rhi()->newTexture2D(s_cameraWidth, s_cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));

      unsigned int x, y, fileChannels;
      char filename[32];
      sprintf(filename, "camera%zu_mask.png", cameraIdx);
      uint8_t* maskData = stbi_load(filename, (int*) &x, (int*) &y, (int*) &fileChannels, 1);
      if (maskData && ((x != s_cameraWidth) || (y != s_cameraHeight))) {
        printf("Mask file \"%s\" dimensions %dx%d do not match camera dimensions %zux%zu. The mask will not be applied.\n", filename, x, y, s_cameraWidth, s_cameraHeight);
        free(maskData);
        maskData = NULL;
      }

      if (!maskData) {
        printf("No usable mask data found in \"%s\" for camera %zu. A template will be created.\n", filename, cameraIdx);

        x = s_cameraWidth;
        y = s_cameraHeight;
        maskData = (uint8_t*) malloc(x * y);

        // Save a snapshot from this camera as a template.
        argusCamera->readFrame();

        RHIRenderTarget::ptr snapRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({cameraMask[cameraIdx]}));
        rhi()->beginRenderPass(snapRT, kLoadInvalidate);
        // This pipeline flips the Y axis for OpenCV's coordinate system, which is the same as the PNG coordinate system
        rhi()->bindRenderPipeline(camGreyscalePipeline);
        rhi()->loadTexture(ksImageTex, argusCamera->rgbTexture(cameraIdx), linearClampSampler);
        rhi()->drawNDCQuad();
        rhi()->endRenderPass(snapRT);

        rhi()->readbackTexture(cameraMask[cameraIdx], 0, kVertexElementTypeUByte1N, maskData);
        char templateFilename[32];
        sprintf(templateFilename, "camera%zu_mask_template.png", cameraIdx);
        stbi_write_png(templateFilename, s_cameraWidth, s_cameraHeight, 1, maskData, /*rowBytes=*/s_cameraWidth);

        // Fill a completely white mask for upload
        memset(maskData, 0xff, x * y);
      } else {
        printf("Loaded mask data for camera %zu\n", cameraIdx);
      }

      // Y-flip the image to convert from PNG to GL coordsys
      char* flippedMask = new char[s_cameraWidth * s_cameraHeight];
      for (size_t row = 0; row < s_cameraHeight; ++row) {
        memcpy(flippedMask + (row * s_cameraWidth), maskData + (((s_cameraHeight - 1) - row) * s_cameraWidth), s_cameraWidth);
      }

      rhi()->loadTextureData(cameraMask[cameraIdx], kVertexElementTypeUByte1N, flippedMask);

      delete[] flippedMask;

      free(maskData);
    }
  }
#endif

  {
    // Load initial autocontrol region of interest
    argusCamera->setAcRegion(/*center=*/ glm::vec2(0.5f, 0.5f), /*size=*/ glm::vec2(0.5f, 0.5f));

    // Load settings
    loadSettings();

    // Accumulators to track frame timing statistics
    uint64_t frameCounter = 0;
    boost::accumulators::accumulator_set<double, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
      > > captureLatency;

    uint64_t previousCaptureTimestamp = 0;
    boost::accumulators::accumulator_set<double, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
      > > captureInterval;

    uint64_t previousFrameTimestamp = 0;
    boost::accumulators::accumulator_set<double, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
      > > frameInterval;

    io.DeltaTime = 1.0f / 60.0f; // Will be updated during frame-timing computation

    RHIRenderTarget::ptr guiRT;
    RHISurface::ptr guiTex;

    guiTex = rhi()->newTexture2D(io.DisplaySize.x * io.DisplayFramebufferScale.x, io.DisplaySize.y * io.DisplayFramebufferScale.y, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    guiRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ guiTex }));

    double currentCaptureLatencyMs = 0.0;
    double currentCaptureIntervalMs = 0.0;

    bool drawUI = false;
    bool debugEnableDepthMapGenerator = true;
    int restartSkipFrameCounter = 0;
    boost::scoped_ptr<CameraSystem::CalibrationContext> calibrationContext;
    boost::scoped_ptr<IDebugOverlay> debugOverlay;


    // Warm up the depth map generator. It might need to do expensive initialization tasks once it sees the
    // camera system configuration (like compiling CUDA kernels), so we run a few frames through it
    // before settling in to the timing-sensitive repeating capture loop.
    if (depthMapGenerator) {
      argusCamera->setRepeatCapture(false);
      argusCamera->readFrame();
      uint64_t processFrameStart = currentTimeNs();
      depthMapGenerator->processFrame();
      printf("Depth Map Generator warmup took %.3f ms\n", deltaTimeMs(processFrameStart, currentTimeNs()));
#if 1
      argusCamera->readFrame();
      processFrameStart = currentTimeNs();
      depthMapGenerator->processFrame();
      printf("Depth Map Generator 2nd frame took %.3f ms\n", deltaTimeMs(processFrameStart, currentTimeNs()));
#else
      for (size_t i = 0; i < 100; ++i) {
        uint64_t readFrameStart = currentTimeNs();
        argusCamera->readFrame();
        processFrameStart = currentTimeNs();
        depthMapGenerator->processFrame();
        uint64_t end = currentTimeNs();
        printf("[%zu] readFrame %.3f ms processFrame %.3f ms total %.3f ms\n", i, deltaTimeMs(readFrameStart, processFrameStart), deltaTimeMs(processFrameStart, end), deltaTimeMs(readFrameStart, end));
      }
#endif
    }

    // Start repeating capture
    if (!debugNoRepeatingCapture)
      argusCamera->setRepeatCapture(true);

    // Main display loop
    while (!want_quit) {
      FrameTimingData timingData;
      uint64_t frameStartTimeNs = currentTimeNs();

      if (testButton(kButtonPower)) {
        drawUI = !drawUI;
      }

      // Force UI on if we're calibrating.
      drawUI |= (!!calibrationContext);

      if (!drawUI) {
        // calling testButton eats the inputs, so only do that if we're not drawing the UI.
        if (testButton(kButtonDown)) {
          debugEnableDepthMapGenerator = !debugEnableDepthMapGenerator;
        }
      }

      ImGui_ImplFxRHI_NewFrame();
      ImGui_ImplInputListener_NewFrame();
      ImGui::NewFrame();

      ++frameCounter;

      if (debugInitOnly && frameCounter >= 10) {
        printf("DEBUG: Initialization-only testing requested, exiting now.\n");
        want_quit = true;
      }

      if (restartSkipFrameCounter <= 0) {
        // Only use repeating captures if we're not in calibration. The variable CPU-side delays for calibration image processing usually end up crashing libargus.
        if (!debugNoRepeatingCapture)
          argusCamera->setRepeatCapture(!((bool) calibrationContext));

        argusCamera->readFrame();
      } else {
        restartSkipFrameCounter -= 1;
      }

      timingData.captureTimeMs = deltaTimeMs(frameStartTimeNs, currentTimeNs());

      if (previousCaptureTimestamp) {
        currentCaptureIntervalMs = static_cast<double>(argusCamera->frameSensorTimestamp(0) - previousCaptureTimestamp) / 1000000.0;
        captureInterval(currentCaptureIntervalMs);
      }
      previousCaptureTimestamp = argusCamera->frameSensorTimestamp(0);

      // TODO move this inside CameraSystem
      if (debugEnableDepthMapGenerator && depthMapGenerator && !calibrationContext && !(rdmaContext && rdmaContext->hasPeerConnections())) {
        depthMapGenerator->processFrame();
      }

      if (calibrationContext && calibrationContext->finished()) {
        calibrationContext.reset();
      }

      if (calibrationContext) {
        calibrationContext->processFrame();
      }


      if (calibrationContext || drawUI) {
        // GUI support
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, io.DisplaySize.y), 0, /*pivot=*/ImVec2(0.5f, 1.0f)); // bottom-center aligned
        ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always); // always auto-size to contents, since we don't provide a way to resize the UI
        ImGui::Begin("Overlay", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        bool settingsDirty = false;

        if (calibrationContext) {

          calibrationContext->processUI();

        } else {
          {
            float ev = argusCamera->exposureCompensation();
            if (ImGui::SliderFloat("Exposure", &ev, -10.0f, 10.0f, "%.1f", ImGuiSliderFlags_None)) {
              argusCamera->setExposureCompensation(ev);
              settingsDirty = true;
            }
          }

          if (ImGui::CollapsingHeader("Render/UI Settings")) {
            // settingsDirty |= ImGui::Checkbox("Mask", &useMask); // Disabled for now
            settingsDirty |= ImGui::SliderFloat("Pano Tx Scale", &panoTxScale, 0.0f, 10.0f);
            settingsDirty |= ImGui::SliderFloat("Pano Clip Offset", &panoClipOffset, -0.5f, 0.5f);
            settingsDirty |= ImGui::SliderFloat("Pano Clip Scale", &panoClipScale, 0.0f, 1.0f);
            settingsDirty |= ImGui::SliderFloat("Zoom", &zoomFactor, 0.5f, 2.0f);
            settingsDirty |= ImGui::SliderFloat("Stereo Offset", &stereoOffset, -0.5f, 0.5f);
            {
              glm::vec2 acCenter = argusCamera->acRegionCenter();
              glm::vec2 acSize = argusCamera->acRegionSize();
              bool dirty = ImGui::SliderFloat2("AC Region Center", &acCenter[0], 0.0f, 1.0f);
              dirty |=     ImGui::SliderFloat2("AC Region Size",   &acSize[0],   0.0f, 1.0f);
              if (dirty) {
                argusCamera->setAcRegion(acCenter, acSize);
                settingsDirty = true;
              }
            }
            settingsDirty |= ImGui::SliderFloat("UI Scale", &uiScale, 0.05f, 1.5f);
            settingsDirty |= ImGui::SliderFloat("UI Depth", &uiDepth, 0.2f, 2.5f);
            if (ImGui::Button("Save Settings")) {
              saveSettings();
            }
          }

          if (ImGui::CollapsingHeader("Calibration")) {

            for (size_t cameraIdx = 0; cameraIdx < cameraSystem->cameras(); ++cameraIdx) {
              char caption[64];
              sprintf(caption, "Calibrate camera %zu", cameraIdx);
              if (ImGui::Button(caption)) {
                calibrationContext.reset(cameraSystem->calibrationContextForCamera(cameraIdx));
              }
            }
            for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
              ImGui::PushID(viewIdx);

              CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
              if (v.isStereo) {
                char caption[64];
                sprintf(caption, "Calibrate stereo view %zu", viewIdx);
                if (ImGui::Button(caption)) {
                  calibrationContext.reset(cameraSystem->calibrationContextForView(viewIdx));
                }
                sprintf(caption, "View %zu is Panorama", viewIdx);
                ImGui::Checkbox(caption, &v.isPanorama);
              }

              if (v.isStereo && viewIdx != 0) {
                // Autocalibration for secondary stereo views
                char caption[64];
                sprintf(caption, "Calibrate offset for stereo view %zu", viewIdx);
                if (ImGui::Button(caption)) {
                  calibrationContext.reset(cameraSystem->calibrationContextForStereoViewOffset(0, viewIdx));
                }
              } else {
                // Direct view transform editing
                ImGui::Text("View %zu Transform", viewIdx);
                // convert to and from millimeters for editing
                glm::vec3 txMM = v.viewTranslation * 1000.0f;
                if (ImGui::DragFloat3("Tx", &txMM[0], /*speed=*/ 0.1f, /*min=*/ -500.0f, /*max=*/ 500.0f, "%.1fmm")) {
                  v.viewTranslation = txMM / 1000.0f;
                }
                ImGui::DragFloat3("Rx", &v.viewRotation[0], /*speed=*/0.1f, /*min=*/ -75.0f, /*max=*/ 75.0f, "%.1fdeg");
              }

              ImGui::PopID(); // viewIdx
            }
            if (ImGui::Button("Save Calibration")) {
              cameraSystem->saveCalibrationData();
            }
          } // Calibration header

          if (debugEnableDepthMapGenerator && depthMapGenerator && ImGui::CollapsingHeader("Depth Backend")) {
            depthMapGenerator->renderIMGUI();
            if (ImGui::Button("Save Depth Backend Settings")) {
              depthMapGenerator->saveSettings();
            }
          }
        }

        {
          const auto& meta = argusCamera->frameMetadata(0);
          ImGui::Text("Dur=1/%usec Exp=1/%usec %uISO DGain=%f AGain=%f",
            (unsigned int) (1000000.0f / static_cast<float>(meta.frameDurationNs/1000)), (unsigned int) (1000000.0f / static_cast<float>(meta.sensorExposureTimeNs/1000)), meta.sensorSensitivityISO, meta.ispDigitalGain, meta.sensorAnalogGain);
        }

        // Update inter-sensor timing data
        {
          int64_t sensor0Timestamp = static_cast<int64_t>(argusCamera->frameSensorTimestamp(0));
          SensorTimingData td;
          for (size_t sensorIdx = 1; sensorIdx < argusCamera->streamCount(); ++sensorIdx) {
            td.timestampDelta[sensorIdx - 1] = static_cast<double>(static_cast<int64_t>(argusCamera->frameSensorTimestamp(sensorIdx)) - sensor0Timestamp) / 1000000.0;
          }
          s_sensorTimingData.push_back(td);
        }

        // Skip perf data to save UI space if we're calibrating
        if (!calibrationContext && ImGui::CollapsingHeader("Performance")) {
          int plotFlags = ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;

          if (ImPlot::BeginPlot("##FrameTiming", ImVec2(-1,150), /*flags=*/ plotFlags)) {
              ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
              ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);
              ImPlot::SetupAxisLimits(ImAxis_X1, 0, s_timingDataBuffer.size(), ImPlotCond_Always);
              ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 12.0f, ImPlotCond_Always);
              ImPlot::SetupFinish();

              ImPlot::PlotLine("Capture", &s_timingDataBuffer.data()[0].captureTimeMs, s_timingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::PlotLine("Submit",  &s_timingDataBuffer.data()[0].submitTimeMs,  s_timingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::EndPlot();
          }
          if (ImPlot::BeginPlot("###CaptureLatencyInterval", ImVec2(-1,150), /*flags=*/ plotFlags)) {
              ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
              ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);
              ImPlot::SetupAxisLimits(ImAxis_X1, 0, s_timingDataBuffer.size(), ImPlotCond_Always);
              ImPlot::SetupFinish();

              ImPlot::PlotBars("Adjustments", &s_timingDataBuffer.data()[0].captureIntervalAdjustmentMarker, s_timingDataBuffer.size(), /*width=*/ 0.67, /*shift=*/ 0, /*flags=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::PlotLine("Capture Latency", &s_timingDataBuffer.data()[0].captureLatencyMs, s_timingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::PlotLine("Capture Interval", &s_timingDataBuffer.data()[0].captureIntervalMs, s_timingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::EndPlot();
          }

          if ((argusCamera->streamCount() > 1) && ImPlot::BeginPlot("###InterSensorTiming", ImVec2(-1,150), /*flags=*/ plotFlags)) {
              ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
              ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit);
              ImPlot::SetupAxisLimits(ImAxis_X1, 0, s_timingDataBuffer.size(), ImPlotCond_Always);
              ImPlot::SetupFinish();

              for (size_t sensorIdx = 1; sensorIdx < argusCamera->streamCount(); ++sensorIdx) {
                char idbuf[32];
                sprintf(idbuf, "Sensor %zu", sensorIdx);
                ImPlot::PlotLine(idbuf, &s_sensorTimingData.data()[0].timestampDelta[sensorIdx-1], s_sensorTimingData.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, s_sensorTimingData.offset(), sizeof(SensorTimingData));
              }
              ImPlot::EndPlot();
          }

          settingsDirty |= argusCamera->renderPerformanceTuningIMGUI();

          if (ImGui::Button("Restart Capture")) {
            argusCamera->stop(); // will automatically restart on next frame when we call setRepeatCapture again
            restartSkipFrameCounter = 3; // skip a few frames before restarting to smooth out the timing glitch we just caused
          }

          if (depthMapGenerator) {
            depthMapGenerator->renderIMGUIPerformanceGraphs();
          }
        }

        ImGui::Text("Lat=%.1fms (%.1fms-%.1fms) %.1fFPS", currentCaptureLatencyMs, boost::accumulators::min(captureLatency), boost::accumulators::max(captureLatency), io.Framerate);

        if (ImGui::CollapsingHeader("Remote Debug")) {
          ImGui::Text("Debug URL: %s", renderDebugURL().c_str());
          ImGui::Checkbox("Distortion correction", &debugUseDistortion);
          if (ImGui::RadioButton("No overlay", !debugOverlay)) {
            debugOverlay.reset();
          }
#ifdef HAVE_OPENCV_CUDA
          if (ImGui::RadioButton("Focus Assist", debugOverlay && debugOverlay->overlayType() == kDebugOverlayFocusAssist)) {
            if ((!debugOverlay) || debugOverlay->overlayType() != kDebugOverlayFocusAssist)
              debugOverlay.reset(new FocusAssistDebugOverlay(argusCamera));
          }
#endif // HAVE_OPENCV_CUDA

          if (debugOverlay)
            debugOverlay->renderIMGUI();
        }

        ImGui::End();

        if (settingsDirty) {
          settingsDirtyFrame = frameCounter;
        } else if ((settingsDirtyFrame != 0) && (frameCounter >= (settingsDirtyFrame + settingsAutosaveIntervalFrames))) {
          saveSettings();
        }

      } else {
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, 0), 0, /*pivot=*/ImVec2(0.5f, 0.0f)); // top-center aligned
        ImGui::Begin("StatusBar", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        char timebuf[64];
        time_t t = time(NULL);
        strftime(timebuf, 64, "%a %b %e %T", localtime(&t));
        ImGui::TextUnformatted(timebuf);
        ImGui::SameLine(); ImGui::Separator(); ImGui::SameLine();
        ImGui::Text("Lat=%.1fms (%.1fms-%.1fms) %.1fFPS", currentCaptureLatencyMs, boost::accumulators::min(captureLatency), boost::accumulators::max(captureLatency), io.Framerate);

        ImGui::End();
      }

      rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
      rhi()->beginRenderPass(guiRT, kLoadClear);
      ImGui::Render();
      ImGui_ImplFxRHI_RenderDrawData(guiRT, ImGui::GetDrawData());
      rhi()->endRenderPass(guiRT);

      if ((frameCounter & 0x7fUL) == 0) {
#ifdef LATENCY_DEBUG
        printf("Capture latency: min=%.3g max=%.3g mean=%.3g median=%.3g\n",
          boost::accumulators::min(captureLatency),
          boost::accumulators::max(captureLatency),
          boost::accumulators::mean(captureLatency),
          boost::accumulators::median(captureLatency));

        printf("Capture interval: min=%.3g max=%.3g mean=%.3g median=%.3g\n",
          boost::accumulators::min(captureInterval),
          boost::accumulators::max(captureInterval),
          boost::accumulators::mean(captureInterval),
          boost::accumulators::median(captureInterval));

        printf("Frame interval: % .6f ms (% .6f fps) min=%.3g max=%.3g median=%.3g\n",
          static_cast<double>(boost::accumulators::mean(frameInterval)) / 1000000.0,
          1000000000.0 / static_cast<double>(boost::accumulators::mean(frameInterval)),

          static_cast<double>(boost::accumulators::min(frameInterval)) / 1000000.0,
          static_cast<double>(boost::accumulators::max(frameInterval)) / 1000000.0,
          static_cast<double>(boost::accumulators::median(frameInterval)) / 1000000.0);
#endif

        captureLatency = {};
        captureInterval = {};
        frameInterval = {};
      }

      rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

      // Note that our camera uses reversed depth projection -- we clear to 0 and use a "greater" depth-test.
      rhi()->setClearDepth(0.0f);
      rhi()->beginRenderPass(eyeRT, kLoadClear);
      rhi()->bindDepthStencilState(standardGreaterDepthStencilState);
      rhi()->setViewports(eyeViewports, 2);

      FxRenderView renderViews[2];
      // TODO actual camera setup here. renderDisparityDepthMap only uses the viewProjection matrix.

      for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
        renderViews[eyeIdx].viewMatrix = eyeView[eyeIdx];
        renderViews[eyeIdx].projectionMatrix = eyeProjection[eyeIdx];
        renderViews[eyeIdx].viewProjectionMatrix = renderViews[eyeIdx].projectionMatrix * renderViews[eyeIdx].viewMatrix;
      }

      for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
        CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);

        if (debugEnableDepthMapGenerator && depthMapGenerator && v.isStereo && !calibrationContext && !(rdmaContext && rdmaContext->hasPeerConnections())) {
          // Single-pass stereo
          rhi()->setViewports(eyeViewports, 2);
          depthMapGenerator->renderDisparityDepthMapStereo(viewIdx, renderViews[0], renderViews[1], cameraSystem->viewWorldTransform(viewIdx));

        } else {
          // TODO logic needs work for single-pass stereo
          for (int eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
            for (int viewEyeIdx = 0; viewEyeIdx < (v.isStereo ? 2 : 1); ++viewEyeIdx) {
              // coordsys right now: -X = left, -Z = into screen
              // (camera is at the origin)
              float stereoOffsetSign = v.isStereo ? ((viewEyeIdx == 0 ? -1.0f : 1.0f)) : 0.0f;
              float viewDepth = 10.0f;
              const glm::vec3 tx = glm::vec3(stereoOffsetSign * stereoOffset, 0.0f, -viewDepth);
              float aspectRatioYScale = static_cast<float>(argusCamera->streamHeight()) / static_cast<float>(argusCamera->streamWidth());
              glm::mat4 mvp;
              float clipMinU = 0.0f;
              float clipMaxU = 1.0f;

              if (v.isStereo && v.isPanorama) {
                //CameraSystem::Camera& cLeft = cameraSystem->cameraAtIndex(v.cameraIndices[0]);
                //CameraSystem::Camera& cRight = cameraSystem->cameraAtIndex(v.cameraIndices[1]);
                static bool debug = true;

                // Compute FOV for panorama view. This is the combined X FoVs of the cameras offset by the additional Y-rotation between views.
                float rx, ry, rz; // radians
                glm::extractEulerAngleYXZ(glm::mat4(glmMat3FromCVMatrix(v.stereoRotation)), rx, ry, rz);

                if (debug) printf("Pano angles %f %f %f\n", glm::degrees(rx), glm::degrees(ry), glm::degrees(rz));

                float stereoFovX = v.fovX;
                float panoFovX = stereoFovX + fabs(glm::degrees(rx)); // degrees

                float clipFrac = (panoFovX - stereoFovX) / stereoFovX;
                clipFrac = 1.0f - ((1.0f - clipFrac) * panoClipScale); // apply scale
                if (viewEyeIdx == 0) // left camera, clip from max side
                  clipMaxU = std::min<float>(clipFrac + panoClipOffset, 1.0f);
                else // right camera, clip from min side
                  clipMinU = std::max<float>(0.0f, (1.0f - clipFrac) + panoClipOffset);

                // half-width is viewDepth * tanf(0.5 * fovx). maps directly to scale factor since the quad we're rendering is two units across (NDC quad)
                float panoFovScaleFactor = viewDepth * tan(glm::radians(panoFovX * 0.5f));
                float fovScaleFactor = 0.5f * panoFovScaleFactor; // half the scale for each view, since it's half of the panorama (approx).
                if (debug) printf("stereoFovX=%f panoFovX=%f clipFrac=%f min=%f max=%f panoFovScaleFactor=%f\n", stereoFovX, panoFovX, clipFrac, clipMinU, clipMaxU, panoFovScaleFactor);

                float rScale = (viewEyeIdx == 0) ? 0.5f : -0.5f;
                glm::mat4 rot = glm::mat4(glm::eulerAngleYXZ(rx * rScale, ry * rScale, rz * rScale));

                float tScale = ((viewEyeIdx == 0) ? 0.5f : -0.5f) * panoTxScale;

                // Position the pano quads in view space by rotating by half of the stereo rotation offset in either direction.
                glm::mat4 model = cameraSystem->viewWorldTransform(viewIdx) * glm::translate(glmVec3FromCV(v.stereoTranslation) * tScale) * glm::translate(tx) * rot * glm::scale(glm::vec3(fovScaleFactor * zoomFactor, fovScaleFactor * zoomFactor * aspectRatioYScale, 1.0f));

                // Intentionally ignoring the eyeView matrix here. Camera to eye stereo offset is controlled directly by the stereoOffset variable
                mvp = renderViews[viewEyeIdx].viewProjectionMatrix * model;

                debug = false; // XXX once

              } else {
                if (v.isStereo && (viewEyeIdx != eyeIdx))
                  continue;

                float fovX = 75.0f; // default guess if no calibration

                // Compute FOV-based prescaling -- figure out the billboard size in world units based on the render depth and FOV
                if (v.isStereo ? v.haveStereoRectificationParameters() : cameraSystem->cameraAtIndex(v.cameraIndices[viewEyeIdx]).haveIntrinsicCalibration()) {
                  fovX = v.isStereo ? v.fovX : cameraSystem->cameraAtIndex(v.cameraIndices[viewEyeIdx]).fovX;
                }

                // half-width is viewDepth * tanf(0.5 * fovx). maps directly to scale factor since the quad we're rendering is two units across (NDC quad)
                float fovScaleFactor = viewDepth * tan(glm::radians(fovX * 0.5f));

                glm::mat4 model = cameraSystem->viewWorldTransform(viewIdx) * glm::translate(tx) * glm::scale(glm::vec3(fovScaleFactor * zoomFactor, fovScaleFactor * zoomFactor * aspectRatioYScale, 1.0f));

                // Intentionally ignoring the eyeView matrix here. Camera to eye stereo offset is controlled directly by the stereoOffset variable
                mvp = renderViews[viewEyeIdx].viewProjectionMatrix * model;
              }

              RHISurface::ptr overlayTex, distortionTex;
              size_t drawFlags = 0;
              if (calibrationContext && calibrationContext->involvesCamera(v.cameraIndices[viewEyeIdx])) {
                // Calibrating a stereo view that includes this camera
                overlayTex = calibrationContext->overlaySurfaceAtIndex(viewEyeIdx);
                distortionTex = calibrationContext->previewDistortionMapForCamera(v.cameraIndices[viewEyeIdx]);
              } else if (v.isStereo) {
                // Drawing this camera as part of a stereo pair
                distortionTex = v.stereoDistortionMap[viewEyeIdx];
                drawFlags = DRAW_FLAGS_USE_MASK;

              } else {
                // Drawing this camera as a mono view
                distortionTex = cameraSystem->cameraAtIndex(v.cameraIndices[viewEyeIdx]).intrinsicDistortionMap;
                drawFlags = DRAW_FLAGS_USE_MASK;

              }

              rhi()->setViewport(eyeViewports[eyeIdx]);
              renderDrawCamera(v.cameraIndices[viewEyeIdx], drawFlags, distortionTex, overlayTex, mvp, clipMinU, clipMaxU);
            } // view-eye loop
          }
        }

      } // view loop

/*
      if (renderSBS && sbsSeparatorWidth) {
        rhi()->bindRenderPipeline(solidQuadPipeline);
        SolidQuadUniformBlock ub;
        // Scale to reduce the X-width of the -1...1 quad to the size requested in sbsSeparatorWidth
        ub.modelViewProjection = renderView.viewProjectionMatrix * glm::translate(glm::vec3(0.0f, 0.0f, -1.0f)) * glm::scale(glm::vec3(static_cast<float>(sbsSeparatorWidth * 4) / static_cast<float>(eyeRT[eyeIdx]->width()), 1.0f, 1.0f));
        ub.color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        rhi()->loadUniformBlockImmediate(ksSolidQuadUniformBlock, &ub, sizeof(SolidQuadUniformBlock));
        rhi()->drawNDCQuad();
      }
*/

      // UI overlay
      {
        rhi()->bindBlendState(standardAlphaOverBlendState);
        rhi()->bindDepthStencilState(disabledDepthStencilState);
        rhi()->bindRenderPipeline(uiLayerStereoPipeline);
        rhi()->loadTexture(ksImageTex, guiTex, linearClampSampler);
        rhi()->setViewports(eyeViewports, 2);

        UILayerStereoUniformBlock ub;
        glm::mat4 modelMatrix = glm::translate(glm::vec3(0.0f, 0.0f, -uiDepth)) * glm::scale(glm::vec3(uiScale * (io.DisplaySize.x / io.DisplaySize.y), uiScale, uiScale));
        ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
        ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;

        rhi()->loadUniformBlockImmediate(ksUILayerStereoUniformBlock, &ub, sizeof(ub));
        rhi()->drawNDCQuad();
      }

      rhi()->endRenderPass(eyeRT);

      // Debug feedback rendering
      {
        RHISurface::ptr debugSurface = renderAcquireDebugSurface();
        if (debugSurface) {
          if (debugOverlay)
            debugOverlay->update();

          RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({debugSurface}));
          rhi()->beginRenderPass(rt, kLoadInvalidate);

          // render each distortion-corrected camera view to the previously allocated region of the debug surface
          for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
            rhi()->setViewport(debugSurfaceCameraRects[cameraIdx]);

            RHISurface::ptr overlayTex, distortionTex;

            if (calibrationContext && calibrationContext->involvesCamera(cameraIdx)) {
              overlayTex = calibrationContext->overlaySurfaceAtIndex(calibrationContext->overlaySurfaceIndexForCamera(cameraIdx));
              distortionTex = calibrationContext->previewDistortionMapForCamera(cameraIdx);
            } else if (debugOverlay) {
              overlayTex = debugOverlay->overlaySurfaceForCamera(cameraIdx);
            } else if (debugUseDistortion) {
              // Distortion-corrected view
              distortionTex = cameraSystem->cameraAtIndex(cameraIdx).intrinsicDistortionMap;
            } else {
              // No-distortion / direct passthrough
            }

            renderDrawCamera(cameraIdx, /*drawFlags=*/0, distortionTex, overlayTex, /*mvp=*/glm::mat4(1.0f) /*identity*/);
          }

          if (drawUI || calibrationContext) {
            // Default to drawing the UI on the center-bottom of the debug surface
            RHIRect uiDestRect = RHIRect::xywh((debugSurface->width() / 2) - (guiTex->width() / 2), debugSurface->height() - guiTex->height(), guiTex->width(), guiTex->height());

            // If a calibration context is active, try to find a camera region to draw the UI over that's not currently being calibrated.
            if (calibrationContext) {
              for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
                if (!calibrationContext->involvesCamera(cameraIdx)) {
                  const RHIRect& cameraRect = debugSurfaceCameraRects[cameraIdx];
                  uiDestRect.x = cameraRect.x + ((cameraRect.width - uiDestRect.width) / 2); // center
                  uiDestRect.y = cameraRect.y;
                  break;
                }
              }
            }

            // If the UI texture is taller than the debug surface, scale it down proportionally.
            // (We will hit this case with a common configuration of 1 or 2 1280x720 cameras)
            if (guiTex->height() >= debugSurface->height()) {
              uiDestRect.y = 0;
              uiDestRect.height = debugSurface->height();
              // apply aspect ratio correction to the X position/size
              float scale = static_cast<float>(debugSurface->height()) / static_cast<float>(guiTex->height());
              float xInsetPx = (scale * 0.5f) * static_cast<float>(guiTex->width());
              uiDestRect.x += xInsetPx;
              uiDestRect.width -= xInsetPx;
            }

            rhi()->setViewport(uiDestRect);

            rhi()->bindBlendState(standardAlphaOverBlendState);
            rhi()->bindRenderPipeline(uiLayerPipeline);
            rhi()->loadTexture(ksImageTex, guiTex, linearClampSampler);
            UILayerUniformBlock uiLayerBlock;
            uiLayerBlock.modelViewProjection = glm::mat4(1.0f);

            rhi()->loadUniformBlockImmediate(ksUILayerUniformBlock, &uiLayerBlock, sizeof(UILayerUniformBlock));
            rhi()->drawNDCQuad();
          }

          rhi()->endRenderPass(rt);
          renderSubmitDebugSurface(debugSurface);
        }
      }

      if (rdmaContext && rdmaContext->hasPeerConnections() && rdmaFrame == 0) {
        // Issue RDMA surface copies and write-buffer flushes
        for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
          CUeglFrame eglFrame;
          CUDA_CHECK(cuGraphicsResourceGetMappedEglFrame(&eglFrame, argusCamera->cudaGraphicsResource(cameraIdx), 0, 0));

          { // Luma copy
            CUDA_MEMCPY2D copyDescriptor;
            memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

            copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyDescriptor.srcDevice = (CUdeviceptr) eglFrame.frame.pPitch[0];
            copyDescriptor.srcPitch = lumaResourceDescriptor.res.pitch2D.pitchInBytes;

            copyDescriptor.WidthInBytes = lumaResourceDescriptor.res.pitch2D.width * 1; //8-bit, 1-channel
            copyDescriptor.Height = lumaResourceDescriptor.res.pitch2D.height;

            copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
            copyDescriptor.dstHost = rdmaCameraLumaBuffers[cameraIdx]->data();
            copyDescriptor.dstPitch = lumaResourceDescriptor.res.pitch2D.pitchInBytes;

            CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
          }

          { // Chroma copy
            CUDA_MEMCPY2D copyDescriptor;
            memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

            copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyDescriptor.srcDevice = (CUdeviceptr) eglFrame.frame.pPitch[1];
            copyDescriptor.srcPitch = chromaResourceDescriptor.res.pitch2D.pitchInBytes;

            copyDescriptor.WidthInBytes = chromaResourceDescriptor.res.pitch2D.width * 2; //8-bit, 2-channel
            copyDescriptor.Height = chromaResourceDescriptor.res.pitch2D.height;

            copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
            copyDescriptor.dstHost = rdmaCameraChromaBuffers[cameraIdx]->data();
            copyDescriptor.dstPitch = chromaResourceDescriptor.res.pitch2D.pitchInBytes;

            CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
          }

          rdmaContext->asyncFlushWriteBuffer(rdmaCameraLumaBuffers[cameraIdx]);
          rdmaContext->asyncFlushWriteBuffer(rdmaCameraChromaBuffers[cameraIdx]);
        }
        // Send "buffers dirty" event
        rdmaContext->asyncSendUserEvent(1, SerializationBuffer());
      }
      // track RDMA send interval
      rdmaFrame += 1;
      if (rdmaFrame >= rdmaInterval) {
        rdmaFrame = 0;
      }

      timingData.submitTimeMs = deltaTimeMs(frameStartTimeNs, currentTimeNs());
      renderHMDFrame();
      {
        uint64_t thisFrameTimestamp = currentTimeNs();
        if (previousFrameTimestamp) {
          uint64_t interval = thisFrameTimestamp - previousFrameTimestamp;
          //if ((frameCounter & 0xff)  == 0xff) {
          //  printf("raw interval %lu\n", interval);
          //}
          frameInterval(interval);

          // Update the target capture interval periodically
#if 0
          if ((frameCounter & 0x1f) == 0x1f) {
            argusCamera->setTargetCaptureIntervalNs(boost::accumulators::rolling_mean(frameInterval));
          }
#endif

          io.DeltaTime = static_cast<double>(interval / 1000000000.0);
        }

        currentCaptureLatencyMs = static_cast<double>(thisFrameTimestamp - argusCamera->frameSensorTimestamp(0)) / 1000000.0;
        captureLatency(currentCaptureLatencyMs);

        previousFrameTimestamp = thisFrameTimestamp;
      }

      timingData.captureLatencyMs = currentCaptureLatencyMs;
      timingData.captureIntervalMs = currentCaptureIntervalMs;
      timingData.captureIntervalAdjustmentMarker = argusCamera->didAdjustCaptureIntervalThisFrame() ? 10.0f : 0.0;

      s_timingDataBuffer.push_back(timingData);
    } // Camera rendering loop
  }


  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  // clear screen
  rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
  rhi()->endRenderPass(windowRenderTarget);
  rhi()->swapBuffers(windowRenderTarget);

  argusCamera->stop();
  delete argusCamera;

  RenderShutdown();

  FxThreading::detail::shutdown();

  return 0;
}

