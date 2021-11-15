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

#include "ArgusCamera.h"
#include "common/CameraSystem.h"
#include "common/DepthMapGenerator.h"
#include "common/DepthWorkerControl.h"
#include "common/FxThreading.h"
#include "common/ScrollingBuffer.h"
#include "InputListener.h"
#include "Render.h"

#include "imgui_backend.h"
#include "implot/implot.h"

#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "rdma/RDMAContext.h"

//#define LATENCY_DEBUG

// Camera config
// Size parameters for sensor mode selection.
// Note that changing the sensor mode will invalidate the calibration
// (Pixel coordinates are baked into the calibration data)
size_t s_cameraWidth, s_cameraHeight;

// Requested capture rate for the camera. This should be the framerate of the display device, with as much precision as possible.
// TODO: autodetect this. (current value pulled from running `fbset`)
const double s_cameraFramerate = 89.527;

// Camera render parameters
float zoomFactor = 1.0f;
float stereoOffset = 0.0f;
bool renderSBS = false;
bool useMask = true;
int sbsSeparatorWidth = 4;
bool debugUseDistortion = true;

// Camera info/state
ArgusCamera* argusCamera;
CameraSystem* cameraSystem;


RDMAContext* rdmaContext;


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


static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

static inline float deltaTimeMs(uint64_t startTimeNs, uint64_t endTimeNs) {
  return static_cast<float>(endTimeNs - startTimeNs) / 1000000.0f;
}

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


  DepthWorkerBackend depthBackend = kDepthWorkerDGPU;
  bool enableRDMA = true;
  bool debugInitOnly = false;
  int rdmaInterval = 2;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth-backend")) {
      if (i == (argc - 1)) {
        printf("--depth-backend: requires argument\n");
        return 1;
      }
      depthBackend = depthBackendStringToEnum(argv[++i]);
    } else if (!strcmp(argv[i], "--disable-rdma")) {
      enableRDMA = false;
    } else if (!strcmp(argv[i], "--debug-init-only")) {
      debugInitOnly = true;
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

  if (depthBackend == kDepthWorkerDepthAI) {
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


  SHMSegment<DepthMapSHM>* shm = NULL;
  DepthMapGenerator* depthMapGenerator = NULL;
  if (depthBackend != kDepthWorkerNone) {
    shm = SHMSegment<DepthMapSHM>::createSegment("depth-worker", 16*1024*1024);
    printf("Waiting for depth worker...\n");
    if (!spawnAndWaitForDepthWorker(depthBackend, &shm->segment()->m_workerReadySem)) {
      return 1;
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
  if (!RenderInit()) {
    printf("RenderInit() failed\n");
    return 1;
  }

  FxThreading::detail::init();

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = NULL; // Disable INI file load/save
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Keyboard navigation (mapped from the InputListener media remote interface)
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Gamepad navigation (not used right now)
  ImGui_ImplOpenGL3_Init(NULL);
  ImGui_ImplInputListener_Init();

  io.DisplaySize = ImVec2(512.0f, 512.0f); // Not the full size, but the size of our overlay RT
  io.DisplayFramebufferScale = ImVec2(2.0f, 2.0f);

  // Open the cameras
  argusCamera = new ArgusCamera(renderEGLDisplay(), renderEGLContext(), s_cameraFramerate);
  s_cameraWidth = argusCamera->streamWidth();
  s_cameraHeight = argusCamera->streamHeight();

  std::vector<RHIRect> debugSurfaceCameraRects;
  {
    // Size and allocate debug surface area based on camera count
    unsigned int debugColumns = 1, debugRows = 1;
    if (argusCamera->streamCount() > 1) {
      debugColumns = 2;
      debugRows = (argusCamera->streamCount() + 1) / 2; // round up
    }
    unsigned int dsW = debugColumns * argusCamera->streamWidth();
    unsigned int dsH = debugRows * argusCamera->streamHeight();
    printf("Debug stream: selected a %ux%u layout on a %ux%u surface for %zu cameras\n", debugColumns, debugRows, dsW, dsH, argusCamera->streamCount());

    for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
      unsigned int col = cameraIdx % debugColumns;
      unsigned int row = cameraIdx / debugColumns;
      RHIRect r = RHIRect::xywh(col * argusCamera->streamWidth(), row * argusCamera->streamHeight(), argusCamera->streamWidth(), argusCamera->streamHeight());
      printf("  [%zu] (%ux%u) +(%u, %u)\n", cameraIdx, r.width, r.height, r.x, r.y);
      debugSurfaceCameraRects.push_back(r);
    }
    renderSetDebugSurfaceSize(dsW, dsH);
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


  // TODO move this depth map generator init to CameraSystem
  if (currentDepthWorkerBackend() != kDepthWorkerNone) {
    depthMapGenerator = new DepthMapGenerator(cameraSystem, shm);
  }


  // Create the RDMA configuration buffer
  RDMABuffer::ptr configBuf;
  if (rdmaContext) {
    configBuf = rdmaContext->newManagedBuffer("config", 8192, kRDMABufferUsageWriteSource); // TODO this should be read-source once we implement read

    SerializationBuffer cfg;
    cfg.put_u32(argusCamera->streamCount());
    cfg.put_u32(argusCamera->streamWidth());
    cfg.put_u32(argusCamera->streamHeight());

    memcpy(configBuf->data(), cfg.data(), cfg.size());
    rdmaContext->asyncFlushWriteBuffer(configBuf);
  }


  // Create RDMA camera buffers and render surfaces
  std::vector<RHISurface::ptr> rdmaRenderSurfaces;
  std::vector<RHIRenderTarget::ptr> rdmaRenderTargets;
  std::vector<RDMABuffer::ptr> rdmaCameraBuffers;
  if (rdmaContext) {
    rdmaRenderSurfaces.resize(argusCamera->streamCount());
    rdmaRenderTargets.resize(argusCamera->streamCount());
    rdmaCameraBuffers.resize(argusCamera->streamCount());

    for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
      rdmaRenderSurfaces[cameraIdx] = rhi()->newTexture2D(argusCamera->streamWidth(), argusCamera->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
      rdmaRenderTargets[cameraIdx] = rhi()->compileRenderTarget(RHIRenderTargetDescriptor( {rdmaRenderSurfaces[cameraIdx] } ));

      char bufferName[32];
      sprintf(bufferName, "camera%zu", cameraIdx);
      rdmaCameraBuffers[cameraIdx] = rdmaContext->newManagedBuffer(bufferName, argusCamera->streamWidth() * argusCamera->streamHeight() * 4, kRDMABufferUsageWriteSource);
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
    argusCamera->setRepeatCapture(true);

    // Camera rendering mode
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
    float uiScale = 1.0f;
    boost::scoped_ptr<CameraSystem::CalibrationContext> calibrationContext;

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

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplInputListener_NewFrame();
      ImGui::NewFrame();

      ++frameCounter;

      if (debugInitOnly && frameCounter >= 10) {
        printf("DEBUG: Initialization-only testing requested, exiting now.\n");
        want_quit = true;
      }

      // Only use repeating captures if we're not in calibration. The variable CPU-side delays for calibration image processing usually end up crashing libargus.
      argusCamera->setRepeatCapture(!((bool) calibrationContext));

      argusCamera->readFrame();
      timingData.captureTimeMs = deltaTimeMs(frameStartTimeNs, currentTimeNs());

      if (previousCaptureTimestamp) {
        currentCaptureIntervalMs = static_cast<double>(argusCamera->frameSensorTimestamp(0) - previousCaptureTimestamp) / 1000000.0;
        captureInterval(currentCaptureIntervalMs);
      }
      previousCaptureTimestamp = argusCamera->frameSensorTimestamp(0);


      if (rdmaContext && rdmaContext->hasPeerConnections()) {
        // Issue renders/copies to populate RDMA surfaces
        for (size_t cameraIdx = 0; cameraIdx < rdmaRenderTargets.size(); ++cameraIdx) {
          rhi()->beginRenderPass(rdmaRenderTargets[cameraIdx], kLoadInvalidate);
          // flip Y on MVP to fix coordinate system orientation
          glm::mat4 mvp = glm::mat4(1.0f);
          mvp[1][1] = -1.0f;
          renderDrawCamera(cameraIdx, /*flags=*/0, /*distortion=*/RHISurface::ptr(), /*overlay=*/RHISurface::ptr(), mvp);
          rhi()->endRenderPass(rdmaRenderTargets[cameraIdx]);
        }

      }

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
        ImGui::Begin("Overlay", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        if (calibrationContext) {

          calibrationContext->processUI();

        } else {
          //ImGui::Text("Config");
          ImGui::Checkbox("SBS", &renderSBS);
          ImGui::Checkbox("Mask", &useMask);
          ImGui::SliderFloat("Zoom", &zoomFactor, 0.5f, 2.0f);
          ImGui::SliderFloat("Stereo Offset", &stereoOffset, -0.5f, 0.5f);
          if (renderSBS) {
            ImGui::SliderInt("Separator Width", (int*) &sbsSeparatorWidth, 0, 32);
          }

          for (size_t cameraIdx = 0; cameraIdx < cameraSystem->cameras(); ++cameraIdx) {
            char caption[64];
            sprintf(caption, "Calibrate camera %zu", cameraIdx);
            if (ImGui::Button(caption)) {
              calibrationContext.reset(cameraSystem->calibrationContextForCamera(cameraIdx));
            }
          }
          for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
            CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
            if (v.isStereo) {
              char caption[64];
              sprintf(caption, "Calibrate stereo view %zu", viewIdx);
              if (ImGui::Button(caption)) {
                calibrationContext.reset(cameraSystem->calibrationContextForView(viewIdx));
              }
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
              ImGui::PushID(viewIdx);
              ImGui::Text("View %zu Transform", viewIdx);
              // convert to and from millimeters for editing
              glm::vec3 txMM = v.viewTranslation * 1000.0f;
              if (ImGui::DragFloat3("Tx", &txMM[0], /*speed=*/ 0.1f, /*min=*/ -500.0f, /*max=*/ 500.0f, "%.1fmm")) {
                v.viewTranslation = txMM / 1000.0f;
              }
              ImGui::DragFloat3("Rx", &v.viewRotation[0], /*speed=*/0.1f, /*min=*/ -75.0f, /*max=*/ 75.0f, "%.1fdeg");
              ImGui::PopID();
            }
          }
          if (ImGui::Button("Save Settings")) {
            cameraSystem->saveCalibrationData();
          }
          if (debugEnableDepthMapGenerator && depthMapGenerator) {
            depthMapGenerator->renderIMGUI();
          }
        }


        ImGui::SliderFloat("UI Scale", &uiScale, 0.5f, 1.5f);

        {
          const auto& meta = argusCamera->frameMetadata(0);
          ImGui::Text("Exp=1/%usec %uISO DGain=%f AGain=%f",
            (unsigned int) (1000000.0f / static_cast<float>(meta.sensorExposureTimeNs/1000)), meta.sensorSensitivityISO, meta.ispDigitalGain, meta.sensorAnalogGain);
        }

        // Skip perf data to save UI space if we're calibrating
        if (!calibrationContext) {
          ImPlot::SetNextPlotLimitsY(0, 12.0f);
          if (ImPlot::BeginPlot("##FrameTiming", NULL, NULL, ImVec2(-1,150), 0, /*xFlags=*/ ImPlotAxisFlags_NoTickLabels, /*yFlags=*/ /*ImPlotAxisFlags_NoTickLabels*/ ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin)) {
              ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL,0.5f);
              ImPlot::PlotLine("Capture", &s_timingDataBuffer.data()[0].captureTimeMs, s_timingDataBuffer.size(), /*-INFINITY,*/ 1, 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::PlotLine("Submit",  &s_timingDataBuffer.data()[0].submitTimeMs, s_timingDataBuffer.size(), /*-INFINITY,*/ 1, 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::EndPlot();
          }
          if (ImPlot::BeginPlot("###CaptureLatencyInterval", NULL, NULL, ImVec2(-1,150), 0, /*xFlags=*/ ImPlotAxisFlags_NoTickLabels, /*yFlags=*/ /*ImPlotAxisFlags_NoTickLabels*/ ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin)) {
              ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL,0.5f);
              ImPlot::PlotBars("Adjustments", &s_timingDataBuffer.data()[0].captureIntervalAdjustmentMarker, s_timingDataBuffer.size(), /*width=*/ 0.67, /*shift=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::PlotLine("Capture Latency", &s_timingDataBuffer.data()[0].captureLatencyMs, s_timingDataBuffer.size(), /*-INFINITY,*/ 1, 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::PlotLine("Capture Interval", &s_timingDataBuffer.data()[0].captureIntervalMs, s_timingDataBuffer.size(), /*-INFINITY,*/ 1, 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::EndPlot();
          }

          {
            bool v = argusCamera->willAdjustCaptureInterval();
            if (ImGui::Checkbox("Auto-adjust capture interval", &v))
              argusCamera->setAdjustCaptureInterval(v);
          }
        }

        ImGui::Text("Lat=%.1fms (%.1fms-%.1fms) %.1fFPS", currentCaptureLatencyMs, boost::accumulators::min(captureLatency), boost::accumulators::max(captureLatency), io.Framerate);
        ImGui::Text("Debug URL: %s", renderDebugURL().c_str());
        ImGui::Checkbox("Debug output: Distortion correction", &debugUseDistortion);
        ImGui::End();

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
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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

      for (int eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        // Note that our camera uses reversed depth projection -- we clear to 0 and use a "greater" depth-test.
        rhi()->setClearDepth(0.0f);
        rhi()->beginRenderPass(eyeRT[eyeIdx], kLoadClear);
        rhi()->bindDepthStencilState(standardGreaterDepthStencilState);

        FxRenderView renderView;
        // TODO actual camera setup here. renderDisparityDepthMap only uses the viewProjection matrix.

        renderView.viewMatrix = eyeView[eyeIdx];
        renderView.projectionMatrix = eyeProjection[eyeIdx];
        renderView.viewProjectionMatrix = renderView.projectionMatrix * renderView.viewMatrix;

        for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
          CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);

          if (debugEnableDepthMapGenerator && depthMapGenerator && v.isStereo && !calibrationContext && !(rdmaContext && rdmaContext->hasPeerConnections())) {

            depthMapGenerator->renderDisparityDepthMap(viewIdx, renderView, cameraSystem->viewWorldTransform(viewIdx));

          } else {
            for (int viewEyeIdx = 0; viewEyeIdx < (v.isStereo ? 2 : 1); ++viewEyeIdx) {
              if (renderSBS == false && v.isStereo && (viewEyeIdx != eyeIdx))
                continue;

              // coordsys right now: -X = left, -Z = into screen
              // (camera is at the origin)
              float stereoOffsetSign = v.isStereo ? ((viewEyeIdx == 0 ? -1.0f : 1.0f)) : 0.0f;
              float viewDepth = 10.0f;

              const glm::vec3 tx = glm::vec3(stereoOffsetSign * stereoOffset, 0.0f, -viewDepth);
              double fovX = 75.0f; // default guess if no calibration

              // Compute FOV-based prescaling -- figure out the billboard size in world units based on the render depth and FOV
              if (v.isStereo ? v.haveStereoRectificationParameters() : cameraSystem->cameraAtIndex(v.cameraIndices[viewEyeIdx]).haveIntrinsicCalibration()) {
                fovX = v.isStereo ? v.fovX : cameraSystem->cameraAtIndex(v.cameraIndices[viewEyeIdx]).fovX;
              }

              // half-width is viewDepth * tanf(0.5 * fovx). maps directly to scale factor since the quad we're rendering is two units across (NDC quad)
              float fovScaleFactor = viewDepth * tan((fovX * 0.5) * (M_PI/180.0f));

              glm::mat4 model = cameraSystem->viewWorldTransform(viewIdx) * glm::translate(tx) * glm::scale(glm::vec3(fovScaleFactor * zoomFactor, fovScaleFactor * zoomFactor * (static_cast<float>(argusCamera->streamHeight()) / static_cast<float>(argusCamera->streamWidth())) , 1.0f));

              // Intentionally ignoring the eyeView matrix here. Camera to eye stereo offset is controlled directly by the stereoOffset variable
              glm::mat4 mvp = renderView.viewProjectionMatrix * model;

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
              

              renderDrawCamera(v.cameraIndices[viewEyeIdx], drawFlags, distortionTex, overlayTex, mvp);

    /*
              float uClipFrac = 0.75f;
      #if 1
              // Compute the clipping parameters to cut the views off at the centerline of the view (x=0 in model space)
              {
                glm::vec3 worldP0 = glm::vec3(model * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f));
                glm::vec3 worldP1 = glm::vec3(model * glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f));

                float xLen = fabs(worldP0.x - worldP1.x);
                // the coordinate that's closest to X0 will be the one we want to clip
                float xOver = std::min<float>(fabs(worldP0.x), fabs(worldP1.x));

                uClipFrac = (xLen - xOver)/xLen;
              }
      #endif

              if (cameraIdx == 0) { // left
                ub.minUV = glm::vec2(0.0f,  0.0f);
                ub.maxUV = glm::vec2(uClipFrac, 1.0f);
              } else { // right
                ub.minUV = glm::vec2(1.0f - uClipFrac, 0.0f);
                ub.maxUV = glm::vec2(1.0f,  1.0f);
              }
    */

            } // view-eye loop
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
        rhi()->bindBlendState(standardAlphaOverBlendState);
        rhi()->bindDepthStencilState(disabledDepthStencilState);
        rhi()->bindRenderPipeline(uiLayerPipeline);
        rhi()->loadTexture(ksImageTex, guiTex, linearClampSampler);
        UILayerUniformBlock uiLayerBlock;
        float uiScaleBase = 0.75f;
        uiLayerBlock.modelViewProjection = renderView.viewProjectionMatrix * glm::translate(glm::vec3(0.0f, 0.0f, -1.2f)) * glm::scale(glm::vec3(uiScaleBase * uiScale * (io.DisplaySize.x / io.DisplaySize.y), uiScaleBase * uiScale, uiScaleBase * uiScale));

        rhi()->loadUniformBlockImmediate(ksUILayerUniformBlock, &uiLayerBlock, sizeof(UILayerUniformBlock));
        rhi()->drawNDCQuad();

        rhi()->endRenderPass(eyeRT[eyeIdx]);
      }

      // Debug feedback rendering
      {
        RHISurface::ptr debugSurface = renderAcquireDebugSurface();
        if (debugSurface) {
          RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({debugSurface}));
          rhi()->beginRenderPass(rt, kLoadInvalidate);

          // render each distortion-corrected camera view to the previously allocated region of the debug surface
          for (size_t cameraIdx = 0; cameraIdx < argusCamera->streamCount(); ++cameraIdx) {
            rhi()->setViewport(debugSurfaceCameraRects[cameraIdx]);

            RHISurface::ptr overlayTex, distortionTex;

            if (calibrationContext && calibrationContext->involvesCamera(cameraIdx)) {
              overlayTex = calibrationContext->overlaySurfaceAtIndex(calibrationContext->overlaySurfaceIndexForCamera(cameraIdx));
              distortionTex = calibrationContext->previewDistortionMapForCamera(cameraIdx);
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
        // Issue RDMA surface readbacks and write-buffer flushes
        for (size_t cameraIdx = 0; cameraIdx < rdmaRenderTargets.size(); ++cameraIdx) {
          CUgraphicsResource pReadResource = rdmaRenderSurfaces[cameraIdx]->cuGraphicsResource();
          CUDA_CHECK(cuGraphicsMapResources(1, &pReadResource, 0));

          CUmipmappedArray pReadMip = NULL;
          CUDA_CHECK(cuGraphicsResourceGetMappedMipmappedArray(&pReadMip, pReadResource));

          CUarray pReadArray = NULL;
          CUDA_CHECK(cuMipmappedArrayGetLevel(&pReadArray, pReadMip, 0));

          CUDA_MEMCPY2D copyDescriptor;
          memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

          copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
          copyDescriptor.srcArray = pReadArray;

          copyDescriptor.WidthInBytes = rdmaRenderSurfaces[cameraIdx]->width() * 4 /*bytes/pixel*/;
          copyDescriptor.Height = rdmaRenderSurfaces[cameraIdx]->height();

          copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
          copyDescriptor.dstHost = rdmaCameraBuffers[cameraIdx]->data();
          copyDescriptor.dstPitch = copyDescriptor.WidthInBytes;

          CUDA_CHECK(cuMemcpy2D(&copyDescriptor));

          cuGraphicsUnmapResources(1, &pReadResource, 0);

          rdmaContext->asyncFlushWriteBuffer(rdmaCameraBuffers[cameraIdx]);
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

