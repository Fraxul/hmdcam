#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>

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
#include "rhi/CudaUtil.h"

#include "ArgusCamera.h"
#include "common/CameraSystem.h"
#include "common/DepthMapGenerator.h"
#include "InputListener.h"
#include "Render.h"

#include "imgui_backend.h"

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

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
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

  rdmaContext = RDMAContext::createServerContext();

  if (!rdmaContext) {
    printf("RDMA server context initialization failed; RDMA service will be unavailable.\n");
  }

  startInputListenerThread();
  if (!RenderInit()) {
    printf("RenderInit() failed\n");
    return 1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = NULL; // Disable INI file load/save
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Keyboard navigation (mapped from the InputListener media remote interface)
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Gamepad navigation (not used right now)
  ImGui_ImplOpenGL3_Init(NULL);
  ImGui_ImplInputListener_Init();

  io.DisplaySize = ImVec2(512.0f, 512.0f); // Not the full size, but the size of our overlay RT
  io.DisplayFramebufferScale = ImVec2(2.0f, 2.0f);

  // Open the cameras
  // TODO have ArgusCamera autodetect indices of and open all CSI cameras
  argusCamera = new ArgusCamera(renderEGLDisplay(), renderEGLContext(), {0, 1}, s_cameraFramerate);
  s_cameraWidth = argusCamera->streamWidth();
  s_cameraHeight = argusCamera->streamHeight();

  renderSetDebugSurfaceSize(s_cameraWidth * 2, s_cameraHeight);

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
  {
    CameraSystem::View& v = cameraSystem->viewAtIndex(0);
    if (v.isStereo && v.haveStereoRectificationParameters()) {
      v.depthMapGenerator = new DepthMapGenerator(cameraSystem, 0);
    }
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
    bool debugModeSwitch = true;
    float uiScale = 1.0f;
    boost::scoped_ptr<CameraSystem::CalibrationContext> calibrationContext;

    while (!want_quit) {
      if (testButton(kButtonPower)) {
        drawUI = !drawUI;
      }

      if (!drawUI) {
        // calling testButton eats the inputs, so only do that if we're not drawing the UI.
        if (testButton(kButtonDown)) {
          debugModeSwitch = !debugModeSwitch;
        }
      }

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplInputListener_NewFrame();
      ImGui::NewFrame();

      ++frameCounter;

      // Only use repeating captures if we're not in calibration. The variable CPU-side delays for calibration image processing usually end up crashing libargus.
      argusCamera->setRepeatCapture(!((bool) calibrationContext));

      argusCamera->readFrame();

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
      if (!calibrationContext && !(rdmaContext && rdmaContext->hasPeerConnections())) {
        for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
          if (cameraSystem->viewAtIndex(viewIdx).depthMapGenerator) {
            cameraSystem->viewAtIndex(viewIdx).depthMapGenerator->processFrame();
          }
        }
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
            if (v.depthMapGenerator) {
              v.depthMapGenerator->renderIMGUI();
            }
          }
        }


        ImGui::SliderFloat("UI Scale", &uiScale, 0.5f, 1.5f);
        ImGui::Checkbox("Debug output: Distortion correction", &debugUseDistortion);
        ImGui::Text("Debug URL: %s", renderDebugURL().c_str());

        {
          const auto& meta = argusCamera->frameMetadata(0);
          ImGui::Text("Exp=1/%usec %uISO DGain=%f AGain=%f",
            (unsigned int) (1000000.0f / static_cast<float>(meta.sensorExposureTimeNs/1000)), meta.sensorSensitivityISO, meta.ispDigitalGain, meta.sensorAnalogGain);
        }

        ImGui::Text("Lat=%.1fms (%.1fms-%.1fms) %.1fFPS", currentCaptureLatencyMs, boost::accumulators::min(captureLatency), boost::accumulators::max(captureLatency), io.Framerate);
        ImGui::End();

        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
        rhi()->beginRenderPass(guiRT, kLoadClear);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        rhi()->endRenderPass(guiRT);
      } else {
        ImGui::EndFrame();
      }

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
        rhi()->beginRenderPass(eyeRT[eyeIdx], kLoadClear);

        for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
          CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);

          if (debugModeSwitch && v.isStereo && v.depthMapGenerator && !calibrationContext && !(rdmaContext && rdmaContext->hasPeerConnections())) {
            FxRenderView renderView;
            // TODO actual camera setup here. renderDisparityDepthMap only uses the viewProjection matrix.
            renderView.viewMatrix = eyeView[eyeIdx];
            renderView.projectionMatrix = eyeProjection[eyeIdx];
            renderView.viewProjectionMatrix = renderView.projectionMatrix * renderView.viewMatrix; 

            v.depthMapGenerator->renderDisparityDepthMap(renderView);

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

              glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(fovScaleFactor * zoomFactor, fovScaleFactor * zoomFactor * (static_cast<float>(argusCamera->streamHeight()) / static_cast<float>(argusCamera->streamWidth())) , 1.0f));

              // Intentionally ignoring the eyeView matrix here. Camera to eye stereo offset is controlled directly by the stereoOffset variable
              glm::mat4 mvp = eyeProjection[eyeIdx] * eyeView[eyeIdx] * model;

              RHISurface::ptr overlayTex, distortionTex;
              size_t drawFlags = 0;
              if (calibrationContext && calibrationContext->isViewContext() && calibrationContext->getCameraOrViewIndex() == viewIdx) {
                // Calibrating a stereo view that includes this camera
                overlayTex = calibrationContext->overlaySurfaceAtIndex(viewEyeIdx);
                distortionTex = cameraSystem->cameraAtIndex(v.cameraIndices[viewEyeIdx]).intrinsicDistortionMap;

              } else if (calibrationContext && calibrationContext->isCameraContext() && calibrationContext->getCameraOrViewIndex() == v.cameraIndices[viewEyeIdx]) {
                // Calibrating this camera's intrinsic distortion
                overlayTex = calibrationContext->overlaySurfaceAtIndex(0);

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
          ub.modelViewProjection = eyeProjection[eyeIdx] * eyeView[eyeIdx] * glm::translate(glm::vec3(0.0f, 0.0f, -1.0f)) * glm::scale(glm::vec3(static_cast<float>(sbsSeparatorWidth * 4) / static_cast<float>(eyeRT[eyeIdx]->width()), 1.0f, 1.0f));
          ub.color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
          rhi()->loadUniformBlockImmediate(ksSolidQuadUniformBlock, &ub, sizeof(SolidQuadUniformBlock));
          rhi()->drawNDCQuad();
        }
*/

        // UI overlay
        if (drawUI || calibrationContext) {
          rhi()->bindBlendState(standardAlphaOverBlendState);
          rhi()->bindRenderPipeline(uiLayerPipeline);
          rhi()->loadTexture(ksImageTex, guiTex, linearClampSampler);
          UILayerUniformBlock uiLayerBlock;
          float uiScaleBase = 0.15f;
          uiLayerBlock.modelViewProjection = eyeProjection[eyeIdx] * eyeView[eyeIdx] * glm::translate(glm::vec3(0.0f, 0.0f, -0.25f)) * glm::scale(glm::vec3(uiScaleBase * uiScale * (io.DisplaySize.x / io.DisplaySize.y), uiScaleBase * uiScale, uiScaleBase * uiScale));

          rhi()->loadUniformBlockImmediate(ksUILayerUniformBlock, &uiLayerBlock, sizeof(UILayerUniformBlock));
          rhi()->drawNDCQuad();
        }

        rhi()->endRenderPass(eyeRT[eyeIdx]);
      }

      // Debug feedback rendering
      {
        RHISurface::ptr debugSurface = renderAcquireDebugSurface();
        if (debugSurface) {
          RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({debugSurface}));
          rhi()->beginRenderPass(rt, kLoadInvalidate);

          // render each distortion-corrected camera view to half of the debug surface
          RHIRect leftRect = RHIRect::xywh(0, 0, debugSurface->width() / 2, debugSurface->height());
          RHIRect rightRect = RHIRect::xywh(debugSurface->width() / 2, 0, debugSurface->width() / 2, debugSurface->height());

          // TODO split up debug view for more than 2 cameras
          for (int cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
            rhi()->setViewport(cameraIdx == 0 ? leftRect : rightRect);

            RHISurface::ptr overlayTex, distortionTex;
            distortionTex = cameraSystem->cameraAtIndex(cameraIdx).intrinsicDistortionMap;

            if (calibrationContext && calibrationContext->involvesCamera(cameraIdx)) {
              if (calibrationContext->isViewContext()) {
                // Calibrating a stereo view that includes this camera
                overlayTex = calibrationContext->overlaySurfaceAtIndex(calibrationContext->overlaySurfaceIndexForCamera(cameraIdx));
              } else {
                // Intrinsic calibration for camera. Disable distortion correction.
                overlayTex = calibrationContext->overlaySurfaceAtIndex(calibrationContext->overlaySurfaceIndexForCamera(cameraIdx));
                distortionTex = RHISurface::ptr();
              }
            } else if (debugUseDistortion) {
              // Distortion-corrected view
            } else {
              // No-distortion / direct passthrough
              distortionTex = RHISurface::ptr();
            }

            renderDrawCamera(cameraIdx, /*drawFlags=*/0, distortionTex, overlayTex, /*mvp=*/glm::mat4(1.0f) /*identity*/);
          }

          if (drawUI || calibrationContext) {
            // Draw the UI on the center-bottom of the debug surface
            RHIRect uiDestRect = RHIRect::xywh((debugSurface->width() / 2) - (guiTex->width() / 2), debugSurface->height() - guiTex->height(), guiTex->width(), guiTex->height());
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

      if (rdmaContext) {
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

  return 0;
}

