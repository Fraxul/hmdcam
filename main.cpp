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

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"

#include "ArgusCamera.h"
#include "InputListener.h"
#include "Render.h"

#include "imgui_backend.h"

#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "Calibration.h"

//#define LATENCY_DEBUG

// Camera config
// Size parameters for sensor mode selection.
// Note that changing the sensor mode will invalidate the calibration
// (Pixel coordinates are baked into the calibration data)
size_t s_cameraWidth, s_cameraHeight;

// Requested capture rate for the camera. This should be the framerate of the display device, with as much precision as possible.
// TODO: autodetect this. (current value pulled from running `fbset`)
const double s_cameraFramerate = 89.527;

// #define SWAP_CAMERA_EYES

// Mapping of libargus camera device ID to index 0 (left) and 1 (right).
#ifdef SWAP_CAMERA_EYES
  #define LEFT_CAMERA_INDEX 1
  #define RIGHT_CAMERA_INDEX 0
#else
  #define LEFT_CAMERA_INDEX 0
  #define RIGHT_CAMERA_INDEX 1
#endif

// Camera render parameters
float scaleFactor = 1.0f;
float stereoOffset = 0.0f;
bool renderSBS = true;
bool useMask = true;
int sbsSeparatorWidth = 4;

// Camera info/state
ArgusCamera* stereoCamera;
RHISurface::ptr cameraDistortionMap[2];
RHISurface::ptr cameraMask[2];

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

int main(int argc, char* argv[]) {

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
  stereoCamera = new ArgusCamera(renderEGLDisplay(), renderEGLContext(), {LEFT_CAMERA_INDEX, RIGHT_CAMERA_INDEX}, s_cameraFramerate);
  s_cameraWidth = stereoCamera->streamWidth();
  s_cameraHeight = stereoCamera->streamHeight();

  renderSetDebugSurfaceSize(s_cameraWidth * 2, s_cameraHeight);

  // Generate derived data for calibration
  initCalibration();

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  // Calibration mode
  {
    // Try reading calibration data from the file
    readCalibrationData();

    // Calibrate individual cameras
    if (!haveIntrinsicCalibration()) {
      doIntrinsicCalibration();

      if (want_quit)
        goto quit;

      // Incrementally save intrinsic calibration data if it was updated this run
      saveCalibrationData(); 
    }

    if (!haveStereoCalibration()) {
      doStereoCalibration();
      saveCalibrationData();
    }

    if (want_quit)
      goto quit;

    generateCalibrationDerivedData();

    // Compute new distortion maps with the now-valid stereo calibration.
    updateCameraDistortionMap(0, true);
    updateCameraDistortionMap(1, true);

  } // Calibration mode

  // Load masks.
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
        stereoCamera->readFrame();

        RHIRenderTarget::ptr snapRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({cameraMask[cameraIdx]}));
        rhi()->beginRenderPass(snapRT, kLoadInvalidate);
        // This pipeline flips the Y axis for OpenCV's coordinate system, which is the same as the PNG coordinate system
        rhi()->bindRenderPipeline(camGreyscalePipeline);
        rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
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

  {
    stereoCamera->setRepeatCapture(true);

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

    while (!want_quit) {
      if (testButton(kButtonPower)) {
        drawUI = !drawUI;
      }

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplInputListener_NewFrame();
      ImGui::NewFrame();

      ++frameCounter;

      stereoCamera->readFrame();

      if (previousCaptureTimestamp) {
        currentCaptureIntervalMs = static_cast<double>(stereoCamera->frameSensorTimestamp(0) - previousCaptureTimestamp) / 1000000.0;
        captureInterval(currentCaptureIntervalMs);
      }
      previousCaptureTimestamp = stereoCamera->frameSensorTimestamp(0);


      if (drawUI) {
        // GUI support
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, io.DisplaySize.y), 0, /*pivot=*/ImVec2(0.5f, 1.0f)); // bottom-center aligned
        ImGui::Begin("Overlay", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        //ImGui::Text("Config");
        ImGui::Checkbox("SBS", &renderSBS);
        ImGui::Checkbox("Mask", &useMask);
        ImGui::SliderFloat("Scale", &scaleFactor, 0.5f, 2.0f);
        ImGui::SliderFloat("Stereo Offset", &stereoOffset, -0.5f, 0.5f);
        if (renderSBS) {
          ImGui::SliderInt("Separator Width", (int*) &sbsSeparatorWidth, 0, 32);
        }
        {
          const auto& meta = stereoCamera->frameMetadata(0);
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

        for (int cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
          if (renderSBS == false && (cameraIdx != eyeIdx))
            continue;

          rhi()->bindRenderPipeline(camUndistortMaskPipeline);
          rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
          rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[cameraIdx], linearClampSampler);
          rhi()->loadTexture(ksMaskTex, useMask ? cameraMask[cameraIdx] : disabledMaskTex, linearClampSampler);

          // coordsys right now: -X = left, -Z = into screen
          // (camera is at the origin)
          float stereoOffsetSign = (cameraIdx == 0 ? -1.0f : 1.0f);
          const glm::vec3 tx = glm::vec3(stereoOffsetSign * stereoOffset, 0.0f, -3.0f);
          glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(stereoCamera->streamWidth()) / static_cast<float>(stereoCamera->streamHeight())), scaleFactor, 1.0f)); // TODO
          // Intentionally ignoring the eyeView matrix here. Camera to eye stereo offset is controlled directly by the stereoOffset variable
          glm::mat4 mvp = eyeProjection[eyeIdx] * eyeView[eyeIdx] * model;

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

          NDCClippedQuadUniformBlock ub;
          ub.modelViewProjection = mvp;
          if (cameraIdx == 0) { // left
            ub.minUV = glm::vec2(0.0f,  0.0f);
            ub.maxUV = glm::vec2(uClipFrac, 1.0f);
          } else { // right
            ub.minUV = glm::vec2(1.0f - uClipFrac, 0.0f);
            ub.maxUV = glm::vec2(1.0f,  1.0f);
          }

          rhi()->loadUniformBlockImmediate(ksNDCClippedQuadUniformBlock, &ub, sizeof(NDCClippedQuadUniformBlock));

          rhi()->drawNDCQuad();
        } // camera loop

        if (renderSBS && sbsSeparatorWidth) {
          rhi()->bindRenderPipeline(solidQuadPipeline);
          SolidQuadUniformBlock ub;
          // Scale to reduce the X-width of the -1...1 quad to the size requested in sbsSeparatorWidth
          ub.modelViewProjection = eyeProjection[eyeIdx] * eyeView[eyeIdx] * glm::translate(glm::vec3(0.0f, 0.0f, -3.0f)) * glm::scale(glm::vec3(static_cast<float>(sbsSeparatorWidth * 4) / static_cast<float>(eyeRT[eyeIdx]->width()), 1.0f, 1.0f));
          ub.color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
          rhi()->loadUniformBlockImmediate(ksSolidQuadUniformBlock, &ub, sizeof(SolidQuadUniformBlock));
          rhi()->drawNDCQuad();
        }

        // UI overlay
        if (drawUI) {
          rhi()->bindBlendState(standardAlphaOverBlendState);
          rhi()->bindRenderPipeline(uiLayerPipeline);
          rhi()->loadTexture(ksImageTex, guiTex, linearClampSampler);
          UILayerUniformBlock uiLayerBlock;
          uiLayerBlock.modelViewProjection = eyeProjection[eyeIdx] * eyeView[eyeIdx] * glm::translate(glm::vec3(0.0f, 0.0f, -2.0f)) * glm::scale(glm::vec3(io.DisplaySize.x / io.DisplaySize.y, 1.0f, 1.0f));

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

          for (int cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
            rhi()->setViewport(cameraIdx == 0 ? leftRect : rightRect);

            rhi()->bindRenderPipeline(camUndistortMaskPipeline);
            rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
            rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[cameraIdx], linearClampSampler);
            rhi()->loadTexture(ksMaskTex, useMask ? cameraMask[cameraIdx] : disabledMaskTex, linearClampSampler);

            NDCClippedQuadUniformBlock ub;
            ub.modelViewProjection = glm::mat4(1.0f); // identity
            ub.minUV = glm::vec2(0.0f);
            ub.maxUV = glm::vec2(1.0f);

            rhi()->loadUniformBlockImmediate(ksNDCClippedQuadUniformBlock, &ub, sizeof(NDCClippedQuadUniformBlock));
            rhi()->drawNDCQuad();
          }

          if (drawUI) {
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
            stereoCamera->setTargetCaptureIntervalNs(boost::accumulators::rolling_mean(frameInterval));
          }
#endif

          io.DeltaTime = static_cast<double>(interval / 1000000000.0);
        }

        currentCaptureLatencyMs = static_cast<double>(thisFrameTimestamp - stereoCamera->frameSensorTimestamp(0)) / 1000000.0;
        captureLatency(currentCaptureLatencyMs);

        previousFrameTimestamp = thisFrameTimestamp;
      }
    } // Camera rendering loop
  }
quit:
  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  // clear screen
  rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
  rhi()->endRenderPass(windowRenderTarget);
  rhi()->swapBuffers(windowRenderTarget);

  stereoCamera->stop();
  delete stereoCamera;

  RenderShutdown();

  return 0;
}

