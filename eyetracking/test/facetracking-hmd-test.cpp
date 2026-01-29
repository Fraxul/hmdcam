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
#include <glm/gtx/euler_angles.hpp>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"

#include "common/ScrollingBuffer.h"
#include "common/Timing.h"
#include "common/glmCvInterop.h"
#include "common/FxCamera.h"
#include "common/FxThreading.h"
#include "common/CANBus.h"

#include "FaceTrackingService.h"
#include "InputListener.h"
#include "Render.h"
#include "RenderBackend.h"

#include "imgui_backend.h"
#include "implot/implot.h"

#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

float uiScale = 0.2f;
float uiDepth = 0.4f;

// Profiling data
struct FrameTimingData {
  FrameTimingData() {}

  float viewRenderTimeMs = 0;
  float distortionRenderTimeMs = 0;
  float submitTimeMs = 0;
};

ScrollingBuffer<FrameTimingData> s_timingDataBuffer(512);

bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;

  // Restore signal handlers so the program is still interruptable if clean shutdown gets stuck
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
}

int main(int argc, char* argv[]) {
  const bool debugRenderTiming = true;
  const bool debugPrintLatency = false;

  // Clear the DISPLAY environment variable. Having this variable exist, even if blank,
  // will break EGL initialization in libargus and nvbufsurface.
  unsetenv("DISPLAY");

  startInputListenerThread();

  if (!RenderInit(kRenderBackendVKDirect)) {
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
  ImGui_ImplInputListener_Init();
  ImGui_ImplFxRHI_Init();

  io.DisplaySize = ImVec2(512.0f, 512.0f); // Not the full size, but the size of our overlay RT
  io.DisplayFramebufferScale = ImVec2(2.0f, 2.0f); // Use HiDPI rendering

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);


  // Facetracking service init
  FaceTrackingService* faceTrackingService = new FaceTrackingService();
  faceTrackingService->m_debugShowFeedbackView = true; // Default to feedback view enabled

  if (argc > 1) {
    printf("Overriding input camera %s\n", argv[1]);
    faceTrackingService->setInputDeviceOverride(argv[1]);
  }

  {
    // Accumulators to track frame timing statistics
    uint64_t frameCounter = 0;

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

    bool drawUI = false;
    bool captureMode = false;

    // Perf queries
    RHITimerQuery::ptr viewRenderQuery = rhi()->newTimerQuery();
    RHITimerQuery::ptr distortionRenderQuery = rhi()->newTimerQuery();

    // Main display loop
    while (!want_quit) {
      FrameTimingData timingData;

      if (debugRenderTiming) {
        // Read previous loop's timer query results
        timingData.viewRenderTimeMs = static_cast<float>(rhi()->getQueryResult(viewRenderQuery)) / 1000000.0f;
        timingData.distortionRenderTimeMs = static_cast<float>(rhi()->getQueryResult(distortionRenderQuery)) / 1000000.0f;
      }

      uint64_t frameStartTimeNs = currentTimeNs();


      // Tracking
      if (faceTrackingService->processFrame()) {
        if ((frameCounter & 127) == 0) {
          if (faceTrackingService->m_processingState.processingThreadAlive()) {
            printf("Frame %zu: %s\n",
              frameCounter, faceTrackingService->getDebugPerfStats());
          }
        }
        faceTrackingService->CANTransmitTrackingData();

        // HACK: Transmit some empty eyetracking frames as well so that the eye controller will use the facetracking CAN data.
        {
          constexpr uint16_t kPortID = 201;
          constexpr uint8_t state = 2; // kStatePupilLock

          SerializationBuffer buf;
          buf.reserve(8);

          buf.put_u8(state);
          buf.put_i16_le(0); // pitch
          buf.put_i16_le(0); // yaw

          canbus()->transmitMessage(kPortID, buf);
        }

      }


      if (ImGui::IsKeyPressed(ImGuiKey_Menu, /*repeat=*/ false) ||
          ImGui::IsKeyPressed(ImGuiKey_F1, /*repeat=*/ false)) {

        drawUI = !drawUI;
      }

      if (!drawUI) {
        if (captureMode && ImGui::IsKeyPressed(ImGuiKey_Space)) {
          faceTrackingService->requestCapture();
        }
      }

      ImGui_ImplFxRHI_NewFrame();
      ImGui_ImplInputListener_NewFrame();
      ImGui::NewFrame();

      ++frameCounter;

      if (drawUI) {
        // Exit capture mode when re-entering UI
        captureMode = false;

        // GUI support
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, io.DisplaySize.y), 0, /*pivot=*/ImVec2(0.5f, 1.0f)); // bottom-center aligned
        ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always); // always auto-size to contents, since we don't provide a way to resize the UI
        ImGui::Begin("Overlay", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);


        // TODO settings go here

        if (ImGui::Button("Capture mode")) {
          captureMode = true;
          drawUI = false; // exit UI when selecting capture mode.
        }

        if (ImGui::CollapsingHeader("Facetracking Config")) {
          faceTrackingService->renderIMGUI();
        }

        if (ImGui::CollapsingHeader("Performance")) {
          int plotFlags = ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;

          if (debugRenderTiming && ImPlot::BeginPlot("##RenderTiming", ImVec2(-1,150), /*flags=*/ plotFlags)) {
              ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
              ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);
              ImPlot::SetupAxisLimits(ImAxis_X1, 0, s_timingDataBuffer.size(), ImPlotCond_Always);
              ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 12.0f, ImPlotCond_Always);
              ImPlot::SetupFinish();

              ImPlot::PlotLine("View Render", &s_timingDataBuffer.data()[0].viewRenderTimeMs, s_timingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::PlotLine("HMD Distortion",  &s_timingDataBuffer.data()[0].distortionRenderTimeMs,  s_timingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, s_timingDataBuffer.offset(), sizeof(FrameTimingData));
              ImPlot::EndPlot();
          }
        } // Performance

        ImGui::End();

      } else {
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, 0), 0, /*pivot=*/ImVec2(0.5f, 0.0f)); // top-center aligned
        ImGui::SetNextWindowSize(ImVec2(0, -1), ImGuiCond_Always); // grow-only auto-size to X, frame auto-size to Y
        ImGui::Begin("StatusBar", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        char timebuf[64];
        time_t t = time(NULL);
        strftime(timebuf, 64, "%a %b %e %T", localtime(&t));
        ImGui::TextUnformatted(timebuf);
        ImGui::SameLine(); ImGui::Separator(); ImGui::SameLine();
        ImGui::Text("%.1fFPS", io.Framerate);

        if (captureMode) {
          ImGui::Text("Capture mode active");
        }

        ImGui::End();


        // Facetracking network output debug
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, io.DisplaySize.y), 0, /*pivot=*/ImVec2(0.5f, 1.0f)); // bottom-center aligned
        ImGui::SetNextWindowSize(ImVec2(0, -1), ImGuiCond_Always); // grow-only auto-size to X, frame auto-size to Y
        ImGui::Begin("FTOutput", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        if (faceTrackingService && faceTrackingService->m_processingState.m_channelData) {
          ImGui::SliderFloat("Brow Position", &faceTrackingService->m_processingState.m_browPosition, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_None);
          const char* channelNames[] = { "brow_position" };

          for (uint32_t channelIdx = 0; channelIdx < faceTrackingService->m_processingState.m_trackingOutputChannels; ++channelIdx) {
            ImGui::SliderFloat(channelNames[channelIdx], &faceTrackingService->m_processingState.m_channelData[channelIdx], 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_None);
          }
        } else {
          ImGui::Text("(no channel data)");
        }
        ImGui::End();
      }

      rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
      rhi()->beginRenderPass(guiRT, kLoadClear);
      ImGui::Render();
      ImGui_ImplFxRHI_RenderDrawData(guiRT, ImGui::GetDrawData());
      rhi()->endRenderPass(guiRT);

      if ((frameCounter & 0x7fUL) == 0) {
        if (debugPrintLatency) {
          printf("Frame interval: % .6f ms (% .6f fps) min=%.3g max=%.3g median=%.3g\n",
            static_cast<double>(boost::accumulators::mean(frameInterval)) / 1000000.0,
            1000000000.0 / static_cast<double>(boost::accumulators::mean(frameInterval)),

            static_cast<double>(boost::accumulators::min(frameInterval)) / 1000000.0,
            static_cast<double>(boost::accumulators::max(frameInterval)) / 1000000.0,
            static_cast<double>(boost::accumulators::median(frameInterval)) / 1000000.0);
        }

        frameInterval = {};
      }

      rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

      // Note that our camera uses reversed depth projection -- we clear to 0 and use a "greater" depth-test.
      if (debugRenderTiming)
        rhi()->beginTimerQuery(viewRenderQuery);

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

      // === Render objects ===




      // Render gizmos

      // Overlays that draw behind the UI
      faceTrackingService->renderSceneGizmos_preUI(renderViews);

      // UI overlay
      {
        rhi()->bindBlendState(standardAlphaOverBlendState);
        rhi()->bindDepthStencilState(disabledDepthStencilState);
        rhi()->bindRenderPipeline(uiLayerStereoPipeline);
        rhi()->loadTexture(ksImageTex, guiTex, linearClampSampler);
        // rhi()->setViewports(eyeViewports, 2); // should already be set

        UILayerStereoUniformBlock ub;
        glm::mat4 modelMatrix = glm::translate(glm::vec3(0.0f, 0.0f, -uiDepth)) * glm::scale(glm::vec3(uiScale * (io.DisplaySize.x / io.DisplaySize.y), uiScale, uiScale));
        ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
        ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;

        rhi()->loadUniformBlockImmediate(ksUILayerStereoUniformBlock, &ub, sizeof(ub));
        rhi()->drawNDCQuad();
      }

      // Cursors and any other FaceTrackingService-provided gizmos
      faceTrackingService->renderSceneGizmos_postUI(renderViews);

      rhi()->endRenderPass(eyeRT);

      if (debugRenderTiming)
        rhi()->endTimerQuery(viewRenderQuery);


      timingData.submitTimeMs = deltaTimeMs(frameStartTimeNs, currentTimeNs());

      if (debugRenderTiming)
        rhi()->beginTimerQuery(distortionRenderQuery);

      renderHMDFrame();

      if (debugRenderTiming)
        rhi()->endTimerQuery(distortionRenderQuery);

      {
        uint64_t thisFrameTimestamp = currentTimeNs();
        if (previousFrameTimestamp) {
          uint64_t interval = thisFrameTimestamp - previousFrameTimestamp;
          frameInterval(interval);
          io.DeltaTime = static_cast<double>(interval / 1000000000.0);
        }

        previousFrameTimestamp = thisFrameTimestamp;
      }

      s_timingDataBuffer.push_back(timingData);
    } // Camera rendering loop
  }


  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  delete faceTrackingService;
  faceTrackingService = nullptr;

  // clear screen
  rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
  rhi()->endRenderPass(windowRenderTarget);
  rhi()->swapBuffers(windowRenderTarget);

  RenderShutdown();

  FxThreading::detail::shutdown();

  return 0;
}

