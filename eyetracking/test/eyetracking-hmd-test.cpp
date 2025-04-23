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

#include "EyeTrackingService.h"
#include "InputListener.h"
#include "Render.h"
#include "RenderBackend.h"

#include "imgui_backend.h"
#include "implot/implot.h"

#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"


FxAtomicString ksCrosshairUniformBlock("CrosshairUniformBlock");
struct CrosshairUniformBlock {
  glm::mat4 modelViewProjection[2];
  glm::vec4 color;
  float thickness;
  float pad2, pad3, pad4;
};
RHIRenderPipeline::ptr crosshairPipeline;

FxAtomicString ksImageTex("imageTex");
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

  crosshairPipeline = rhi()->compileRenderPipeline("shaders/crosshair.vtx.glsl", "shaders/crosshair.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);


  // Eyetracking service init
  EyeTrackingService* eyeTrackingService = new EyeTrackingService();
  RHISurface::ptr eyeTrackingDebugTexture;
  cv::Mat eyeTrackingDebugViewRGBA;

  int eyeTrackingCalibrationPointIdx = -1;
  const glm::vec2 eyeTrackingCalibrationPoints[] = {
    { -1,   0 },
    {  1,   0 },
    {  0,   1 },
    {  0,  -1 },
    {  1,   1 },
    {  1,  -1 },
    { -1,   1 },
    { -1,  -1 },
  };
  const size_t eyeTrackingCalibrationPointCount = (sizeof(eyeTrackingCalibrationPoints) / sizeof(eyeTrackingCalibrationPoints[0]));
  const float eyeTrackingCalibrationPointScales[] = {
    2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20
  };
  const size_t eyeTrackingCalibrationScaleCount = (sizeof(eyeTrackingCalibrationPoints) / sizeof(eyeTrackingCalibrationPoints[0]));
  const size_t eyeTrackingCalibrationTotalSampleCount = eyeTrackingCalibrationPointCount * eyeTrackingCalibrationScaleCount;

  if (argc > 1) {
    printf("Using input filename %s\n", argv[1]);
    eyeTrackingService->setInputFilename(0, argv[1]);
  } else {
    eyeTrackingService->setInputFilename(0, "/dev/video4");
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


      // Eyetracking
      if (eyeTrackingService->processFrame()) {
        if ((frameCounter & 127) == 0)
          printf("Frame %zu processing time: %.3fms inference, %.3fms post\n", frameCounter, eyeTrackingService->m_lastFrameProcessingTimeMs, eyeTrackingService->m_lastFramePostProcessingTimeMs);
      }


      if (ImGui::IsKeyPressed(ImGuiKey_Menu, /*repeat=*/ false) ||
          ImGui::IsKeyPressed(ImGuiKey_F1, /*repeat=*/ false)) {

        drawUI = !drawUI;
      }

      if (!drawUI) {

        if (eyeTrackingCalibrationPointIdx >= 0) {
          // in calibration mode
          if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
            // Accept calibration for this point, move to next.

            const glm::vec2& calPoint = eyeTrackingCalibrationPoints[eyeTrackingCalibrationPointIdx % eyeTrackingCalibrationPointCount] * eyeTrackingCalibrationPointScales[eyeTrackingCalibrationPointIdx / eyeTrackingCalibrationPointCount];

            glm::vec2 measuredPoint = glm::vec2(
              eyeTrackingService->m_processingState[0].m_pupilRawPitchDeg - eyeTrackingService->m_processingState[0].m_centerPitchDeg,
              -(eyeTrackingService->m_processingState[0].m_pupilRawYawDeg - eyeTrackingService->m_processingState[0].m_centerYawDeg));

            printf("=== Calibration point %d: cal angles {%f, %f}, measured {%.3f, %.3f}, delta {%.3f, %.3f}, ratio {%.3f, %.3f. dot = %.3fdeg}\n",
              eyeTrackingCalibrationPointIdx,
              calPoint.x, calPoint.y,
              measuredPoint.x, measuredPoint.y,
              measuredPoint.x - calPoint.x, measuredPoint.y - calPoint.y,
              measuredPoint.x / calPoint.x, measuredPoint.y / calPoint.y,
              glm::degrees(glm::acos(glm::dot(eyeTrackingService->m_processingState[0].fitPupilNormal(), eyeTrackingService->m_processingState[0].centerPupilNormal()))));

            glm::vec3 pn = eyeTrackingService->m_processingState[0].fitPupilNormal();
            glm::vec3 cn = eyeTrackingService->m_processingState[0].centerPupilNormal();
            FILE* fp = fopen("calibration-samples.txt", "ab");
            fprintf(fp, "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
              calPoint.x, calPoint.y,
              measuredPoint.x, measuredPoint.y,
              pn.x, pn.y, pn.z,
              cn.x, cn.y, cn.z);
            fclose(fp);

            eyeTrackingCalibrationPointIdx = eyeTrackingCalibrationPointIdx + 1;
            if (eyeTrackingCalibrationPointIdx >= eyeTrackingCalibrationTotalSampleCount)
              eyeTrackingCalibrationPointIdx = -1;
          }
        }
      }

      ImGui_ImplFxRHI_NewFrame();
      ImGui_ImplInputListener_NewFrame();
      ImGui::NewFrame();

      ++frameCounter;

      if (drawUI) {
        // GUI support
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, io.DisplaySize.y), 0, /*pivot=*/ImVec2(0.5f, 1.0f)); // bottom-center aligned
        ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_Always); // always auto-size to contents, since we don't provide a way to resize the UI
        ImGui::Begin("Overlay", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);


        // TODO settings go here

        if (ImGui::Button("ET calibration points")) {
          eyeTrackingCalibrationPointIdx = 0;
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
        ImGui::Begin("StatusBar", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        char timebuf[64];
        time_t t = time(NULL);
        strftime(timebuf, 64, "%a %b %e %T", localtime(&t));
        ImGui::TextUnformatted(timebuf);
        ImGui::SameLine(); ImGui::Separator(); ImGui::SameLine();
        ImGui::Text("%.1fFPS", io.Framerate);

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




      // Render eyetracking debug view
      {
        const cv::Mat& debugView = eyeTrackingService->getDebugViewForEye(0);
        if (debugView.cols && debugView.rows) {

          if (!eyeTrackingDebugTexture || (eyeTrackingDebugTexture->width() != debugView.cols) || (eyeTrackingDebugTexture->height() != debugView.rows)) {
            eyeTrackingDebugTexture = rhi()->newTexture2D(debugView.cols, debugView.rows, kSurfaceFormat_RGBA8);
          }

          // TODO this is probably inefficient
          cv::cvtColor(/*src=*/ debugView, /*dst=*/ eyeTrackingDebugViewRGBA, cv::COLOR_BGR2RGBA);
          rhi()->loadTextureData(eyeTrackingDebugTexture, kVertexElementTypeUByte4N, eyeTrackingDebugViewRGBA.ptr());


          rhi()->bindBlendState(disabledBlendState);
          rhi()->bindDepthStencilState(disabledDepthStencilState);
          rhi()->bindRenderPipeline(uiLayerStereoPipeline);
          rhi()->loadTexture(ksImageTex, eyeTrackingDebugTexture, linearClampSampler);
          // rhi()->setViewports(eyeViewports, 2); // should already be set

          UILayerStereoUniformBlock ub;
          glm::mat4 modelMatrix = glm::translate(glm::vec3(0.0f, 0.0f, -uiDepth)) * glm::scale(glm::vec3(uiScale * (static_cast<float>(eyeTrackingDebugTexture->width()) / static_cast<float>(eyeTrackingDebugTexture->height())), -uiScale, uiScale));
          ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
          ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;

          rhi()->loadUniformBlockImmediate(ksUILayerStereoUniformBlock, &ub, sizeof(ub));
          rhi()->drawNDCQuad();

        }
      }

      // Render eyetracking gizmos

      // Eyetracking calibration point
      if (eyeTrackingCalibrationPointIdx >= 0 && eyeTrackingCalibrationPointIdx < eyeTrackingCalibrationTotalSampleCount) {
        CrosshairUniformBlock ub;

        const glm::vec2& calPoint = eyeTrackingCalibrationPoints[eyeTrackingCalibrationPointIdx % eyeTrackingCalibrationPointCount] * eyeTrackingCalibrationPointScales[eyeTrackingCalibrationPointIdx / eyeTrackingCalibrationPointCount];

        // TODO wire up rotation angles
        glm::mat4 modelMatrix =
            glm::eulerAngleXY(glm::radians(calPoint.x), glm::radians(calPoint.y))
          * glm::translate(glm::vec3(0.0f, 0.0f, -uiDepth))
          * glm::scale(glm::vec3(0.005f));

        ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
        ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;
        ub.color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        ub.thickness = 0.8f;

        rhi()->bindRenderPipeline(crosshairPipeline);
        rhi()->bindDepthStencilState(disabledDepthStencilState);

        rhi()->loadUniformBlockImmediate(ksCrosshairUniformBlock, &ub, sizeof(ub));
        // rhi()->setViewports(eyeViewports, 2); // should already be set

        rhi()->bindStreamBuffer(0, ndcQuadVBO);
        rhi()->drawPrimitives(0, 4, /*instanceCount=*/ 2);
      }


      // Eye crosshair
      if (eyeTrackingService->m_processingState[0].m_calibrationState == EyeTrackingService::kCalibrated) {
        CrosshairUniformBlock ub;

        glm::mat4 centerOffsetMatrix = glm::transpose(glm::eulerAngleXY(glm::radians(eyeTrackingService->m_processingState[0].m_centerPitchDeg), -glm::radians(eyeTrackingService->m_processingState[0].m_centerYawDeg)));

        // TODO wire up rotation angles
        glm::mat4 modelMatrix =
            centerOffsetMatrix
          * glm::eulerAngleXYZ(
              glm::radians( eyeTrackingService->m_processingState[0].m_pupilRawPitchDeg),
              glm::radians(-eyeTrackingService->m_processingState[0].m_pupilRawYawDeg),
              glm::radians(0.0f))
          * glm::translate(glm::vec3(0.0f, 0.0f, -uiDepth))
          * glm::scale(glm::vec3(0.005f));

        ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
        ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;
        ub.color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        ub.thickness = 0.5f;

        rhi()->bindRenderPipeline(crosshairPipeline);
        rhi()->bindDepthStencilState(disabledDepthStencilState);

        rhi()->loadUniformBlockImmediate(ksCrosshairUniformBlock, &ub, sizeof(ub));
        // rhi()->setViewports(eyeViewports, 2); // should already be set

        rhi()->bindStreamBuffer(0, ndcQuadVBO);
        rhi()->drawPrimitives(0, 4, /*instanceCount=*/ 2);
      }

      // Center crosshair
      if (eyeTrackingService->m_processingState[0].m_calibrationState <= EyeTrackingService::kCentering) {
        CrosshairUniformBlock ub;

        glm::mat4 modelMatrix =
            glm::translate(glm::vec3(0.0f, 0.0f, -uiDepth))
          * glm::scale(glm::vec3(0.00375f));

        ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
        ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;
        ub.color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        ub.thickness = 0.75f;

        rhi()->bindRenderPipeline(crosshairPipeline);
        rhi()->bindDepthStencilState(disabledDepthStencilState);

        rhi()->loadUniformBlockImmediate(ksCrosshairUniformBlock, &ub, sizeof(ub));
        // rhi()->setViewports(eyeViewports, 2); // should already be set

        rhi()->bindStreamBuffer(0, ndcQuadVBO);
        rhi()->drawPrimitives(0, 4, /*instanceCount=*/ 2);
      }


      if (debugRenderTiming)
        rhi()->endTimerQuery(viewRenderQuery);


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

      rhi()->endRenderPass(eyeRT);


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

  delete eyeTrackingService;
  eyeTrackingService = nullptr;

  // clear screen
  rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
  rhi()->endRenderPass(windowRenderTarget);
  rhi()->swapBuffers(windowRenderTarget);

  RenderShutdown();

  FxThreading::detail::shutdown();

  return 0;
}

