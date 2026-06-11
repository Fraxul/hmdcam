#include "imgui.h"
#include "imgui/backends/imgui_impl_sdl2.h"
#include "implot/implot.h"
#include <stdio.h>
#include <SDL.h>
#include <SDL_vulkan.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <glm/gtx/transform.hpp>

#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "CharucoMultiViewCalibration.h"
#include "DebugCameraProvider.h"
#include "RenderBackendSDLVK.h"
#include "common/CameraSystem.h"
#include "common/FxCamera.h"
#include "common/FxThreading.h"
#include "common/DepthMapGenerator.h"
#include "common/glmCvInterop.h"
#include "common/remapArray.h"

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICUDA.h"
#include "rhi/imgui/RHIImGuiBackend.h"
#include "rhi/vk/RHIVK.h"

#include <cuda.h>

DebugCameraProvider* cameraProvider;
CameraSystem* cameraSystem;
FxCamera* sceneCamera;
DepthMapGenerator* depthMapGenerator;
SHMSegment<DepthMapSHM>* shm;
RenderBackendSDLVK* renderBackend;
RHIRenderTarget::ptr windowRenderTarget;

extern FxAtomicString ksDistortionMap;
extern cv::Ptr<cv::aruco::CharucoBoard> charucoBoard();
static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 5, CV_64FC1);

RHIRenderPipeline::ptr camNdcQuadPipeline;

static FxAtomicString ksDisparityScaleUniformBlock("DisparityScaleUniformBlock");
struct DisparityScaleUniformBlock {
  uint32_t viewportOffsetX;
  uint32_t viewportOffsetY;
  float disparityScale;
  uint32_t sourceLevel;

  uint32_t maxValidDisparityRaw;
  uint32_t colorMapMode;
  float pad3;
  float pad4;
};

static FxAtomicString ksColorMapUniformBlock("ColorMapUniformBlock");
struct ColorMapUniformBlock {
  float displayRangeMin;
  float displayRangeMax;
  uint32_t sourceLevel;
  uint32_t colorMapMode;
};

static FxAtomicString ksNDCQuadUniformBlock("NDCQuadUniformBlock");
struct NDCQuadUniformBlock {
  glm::mat4 modelViewProjection;
};

// ImTextureID is a raw RHISurface* (matches the hmdcam ImGui backend's
// convention; see ImGui_ImplFxRHI_Init's setTexID call).
void ImGui_Image(RHISurface::ptr img, const ImVec2& uv0 = ImVec2(0, 0), const ImVec2& uv1 = ImVec2(1, 1)) {
  if (img)
    ImGui::Image((ImTextureID) img.get(), ImVec2(img->width(), img->height()), uv0, uv1);
  else
    ImGui::Text("<null image>");
}

template <typename T>
static std::vector<T> flattenVector(const std::vector<std::vector<T>>& in) {
  std::vector<T> res;
  size_t s = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    s += in[i].size();
  }
  res.reserve(s);
  for (size_t i = 0; i < in.size(); ++i) {
    for (size_t j = 0; j < in[i].size(); ++j) {
      res.push_back(in[i][j]);
    }
  }
  return res;
}

void drawDisparityImageCursorOverlay(glm::vec2 normalizedPoint) {
  ImVec2 rectMin = ImGui::GetItemRectMin();
  ImVec2 rectMax = ImGui::GetItemRectMax();
  ImDrawList* drawList = ImGui::GetWindowDrawList();

  int px = static_cast<int>(normalizedPoint.x * (static_cast<float>(rectMax.x - rectMin.x))) + rectMin.x;
  int py = static_cast<int>(normalizedPoint.y * (static_cast<float>(rectMax.y - rectMin.y))) + rectMin.y;
  int border = 4;

  ImU32 green = IM_COL32(0, 255, 0, 255);

  drawList->AddLine(ImVec2(rectMin.x, py), ImVec2(px - border, py), green);
  drawList->AddLine(ImVec2(px + border, py), ImVec2(rectMax.x, py), green);

  drawList->AddLine(ImVec2(px, rectMin.y), ImVec2(px, py - border), green);
  drawList->AddLine(ImVec2(px, py + border), ImVec2(px, rectMax.y), green);

  drawList->AddRect(ImVec2(px - border, py - border), ImVec2(px + border, py + border), green);
}

void drawDisparityImageCursorOverlayWithOffset(glm::vec2 normalizedPoint, float normalizedOffset) {
  ImVec2 rectMin = ImGui::GetItemRectMin();
  ImVec2 rectMax = ImGui::GetItemRectMax();
  ImDrawList* drawList = ImGui::GetWindowDrawList();

  int px = static_cast<int>(normalizedPoint.x * (static_cast<float>(rectMax.x - rectMin.x))) + rectMin.x;
  int pxO = static_cast<int>((normalizedPoint.x + normalizedOffset) * (static_cast<float>(rectMax.x - rectMin.x))) + rectMin.x;

  int pxL = px, pxR = pxO;
  if (pxR < pxL) {
    std::swap(pxL, pxR);
  }
  int py = static_cast<int>(normalizedPoint.y * (static_cast<float>(rectMax.y - rectMin.y))) + rectMin.y;
  int border = 4;

  ImU32 green = IM_COL32(0, 255, 0, 255);
  ImU32 magenta = IM_COL32(255, 0, 255, 255);

  // L-R lines
  drawList->AddLine(ImVec2(rectMin.x, py), ImVec2(pxL - border, py), green);
  drawList->AddLine(ImVec2(pxL + border, py), ImVec2(pxR - border, py), magenta);
  drawList->AddLine(ImVec2(pxR + border, py), ImVec2(rectMax.x, py), green);

  // U-D lines
  drawList->AddLine(ImVec2(px, rectMin.y), ImVec2(px, py - border), green);
  drawList->AddLine(ImVec2(px, py + border), ImVec2(px, rectMax.y), green);
  drawList->AddLine(ImVec2(pxO, rectMin.y), ImVec2(pxO, py - border), magenta);
  drawList->AddLine(ImVec2(pxO, py + border), ImVec2(pxO, rectMax.y), magenta);

  // frames
  drawList->AddRect(ImVec2(px - border, py - border), ImVec2(px + border, py + border), green);
  drawList->AddRect(ImVec2(pxO - border, py - border), ImVec2(pxO + border, py + border), magenta);
}

bool updateHoverPositionForLastItem(glm::vec2& hoverPositionNormalized) {
  ImVec2 rectMin = ImGui::GetItemRectMin();
  ImVec2 rectMax = ImGui::GetItemRectMax();
  if (!ImGui::IsMouseHoveringRect(rectMin, rectMax))
    return false;

  ImVec2 mp = ImGui::GetMousePos();
  hoverPositionNormalized = glm::vec2(
    glm::clamp((mp.x - rectMin.x) / (rectMax.x - rectMin.x), 0.0f, 1.0f),
    glm::clamp((mp.y - rectMin.y) / (rectMax.y - rectMin.y), 0.0f, 1.0f));
  return true;
}

int triangulationDisparityScaleInv = 16; // TODO remove
std::vector<glm::vec3> getTriangulatedPointsForView(size_t viewIdx, const std::vector<std::vector<cv::Point2f>>& leftCalibrationPoints, const std::vector<std::vector<cv::Point2f>>& rightCalibrationPoints) {
  std::vector<glm::vec3> res;

  auto lp = flattenVector(leftCalibrationPoints);
  auto rp = flattenVector(rightCalibrationPoints);
  assert(lp.size() == rp.size());

  if (lp.empty())
    return res;

  res.resize(lp.size());

  // Input points (m_calibState->m_calibrationPoints) have been captured in distortion-corrected, stereo-remapped space

  CameraSystem::View& view = cameraSystem->viewAtIndex(viewIdx);
  glm::mat3 R1 = glmMat3FromCVMatrix(view.stereoRectification[0]);
  glm::vec4 depthParameters = view.depthParameters();

  float dispScale = 16.0f / static_cast<float>(triangulationDisparityScaleInv);

  bool isVerticalStereo = view.isVerticalStereo();

  for (size_t pointIdx = 0; pointIdx < lp.size(); ++pointIdx) {
    float x = lp[pointIdx].x;
    float y = lp[pointIdx].y;

    float fDisp;
    if (isVerticalStereo)
      fDisp = fabs(rp[pointIdx].y - lp[pointIdx].y) / dispScale;
    else
      fDisp = fabs(rp[pointIdx].x - lp[pointIdx].x) / dispScale;

    glm::vec3 pp = glm::vec3(x + depthParameters.x, y + depthParameters.y, depthParameters.z);
    float lw = depthParameters.w * fDisp;

    res[pointIdx] = R1 * (pp / lw);
  }
  return res;
}


// Main code
int main(int argc, char** argv) {

  const char* debugHost = NULL;
  DepthMapGeneratorBackend depthBackend = platformDefaultDepthBackend();

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth-backend")) {
      if (i == (argc - 1)) {
        printf("--depth-backend: requires argument\n");
        return 1;
      }
      depthBackend = depthBackendStringToEnum(argv[++i]);
    } else {
      debugHost = argv[i];
    }
  }

  if (!debugHost) {
    printf("usage: %s [--depth-backend none|dgpu|depthai] hostname\n", argv[0]);
    return -1;
  }

  FxThreading::detail::init();

  RHICUDA::initRHICUDA(); // Required before depthMapGenerator

  depthMapGenerator = createDepthMapGenerator(depthBackend);

  // SDL + window init
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    printf("Error: %s\n", SDL_GetError());
    return -1;
  }

  // The Vulkan loader is opened on demand; ask SDL to do that now so
  // SDL_Vulkan_GetInstanceExtensions has something to inspect.
  if (SDL_Vulkan_LoadLibrary(nullptr) != 0) {
    printf("SDL_Vulkan_LoadLibrary failed: %s\n", SDL_GetError());
    return -1;
  }

  SDL_WindowFlags window_flags = (SDL_WindowFlags) (SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window = SDL_CreateWindow("debug-client", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920, 1080, window_flags);
  if (!window) {
    printf("SDL_CreateWindow failed: %s\n", SDL_GetError());
    return -1;
  }

  // RHI/VK bringup. RHIVulkan needs the host-platform surface extension
  // (e.g. VK_KHR_xlib_surface) in its instance-extension list, which is
  // platform-dependent — SDL knows which one.
  std::vector<const char*> sdlInstanceExtensions;
  if (!RenderBackendSDLVK::getRequiredInstanceExtensions(window, sdlInstanceExtensions)) {
    return -1;
  }
  initRHIVulkan(sdlInstanceExtensions);

  renderBackend = new RenderBackendSDLVK(window);
  renderBackend->createPresentation();
  windowRenderTarget = renderBackend->windowRenderTarget();
  static_cast<RHIVK*>(rhi())->setFrameSource(renderBackend);

  sceneCamera = new FxCamera();

  RHISurface::ptr disparityScaleSurface;
  RHIRenderTarget::ptr disparityScaleTarget;
  RHIRenderPipeline::ptr disparityScalePipeline = rhi()->compileRenderPipeline("shaders/ndcQuad.vtx.glsl", "shaders/disparityScale.frag.glsl", fullscreenPassVertexLayout, kPrimitiveTopologyTriangleStrip);

  RHIRenderPipeline::ptr colorMapPipeline = rhi()->compileRenderPipeline("shaders/ndcQuad.vtx.glsl", "shaders/colorMap.frag.glsl", fullscreenPassVertexLayout, kPrimitiveTopologyTriangleStrip);

  RHISurface::ptr disparityDebugColorMapSurface;
  RHIRenderTarget::ptr disparityDebugColorMapTarget;

  RHISurface::ptr confidenceColorMapSurface;
  RHIRenderTarget::ptr confidenceColorMapTarget;

  // Debug server connection
  cameraProvider = new DebugCameraProvider();

  if (!cameraProvider->connect(debugHost)) {
    printf("Failed to connect to debug server (%s)\n", debugHost);
    return -1;
  }

  cameraSystem = new CameraSystem(cameraProvider);

  {
    // Load remote config
    cv::FileStorage fs(cameraProvider->cameraSystemConfig(), cv::FileStorage::MEMORY | cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    cameraSystem->loadCalibrationData(fs);
  }

  std::vector<CharucoMultiViewCalibration*> charucoProcessors;
  charucoProcessors.resize(cameraSystem->views());
  bool enableCharucoDetection = false; // default-off for interaction performance

  // CV processing init
  bool usingRemoteDisparity = false;
  if (!depthMapGenerator) {
    depthMapGenerator = cameraProvider; // Using disparity streamed from the remote system
    usingRemoteDisparity = true;
  }

  depthMapGenerator->setDebugDisparityCPUAccessEnabled(true);
  depthMapGenerator->initWithCameraSystem(cameraSystem);
  depthMapGenerator->loadSettings();
  depthMapGenerator->setPopulateDebugTextures(true);

  // Setup render pipelines for camera sampling
  {
    RHIShaderDescriptor desc("shaders/ndcQuad.vtx.glsl", "shaders/camTexturedQuad.frag.glsl", ndcQuadVertexLayout);
    camNdcQuadPipeline = rhi()->compileRenderPipeline(desc, kPrimitiveTopologyTriangleStrip);
  }

  std::vector<RHIRect> debugSurfaceCameraRects;
  uint32_t debugSurfaceWidth, debugSurfaceHeight;
  {
    // Size and allocate debug surface area based on camera count
    unsigned int debugColumns = 1, debugRows = 1;
    if (cameraProvider->streamCount() > 1) {
      debugColumns = 2;
      debugRows = (cameraProvider->streamCount() + 1) / 2; // round up
    }
    debugSurfaceWidth = debugColumns * cameraProvider->streamWidth();
    debugSurfaceHeight = debugRows * cameraProvider->streamHeight();

    for (size_t cameraIdx = 0; cameraIdx < cameraProvider->streamCount(); ++cameraIdx) {
      unsigned int col = cameraIdx % debugColumns;
      unsigned int row = cameraIdx / debugColumns;
      RHIRect r = RHIRect::xywh(col * cameraProvider->streamWidth(), row * cameraProvider->streamHeight(), cameraProvider->streamWidth(), cameraProvider->streamHeight());
      debugSurfaceCameraRects.push_back(r);
    }
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void) io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplSDL2_InitForVulkan(window);
  ImGui_ImplFxRHI_Init();

  // Our state
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  bool surfaceUpdateEnabled = true;
  bool render2dMode = false;

  // Main loop
  bool done = false;
  while (!done) {

    bool wantSwapchainRecreate = false;

    // Poll and handle events (inputs, window resize, etc.)
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);
      if (event.type == SDL_QUIT)
        done = true;
      if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
        done = true;
      if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED && event.window.windowID == SDL_GetWindowID(window))
        wantSwapchainRecreate = true;

      if (io.WantCaptureKeyboard || io.WantCaptureMouse) {
        // Try to avoid getting stuck in relative mouse mode
        SDL_SetRelativeMouseMode(SDL_FALSE);
      }

      if (!io.WantCaptureKeyboard) {
        if (event.type == SDL_KEYUP && event.key.keysym.scancode == SDL_SCANCODE_LALT) {
          // Stop relative mouse mode on camera-drag release
          SDL_SetRelativeMouseMode(SDL_FALSE);
        }
      }

      if (!io.WantCaptureMouse) {
        if (event.type == SDL_MOUSEMOTION) {
          const uint8_t* keyState = SDL_GetKeyboardState(NULL);
          if (keyState[SDL_SCANCODE_LALT]) {
            // Enter relative mouse mode on first camera-drag motion
            SDL_SetRelativeMouseMode(SDL_TRUE);
            if (event.motion.state & SDL_BUTTON_LMASK) {
              sceneCamera->tumble(glm::vec2(event.motion.xrel, event.motion.yrel));
            } else if (event.motion.state & SDL_BUTTON_RMASK) {
              sceneCamera->dolly(static_cast<float>(event.motion.xrel) * 0.001f);
            } else if (event.motion.state & SDL_BUTTON_MMASK) {
              sceneCamera->track(glm::vec2(-static_cast<float>(event.motion.xrel) / static_cast<float>(io.DisplaySize.x), static_cast<float>(event.motion.yrel) / static_cast<float>(io.DisplaySize.y)));
            }
          } else if (event.motion.state & SDL_BUTTON_RMASK) {
            sceneCamera->spin(glm::vec2(event.motion.xrel, event.motion.yrel));
          }
        }
      }
    }

    if (wantSwapchainRecreate) {
      renderBackend->recreateSwapchain();
    }

    // Start the Dear ImGui frame
    ImGui_ImplFxRHI_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    {
      ImGui::Begin("Debug-Client");

      ImGui::Checkbox("2D Render Mode", &render2dMode);

      ImGui::BeginDisabled(render2dMode);

      { // Camera controls disabled in 2d mode
        glm::vec3 p = sceneCamera->position();
        glm::vec3 t = sceneCamera->targetPosition();
        if (ImGui::InputFloat3("Camera Position", &p[0]))
          sceneCamera->setPosition(p);

        if (ImGui::InputFloat3("Camera Target", &t[0]))
          sceneCamera->setTargetPosition(t);

        if (ImGui::Button("Reset Camera")) {
          sceneCamera->setPosition(glm::vec3(0, 0, 0));
          sceneCamera->setTargetPosition(glm::vec3(0, 0, -5));
        }

        float fov = sceneCamera->fieldOfView();
        if (ImGui::DragFloat("Camera Horizontal FoV", &fov, /*speed=*/ 1.0f, /*min=*/ 20.0f, /*max=*/ 170.0f, /*format=*/ "%.1fdeg")) {
          sceneCamera->setFieldOfView(fov);
        }

        ImGui::Checkbox("Charuco detection", &enableCharucoDetection);
        ImGui::SliderInt("Charuco Disp Scale", &triangulationDisparityScaleInv, 1, 256); // TODO remove
      }
      ImGui::EndDisabled();

      ImGui::Separator();

      for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
        CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
        if (!v.isStereo)
          continue;

        ImGui::PushID(viewIdx);

        ImGui::Text("View %zu", viewIdx);

        glm::vec3 stereoTxMM = glmVec3FromCV(v.stereoTranslation) * 1000.0f;
        if (ImGui::DragFloat3("Stereo Baseline Tx", &stereoTxMM[0], /*speed=*/ 0.1f, /*min=*/ -1000.0f, /*max=*/ 1000.0f, "%.1fmm")) {
          v.stereoTranslation = cvVec3FromGlm(stereoTxMM * 0.001f);
        }

        static float baselineScale = 1.0f;
        ImGui::DragFloat("Baseline Scale", &baselineScale, /*speed=*/ 0.005f, /*min=*/ 0.5f, /*max=*/ 1.5f);
        ImGui::Text("Baseline length: unscaled %.1fmm, scaled %.1fmm",
          glm::length(stereoTxMM), glm::length(stereoTxMM) * baselineScale);

        if (ImGui::Button("Apply Scale / Recompute stereo parameters")) {
          v.stereoTranslation *= baselineScale;
          baselineScale = 1.0f;
          cameraSystem->updateViewStereoDistortionParameters(viewIdx);
        }


        glm::vec3 txMM = v.viewTranslation * 1000.0f;
        if (ImGui::DragFloat3("Tx", &txMM[0], /*speed=*/ 0.1f, /*min=*/ -1000.0f, /*max=*/ 1000.0f, "%.1fmm")) {
          v.viewTranslation = txMM * 0.001f;
        }
        ImGui::DragFloat3("Rx", &v.viewRotation[0], /*speed=*/ 0.1f, /*min=*/ -75.0f, /*max=*/ 75.0f, "%.3fdeg");


        ImGui::Separator();

        ImGui::PopID();
      }

      if (ImGui::Button("Save Calibration and Settings")) {
        cameraSystem->saveCalibrationData();
        depthMapGenerator->saveSettings();
      }

      if (ImGui::Button("Reload Calibration and Settings from Disk")) {
        cameraSystem->loadCalibrationData();
        depthMapGenerator->loadSettings();
      }

      if (usingRemoteDisparity) {
        depthMapGenerator->renderIMGUI();
      } else {
        depthMapGenerator->renderIMGUI();
        depthMapGenerator->renderIMGUIPerformanceGraphs();
      }

      ImGui::ColorEdit3("clear color", (float*) &clear_color); // Edit 3 floats representing a color

      ImGui::Separator();

      ImGui::Checkbox("Enable Surface Updates", &surfaceUpdateEnabled);


      if (ImGui::Button("Capture frame")) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);

        for (size_t streamIdx = 0; streamIdx < cameraProvider->streamCount(); ++streamIdx) {
          char* filenameBuf = new char[64];
          snprintf(filenameBuf, 64, "camera%zu_%lu.png", streamIdx, ts.tv_sec);
          cv::Mat lumaCopy = cameraProvider->cvMatLuma(streamIdx).clone();

          // Async PNG compression since it's expensive
          FxThreading::runFunction([filenameBuf, lumaCopy]() {
            stbi_write_png(filenameBuf, /*x=*/ lumaCopy.cols, /*y=*/ lumaCopy.rows, /*components=*/ lumaCopy.channels(), /*data=*/ lumaCopy.ptr(), /*rowBytes=*/ lumaCopy.step);
            delete[] filenameBuf;
          });
        }
      }


      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();
    }

#if 0
    // distortion test
    if (ImGui::Begin("Distortion test", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
      static RHISurface::ptr testSrf;
      static RHIRenderTarget::ptr testRT;
      static bool s_CPU = false;
      static bool s_CUDA = false;
      static cv::cuda::GpuMat gpuDst;
      static cv::cuda::GpuMat gpuUndistortRectifyMap;
      static cv::Mat remappedGrey, remappedRGBA;

      ImGui::Checkbox("CPU", &s_CPU);
      ImGui::SameLine();
      ImGui::Checkbox("CUDA", &s_CUDA);

      if (!testSrf) {
        testSrf = rhi()->newTexture2D(cameraProvider->streamWidth(), cameraProvider->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
        testRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({testSrf}));
      }

      CameraSystem::View& v = cameraSystem->viewAtIndex(0);
      CameraSystem::Camera& c = cameraSystem->cameraAtIndex(v.cameraIndices[0]);
      cv::Size imageSize = cv::Size(cameraProvider->streamWidth(), cameraProvider->streamHeight());

      if (s_CPU) {
        cv::Mat map1, map2;

        cv::initUndistortRectifyMap(c.intrinsicMatrix, c.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, CV_32F, map1, map2);

        cv::remap(cameraProvider->cvMatLuma(v.cameraIndices[0]), remappedGrey, map1, map2, cv::INTER_LINEAR);
        cv::cvtColor(remappedGrey, remappedRGBA, cv::COLOR_GRAY2RGBA, 4);
        rhi()->loadTextureData(testSrf, kVertexElementTypeUByte4N, remappedRGBA.data);

      } else if (s_CUDA) {
        if (gpuUndistortRectifyMap.empty())
          gpuUndistortRectifyMap = remapArray_initUndistortRectifyMap(c.intrinsicMatrix, c.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, /*downsampleFactor=*/ 1);

        remappedGrey.create(imageSize, CV_8UC4);

        remapArray(cameraProvider->cudaLumaTexObject(v.cameraIndices[0]), imageSize, gpuUndistortRectifyMap, gpuDst, cuStream, /*downsampleFactor=*/ 1);
        gpuDst.download(remappedGrey, cvCudaStream);
        cvCudaStream.waitForCompletion();

        cv::cvtColor(remappedGrey, remappedRGBA, cv::COLOR_GRAY2RGBA, 4);
        rhi()->loadTextureData(testSrf, kVertexElementTypeUByte4N, remappedRGBA.data);

      } else {
        // guts of captureGreyscale
        rhi()->beginRenderPass(testRT, kLoadInvalidate);
        rhi()->bindRenderPipeline(cameraSystem->camGreyscaleUndistortPipeline());
        rhi()->loadTexture(ksDistortionMap, v.stereoDistortionMap[0], linearClampSampler);
        rhi()->loadTexture(ksImageTex, cameraProvider->rgbTexture(v.cameraIndices[0]), linearClampSampler);
        rhi()->drawNDCQuad();
        rhi()->endRenderPass(testRT);
      }
      ImGui_Image(testSrf);
    }
    ImGui::End();
#endif

    for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
      CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
      char windowName[64];
      sprintf(windowName, "View %zu", viewIdx);
      ImGui::Begin(windowName, NULL, ImGuiWindowFlags_AlwaysAutoResize);

      for (size_t viewCameraIdx = 0; viewCameraIdx < v.cameraCount(); ++viewCameraIdx) {
        size_t cameraIdx = v.cameraIndices[viewCameraIdx];
        // TODO apply distortion correction here
        ImGui_Image(cameraProvider->rgbTexture(cameraIdx));
      }
      ImGui::End();
    }

    {
      ImGui::Begin("Internals");
      static int internalsTargetView = 0;
      ImGui::SliderInt("Target View", &internalsTargetView, 0, cameraSystem->views() - 1);

      CameraSystem::View& v = cameraSystem->viewAtIndex(internalsTargetView);
      RHISurface::ptr disparitySurface;
      if (v.isStereo)
        disparitySurface = depthMapGenerator->disparitySurface(internalsTargetView);

      if (disparitySurface) {
        ImGui::PushID("DisparitySurface");

        static int disparityScaleSourceLevel = 0;
        static int disparityScaleColorMapMode = 0;

        float debugDisparityScale = depthMapGenerator->debugDisparityScale();
        if (ImGui::SliderFloat("Debug Disparity Scale", &debugDisparityScale, 0.0f, 2.0f)) {
          depthMapGenerator->setDebugDisparityScale(debugDisparityScale);
        }

        ImGui::SliderInt("Source Level", &disparityScaleSourceLevel, 0, disparitySurface->mipLevels() - 1);
        ImGui::SliderInt("Color Map Mode", &disparityScaleColorMapMode, 0, 1);

        {
          if (!disparityScaleTarget) {
            disparityScaleSurface = rhi()->newTexture2D(disparitySurface->width(), disparitySurface->height(), kSurfaceFormat_RGBA8);
            disparityScaleTarget = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({disparityScaleSurface}));
          }

          rhi()->beginRenderPass(disparityScaleTarget, kLoadInvalidate);
          rhi()->bindRenderPipeline(disparityScalePipeline);

          rhi()->loadTexture(ksImageTex, disparitySurface);
          DisparityScaleUniformBlock ub;
          ub.viewportOffsetX = 0;
          ub.viewportOffsetY = 0;
          ub.disparityScale = depthMapGenerator->debugDisparityScale() * (1.0f / static_cast<float>(depthMapGenerator->maxDisparityRaw()));
          ub.sourceLevel = disparityScaleSourceLevel;
          ub.maxValidDisparityRaw = depthMapGenerator->maxDisparityRaw();
          ub.colorMapMode = disparityScaleColorMapMode;
          rhi()->loadUniformBlockImmediate(ksDisparityScaleUniformBlock, &ub, sizeof(ub));
          rhi()->drawFullscreenPass();
          rhi()->endRenderPass(disparityScaleTarget);
        }

        // normalized (UV) coordinates of mouseover of any of the disparity views
        static glm::vec2 disparityHoverUV = glm::vec2(0.0f, 0.0f);
        ImGui_Image(disparityScaleSurface);
        bool hoverLeft = updateHoverPositionForLastItem(disparityHoverUV);
        drawDisparityImageCursorOverlay(disparityHoverUV);

        ImGui::PopID(); // DisparitySurface

        // Confidence output
        RHISurface::ptr confidenceTexture = depthMapGenerator->confidenceSurface(internalsTargetView);
        if (confidenceTexture) {
          ImGui::PushID("ConfidenceTexture");
          if (!confidenceColorMapTarget) {
            confidenceColorMapSurface = rhi()->newTexture2D(confidenceTexture->width(), confidenceTexture->height(), kSurfaceFormat_RGBA8);
            confidenceColorMapTarget = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({confidenceColorMapSurface}));
          }

          static int confidenceColorMapMode = 0;
          ImGui::SliderInt("Color Map Mode", &confidenceColorMapMode, 0, 1);

          rhi()->beginRenderPass(confidenceColorMapTarget, kLoadInvalidate);
          rhi()->bindRenderPipeline(colorMapPipeline);

          rhi()->loadTexture(ksImageTex, confidenceTexture);
          ColorMapUniformBlock ub;
          ub.displayRangeMin = 0.0f;
          ub.displayRangeMax = 1.0f;
          ub.sourceLevel = 0; // No mips on this source
          ub.colorMapMode = confidenceColorMapMode;
          rhi()->loadUniformBlockImmediate(ksColorMapUniformBlock, &ub, sizeof(ub));
          rhi()->drawFullscreenPass();
          rhi()->endRenderPass(confidenceColorMapTarget);

          ImGui_Image(confidenceColorMapSurface);
          hoverLeft |= updateHoverPositionForLastItem(disparityHoverUV);
          drawDisparityImageCursorOverlay(disparityHoverUV);
          ImGui::PopID();
        }

        // Debug output
        RHISurface::ptr debugResidual = depthMapGenerator->debugResidualSurface(internalsTargetView);
        if (debugResidual) {
          ImGui::PushID("DebugResidual");
          if (!disparityDebugColorMapTarget) {
            disparityDebugColorMapSurface = rhi()->newTexture2D(debugResidual->width(), debugResidual->height(), kSurfaceFormat_RGBA8);
            disparityDebugColorMapTarget = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({disparityDebugColorMapSurface}));
          }

          static float debugMinValue = 0.0f, debugMaxValue = 1.0f;
          static int debugColorMapMode = 0;

          ImGui::SliderFloat("Display Range Min", &debugMinValue, 0.0f, 1.0f);
          ImGui::SliderFloat("Display Range Max", &debugMaxValue, 0.0f, 1.0f);
          if (debugMinValue > debugMaxValue) {
            // Enforce ordering
            std::swap(debugMinValue, debugMaxValue);
          }
          ImGui::SliderInt("Color Map Mode", &debugColorMapMode, 0, 6);

          rhi()->beginRenderPass(disparityDebugColorMapTarget, kLoadInvalidate);
          rhi()->bindRenderPipeline(colorMapPipeline);

          rhi()->loadTexture(ksImageTex, debugResidual);
          ColorMapUniformBlock ub;
          ub.displayRangeMin = debugMinValue;
          ub.displayRangeMax = debugMaxValue;
          ub.sourceLevel = 0; // No mips on this source
          ub.colorMapMode = debugColorMapMode;
          rhi()->loadUniformBlockImmediate(ksColorMapUniformBlock, &ub, sizeof(ub));
          rhi()->drawFullscreenPass();
          rhi()->endRenderPass(disparityDebugColorMapTarget);

          ImGui_Image(disparityDebugColorMapSurface);
          hoverLeft |= updateHoverPositionForLastItem(disparityHoverUV);
          drawDisparityImageCursorOverlay(disparityHoverUV);
          ImGui::PopID();
        }

        float disparitySample = depthMapGenerator->debugPeekDisparityUV(internalsTargetView, disparityHoverUV);
        float disparitySampleNormalized = disparitySample / static_cast<float>(disparitySurface->width());
        float confidenceSample = depthMapGenerator->debugPeekConfidenceUV(internalsTargetView, disparityHoverUV);
        uint8_t residualSample = depthMapGenerator->debugPeekResidualUV(internalsTargetView, disparityHoverUV);

        ImGui_Image(depthMapGenerator->leftGrayscale(internalsTargetView));
        hoverLeft |= updateHoverPositionForLastItem(disparityHoverUV);
        if (hoverLeft) {
          drawDisparityImageCursorOverlay(disparityHoverUV);
        } else {
          drawDisparityImageCursorOverlayWithOffset(disparityHoverUV, disparitySampleNormalized);
        }

        ImGui_Image(depthMapGenerator->rightGrayscale(internalsTargetView));
        if (updateHoverPositionForLastItem(disparityHoverUV)) {
          drawDisparityImageCursorOverlay(disparityHoverUV);
        } else {
          drawDisparityImageCursorOverlayWithOffset(disparityHoverUV, -disparitySampleNormalized);
        }

        glm::vec3 localP = depthMapGenerator->debugPeekLocalPositionUV(internalsTargetView, disparityHoverUV) * 1000.0f;
        ImGui::Text("Hover UV: {%.2f, %.2f} (%d, %d)\nDisparity: %.3f\nConfidence: %.3f\nResidual: %u (%.3f)\n, Local P: %.3fmm, %.3fmm, %.3fmm",
          disparityHoverUV.x, disparityHoverUV.y,
          static_cast<int>(disparityHoverUV.x * static_cast<float>(disparityScaleSurface->width())),
          static_cast<int>(disparityHoverUV.y * static_cast<float>(disparityScaleSurface->height())),
          disparitySample,
          confidenceSample,
          residualSample, static_cast<float>(residualSample) / 255.0f,
          localP.x, localP.y, localP.z);

        static int sampleDisp = 1;
        ImGui::SliderInt("Test Disp", &sampleDisp, 1, depthMapGenerator->maxDisparityPixels());
        ImGui::Text("Sample Disp Depth: %.3fmm\n", 1000.0f * depthMapGenerator->debugComputeDepthForDisparity(internalsTargetView, (float) sampleDisp));
      }


      ImGui::End();
    }

    // Service debug server connection
    if (surfaceUpdateEnabled)
      cameraProvider->updateSurfaces();

    if (enableCharucoDetection) {
      for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
        CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);

        // Create the processor if it doesn't already exist
        if (!charucoProcessors[viewIdx]) {
          if (v.isStereo) {
            charucoProcessors[viewIdx] = new CharucoMultiViewCalibration(cameraSystem, {v.cameraIndices[0], v.cameraIndices[1]}, {viewIdx, viewIdx});
          } else {
            charucoProcessors[viewIdx] = new CharucoMultiViewCalibration(cameraSystem, {v.cameraIndices[0]});
          }
          charucoProcessors[viewIdx]->m_enableFeedbackView = false; // don't need the 2d feedback rendering
        }

        // Clear previous capture results from the processors
        charucoProcessors[viewIdx]->reset();

        // Run process
        charucoProcessors[viewIdx]->processFrame(/*requestCapture=*/ true);
      }
    }

    depthMapGenerator->processFrame();

    // CUDA-to-RHI handoff after depthMapGenerator finishes
    rhi()->signalCUDAToRHI(RHICUDA::defaultAsyncStream);

    // Rendering
    FxRenderView renderView = sceneCamera->toRenderView(static_cast<float>(io.DisplaySize.x) / static_cast<float>(io.DisplaySize.y));


    // Note that our camera uses reversed depth projection -- we clear to 0 and use a "greater" depth-test.
    rhi()->setClearColor(glm::vec4(clear_color.x, clear_color.y, clear_color.z, clear_color.w));
    rhi()->setClearDepth(0.0f);
    rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
    rhi()->bindDepthStencilState(standardGreaterDepthStencilState);

    if (render2dMode) {
      // Render 2d views
      // Figure out how to fit the debug surface into the current window size
      float debugSurfaceAspect = static_cast<float>(debugSurfaceWidth) / static_cast<float>(debugSurfaceHeight);
      float windowAspect = static_cast<float>(io.DisplaySize.x) / static_cast<float>(io.DisplaySize.y);

      int padX = 0, padY = 0;
      float scale = 0;

      if (debugSurfaceAspect < windowAspect) {
        // Window is wider, pad debug surface in X
        scale = static_cast<float>(io.DisplaySize.y) / static_cast<float>(debugSurfaceHeight);
        padX = io.DisplaySize.x - static_cast<int>(scale * static_cast<float>(debugSurfaceWidth));
        padX = std::max<int>(padX, 0);
      } else {
        // Window is taller, pad debug surface in Y
        scale = static_cast<float>(io.DisplaySize.x) / static_cast<float>(debugSurfaceWidth);
        padY = io.DisplaySize.y - static_cast<int>(scale * static_cast<float>(debugSurfaceHeight));
        padY = std::max<int>(padY, 0);
      }

      rhi()->bindRenderPipeline(camNdcQuadPipeline);

      for (size_t cameraIdx = 0; cameraIdx < cameraProvider->streamCount(); ++cameraIdx) {
        RHIRect origVp = debugSurfaceCameraRects[cameraIdx];
        RHIRect scaledVp = RHIRect::xywh(
          static_cast<float>(origVp.x * scale) + (padX / 2),
          static_cast<float>(origVp.y * scale) + (padY / 2),
          static_cast<float>(origVp.width) * scale,
          static_cast<float>(origVp.height) * scale);

        rhi()->setViewport(scaledVp);

        // No-distortion / direct passthrough
        rhi()->loadTexture(ksImageTex, cameraProvider->rgbTexture(cameraIdx), linearClampSampler);
        rhi()->drawNDCQuad();
      }

    } else {
      // Render 3d views
      for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
        CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
        if (v.isStereo)
          depthMapGenerator->renderDisparityDepthMap(viewIdx, renderView);
      }
    }

    if ((render2dMode == false) && enableCharucoDetection) {
      std::vector<glm::vec4> pointsStaging;
      ImGui::Begin("ChAruCo");

      static bool gizmoDepthTest = true;
      ImGui::Checkbox("Depth-test gizmos", &gizmoDepthTest);

      for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
        if (viewIdx != 0)
          ImGui::Separator();
        ImGui::Text("View %zu", viewIdx);

        CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
        CharucoMultiViewCalibration* proc = charucoProcessors[viewIdx];
        if (!proc)
          continue;

        if (proc->m_calibrationPoints.empty() || proc->m_objectPoints.empty() || proc->m_objectPoints[0].empty())
          continue; // nothing detected this frame

        if (v.isStereo) {
          std::vector<cv::Point2f> lp = flattenVector(proc->m_calibrationPoints[0]);
          std::vector<cv::Point2f> rp = flattenVector(proc->m_calibrationPoints[1]);
          std::vector<int> ids = flattenVector(proc->m_objectIds);

          std::vector<glm::vec3> triangulatedPoints = getTriangulatedPointsForView(viewIdx, proc->m_calibrationPoints[0], proc->m_calibrationPoints[1]);

          std::vector<glm::vec3> boardObjectSpacePoints;
          // build object-space point vector of charuco corners
          for (const cv::Point3f& p : charucoBoard()->getChessboardCorners()) {
            boardObjectSpacePoints.push_back(glmVec3FromCV(p));
          }

          char idBuf[32];
          sprintf(idBuf, "charuco%zu_disp", viewIdx);
          if (ImGui::BeginChild(idBuf, ImVec2(0, 256), /*border=*/ false)) {
            for (size_t pointIdx = 0; pointIdx < triangulatedPoints.size(); ++pointIdx) {
              const glm::vec3& p = triangulatedPoints[pointIdx];
              ImGui::Text("[%.2u] %.4f %.4f || %.4f %.4f || %.4f %.4f %.4f",
                ids[pointIdx],
                lp[pointIdx].x, lp[pointIdx].y,
                rp[pointIdx].x, rp[pointIdx].y,
                p[0], p[1], p[2]);
            }
          }
          ImGui::EndChild();


          { // triangulated points
            size_t offset = pointsStaging.size();
            pointsStaging.resize(pointsStaging.size() + triangulatedPoints.size());
            glm::mat4 viewXf = cameraSystem->viewWorldTransform(viewIdx);

            for (int pointIdx = 0; pointIdx < triangulatedPoints.size(); ++pointIdx) {
              pointsStaging[offset + pointIdx] = viewXf * glm::vec4(triangulatedPoints[pointIdx], 1.0f);
            }
          }

        } else {
          // TODO do something useful with markers detected in 2d views
        }
      }

      if (pointsStaging.size()) {
        RHIBuffer::ptr pointsBuf = rhi()->newBufferWithContents(pointsStaging.data(), pointsStaging.size() * sizeof(float) * 4);

        rhi()->bindDepthStencilState(gizmoDepthTest ? standardGreaterDepthStencilState : disabledDepthStencilState);
        // render points as locator gizmos
        drawTriadGizmosForPoints(pointsBuf, pointsStaging.size(), renderView.viewProjectionMatrix, /*scale=*/ -0.025f /*25mm*/);
      }

      ImGui::End();
    }


    ImGui::Render();
    ImGui_ImplFxRHI_RenderDrawData(windowRenderTarget, ImGui::GetDrawData());
    rhi()->endRenderPass(windowRenderTarget);

    rhi()->swapBuffers(windowRenderTarget);
  }

  // Cleanup. Release order matters: anything holding RHI handles must drop
  // them before shutdownRHI() destroys the VK device. ImGui_ImplFxRHI_Shutdown
  // releases the static font atlas + pipeline that would otherwise outlive
  // the RHI device into static-destructor time.
  ImGui_ImplFxRHI_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  FxThreading::detail::shutdown();

  windowRenderTarget.reset();
  shutdownRHI();

  delete renderBackend;

  SDL_DestroyWindow(window);
  SDL_Vulkan_UnloadLibrary();
  SDL_Quit();

  // _exit skips static destructors. Several global RHI::ptr / CUDA / NPP /
  // OpenCV singletons race during static teardown after main returns and
  // segfault; the original GL path here had the same workaround. The
  // explicit cleanup above releases what we actually own.
  _exit(0);
}
