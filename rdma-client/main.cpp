#include "imgui_backend.h"
#include "imgui.h"
#include <stdio.h>
#include <SDL.h>
#include <cuda.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <glm/gtx/transform.hpp>

#include "rdma/RDMAContext.h"
#include "rdma/RDMABuffer.h"
#include "RDMACameraProvider.h"
#include "common/CameraSystem.h"
#include "common/CharucoMultiViewCalibration.h"
#include "common/FxCamera.h"
#include "common/FxThreading.h"
#include "common/DepthMapGenerator.h"
#include "common/glmCvInterop.h"

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/RHIWindowRenderTargetGL.h"
#include "rhi/gl/RHISurfaceGL.h"

RHIWindowRenderTargetGL::ptr windowRenderTarget;
class RHISDLWindowRenderTargetGL : public RHIWindowRenderTargetGL {
public:
  RHISDLWindowRenderTargetGL(SDL_Window* w) : m_window(w) {}
  typedef boost::intrusive_ptr<RHISDLWindowRenderTargetGL> ptr;
  virtual ~RHISDLWindowRenderTargetGL() {}
  virtual void platformSwapBuffers() {
    SDL_GL_SwapWindow(m_window);
  }
protected:
  SDL_Window* m_window;
};

CUdevice cudaDevice;
CUcontext cudaContext;
RDMAContext* rdmaContext;
RDMACameraProvider* cameraProvider;
CameraSystem* cameraSystem;
FxCamera* sceneCamera;
RDMABuffer::ptr configBuffer;
DepthMapGenerator* depthMapGenerator;
SHMSegment<DepthMapSHM>* shm;

extern RHIRenderPipeline::ptr camGreyscalePipeline;
extern RHIRenderPipeline::ptr camGreyscaleUndistortPipeline;
extern FxAtomicString ksDistortionMap;
extern cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard;
static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 5, CV_64FC1);

static FxAtomicString ksImageTex("imageTex");

#if 0
RHIRenderPipeline::ptr meshVertexColorPipeline;
RHIBuffer::ptr meshQuadVBO;

FxAtomicString ksMeshTransformUniformBlock("MeshTransformUniformBlock");
struct MeshTransformUniformBlock {
  glm::mat4 modelViewProjection;
};
#endif


static FxAtomicString ksDisparityScaleUniformBlock("DisparityScaleUniformBlock");
struct DisparityScaleUniformBlock {
  float disparityScale;
  uint32_t sourceLevel;
  uint32_t maxValidDisparityRaw;
  float pad4;
};

void rdmaUserEventCallback(RDMAContext*, uint32_t userEventID, SerializationBuffer payload) {
  switch (userEventID) {
    case 1:
      cameraProvider->flagRDMABuffersDirty();
      break;

    default:
      printf("rdmaUserEventCallback: unhandled userEventID %u\n", userEventID);
      break;
  };
}

void ImGui_Image(RHISurface::ptr img, const ImVec2& uv0 = ImVec2(0,0), const ImVec2& uv1 = ImVec2(1,1)) {
  ImGui::Image((ImTextureID) static_cast<uintptr_t>(static_cast<RHISurfaceGL*>(img.get())->glId()), ImVec2(img->width(), img->height()), uv0, uv1);
}

template <typename T> static std::vector<T> flattenVector(const std::vector<std::vector<T> >& in) {
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
extern int triangulationDisparityScaleInv; // TODO remove

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

  ImU32 green =     IM_COL32(  0, 255,   0, 255);
  ImU32 magenta =   IM_COL32(255,   0, 255, 255);

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



// Main code
int main(int argc, char** argv) {

  const char* rdmaHost = NULL;
  DepthMapGeneratorBackend depthBackend = kDepthBackendDGPU;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth-backend")) {
      if (i == (argc - 1)) {
        printf("--depth-backend: requires argument\n");
        return 1;
      }
      depthBackend = depthBackendStringToEnum(argv[++i]);
    } else {
      rdmaHost = argv[i];
    }
  }

  if (!rdmaHost) {
    printf("usage: %s [--depth-backend none|dgpu|depthai] hostname\n", argv[0]);
    return -1;
  }

  depthMapGenerator = createDepthMapGenerator(depthBackend);
  depthMapGenerator->setDebugDisparityCPUAccessEnabled(true);

  FxThreading::detail::init();


  // RDMA context / connection
  rdmaContext = RDMAContext::createClientContext(rdmaHost);
  if (!rdmaContext) {
    printf("RDMA context initialization failed\n");
    return -1;
  }
  rdmaContext->setUserEventCallback(&rdmaUserEventCallback);

  // GL/window init
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    printf("Error: %s\n", SDL_GetError());
    return -1;
  }

  // Decide GL+GLSL versions
  const char* glsl_version = "#version 130";
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

  // Create window with graphics context
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window = SDL_CreateWindow("rdma-client", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920, 1080, window_flags);
  SDL_GLContext gl_context = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(window, gl_context);
  SDL_GL_SetSwapInterval(1); // Enable vsync

  initRHIGL();
  windowRenderTarget = new RHISDLWindowRenderTargetGL(window);
  sceneCamera = new FxCamera();

#if 0
  meshVertexColorPipeline = rhi()->compileRenderPipeline("shaders/meshVertexColor.vtx.glsl", "shaders/meshVertexColor.frag.glsl", RHIVertexLayout({
      RHIVertexLayoutElement(0, kVertexElementTypeFloat3, "position", 0,                 sizeof(float) * 7),
      RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "color",    sizeof(float) * 3, sizeof(float) * 7)
    }), kPrimitiveTopologyTriangleStrip);

  {
    static const float sampleQuadData[] = {
    //   x      y     z     r     g     b     a
       1.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,  // right-top
       1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,  // right-bottom
      -1.0f,  1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // left-top
      -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}; // left-bottom

    meshQuadVBO = rhi()->newBufferWithContents(sampleQuadData, sizeof(float) * 7 * 4);
  }
#endif

  RHISurface::ptr disparityScaleSurface;
  RHIRenderTarget::ptr disparityScaleTarget;
  RHIRenderPipeline::ptr disparityScalePipeline = rhi()->compileRenderPipeline("shaders/lightPass.vtx.glsl", "shaders/disparityScale.frag.glsl", fullscreenPassVertexLayout, kPrimitiveTopologyTriangleStrip);


  // CUDA init
  {
    cuInit(0);

    cuDeviceGet(&cudaDevice, 0);
    char devName[512];
    cuDeviceGetName(devName, 511, cudaDevice);
    devName[511] = '\0';
    printf("CUDA device: %s\n", devName);

    cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice);
    cuCtxSetCurrent(cudaContext);
  }

  // Read RDMA config from server
  RDMABuffer::ptr configBuf = rdmaContext->newManagedBuffer("config", 8192, kRDMABufferUsageWriteDestination); // TODO should be read source / use RDMA read instead of relying on push

  printf("Waiting to read config...");
  fflush(stdout);
  for (size_t configWait = 0;;  ++configWait) {
    printf(".");
    fflush(stdout);

    SerializationBuffer cfg(configBuf->data(), configBuf->size());
    if (cfg.get_u32()) { // cameraCount
      cameraProvider = new RDMACameraProvider(rdmaContext, cfg);

      printf(" Done.\nReceived config: %zu cameras @ %ux%u\n", cameraProvider->streamCount(), cameraProvider->streamWidth(), cameraProvider->streamHeight());
      break;
    }

    if (configWait > 10) {
      printf(" Timed out.\n");
      return -1;
    }

    sleep(1); // TODO messy
  }

  cameraSystem = new CameraSystem(cameraProvider);
  cameraSystem->loadCalibrationData();

  std::vector<CharucoMultiViewCalibration*> charucoProcessors;
  charucoProcessors.resize(cameraSystem->views());
  bool enableCharucoDetection = false; // default-off for interaction performance

  // CV processing init
  if (depthMapGenerator) {
    depthMapGenerator->initWithCameraSystem(cameraSystem);
    depthMapGenerator->loadSettings();
    depthMapGenerator->setPopulateDebugTextures(true);
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  //ImGui::StyleColorsClassic();

  // Setup Platform/Renderer bindings
  ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Our state
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  // Main loop
  bool done = false;
  while (!done) {

      // Poll and handle events (inputs, window resize, etc.)
      // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
      // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
      // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
      // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);
        if (event.type == SDL_QUIT)
            done = true;
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
            done = true;

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
                sceneCamera->dolly(event.motion.xrel);
              } else if (event.motion.state & SDL_BUTTON_MMASK) {
                sceneCamera->track(glm::vec2(-static_cast<float>(event.motion.xrel) / static_cast<float>(io.DisplaySize.x), static_cast<float>(event.motion.yrel) / static_cast<float>(io.DisplaySize.y)));
              }
            } else if (event.motion.state & SDL_BUTTON_RMASK) {
              sceneCamera->spin(glm::vec2(event.motion.xrel, event.motion.yrel));
            }

          }
        }
      }

      // Start the Dear ImGui frame
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplSDL2_NewFrame(window);
      ImGui::NewFrame();

      windowRenderTarget->platformSetUpdatedWindowDimensions(io.DisplaySize.x, io.DisplaySize.y);

      {
        ImGui::Begin("RDMA-Client");

        {
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
          if (ImGui::DragFloat("Camera Horizontal FoV", &fov, /*speed=*/1.0f, /*min=*/20.0f, /*max=*/170.0f, /*format=*/"%.1fdeg")) {
            sceneCamera->setFieldOfView(fov);
          }
#if 0
          FxRenderView renderView = sceneCamera->toRenderView(static_cast<float>(io.DisplaySize.x) / static_cast<float>(io.DisplaySize.y));
          {
            glm::mat4 m = renderView.viewMatrix;
            ImGui::Text("View");
            for (size_t i = 0; i < 4; ++i) {
              ImGui::Text("% .3f % .3f % .3f % .3f",
                m[i][0], m[i][1], m[i][2], m[i][3]);
            }
          }
          {
            glm::mat4 m = renderView.projectionMatrix;
            ImGui::Text("Projection");
            for (size_t i = 0; i < 4; ++i) {
              ImGui::Text("% .3f % .3f % .3f % .3f",
                m[i][0], m[i][1], m[i][2], m[i][3]);
            }
          }
#endif

          ImGui::Checkbox("Charuco detection", &enableCharucoDetection);
          ImGui::SliderInt("Charuco Disp Scale", &triangulationDisparityScaleInv, 1, 256); // TODO remove

        }

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
          ImGui::DragFloat("Baseline Scale", &baselineScale, /*speed=*/0.005f,  /*min=*/0.5f, /*max=*/1.5f);
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
          ImGui::DragFloat3("Rx", &v.viewRotation[0], /*speed=*/0.1f, /*min=*/ -75.0f, /*max=*/ 75.0f, "%.3fdeg");


          ImGui::Separator();

          ImGui::PopID();
        }

        if (ImGui::Button("Save Settings")) {
          cameraSystem->saveCalibrationData();
          depthMapGenerator->saveSettings();
        }

        depthMapGenerator->renderIMGUI();

        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
      }

      // distortion test
      if (ImGui::Begin("Distortion test", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        static RHISurface::ptr testSrf;
        static RHIRenderTarget::ptr testRT;
        static bool s_CPU = false;
        ImGui::Checkbox("CPU", &s_CPU);

        if (!testSrf) {
          testSrf = rhi()->newTexture2D(cameraProvider->streamWidth(), cameraProvider->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
          testRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ testSrf }));
        }

        CameraSystem::View& v = cameraSystem->viewAtIndex(0);

        if (s_CPU) {

          CameraSystem::Camera& c = cameraSystem->cameraAtIndex(v.cameraIndices[0]);
          cv::Size imageSize = cv::Size(cameraProvider->streamWidth(), cameraProvider->streamHeight());
          cv::Mat map1, map2;

          cv::initUndistortRectifyMap(c.intrinsicMatrix, c.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, CV_32F, map1, map2);

          cv::Mat res;
          cv::remap(cameraProvider->cvMat(v.cameraIndices[0]), res, map1, map2, cv::INTER_LINEAR);
          rhi()->loadTextureData(testSrf, kVertexElementTypeUByte4N, res.data);
        } else {
          // guts of captureGreyscale
          rhi()->beginRenderPass(testRT, kLoadInvalidate);
          rhi()->bindRenderPipeline(camGreyscaleUndistortPipeline);
          rhi()->loadTexture(ksDistortionMap, v.stereoDistortionMap[0], linearClampSampler);
          rhi()->loadTexture(ksImageTex, cameraProvider->rgbTexture(v.cameraIndices[0]), linearClampSampler);
          rhi()->drawNDCQuad();
          rhi()->endRenderPass(testRT);
        }
        ImGui_Image(testSrf);
      }
      ImGui::End();

      for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
        CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
        char windowName[64];
        sprintf(windowName, "View %zu", viewIdx);
        ImGui::Begin(windowName, NULL, ImGuiWindowFlags_AlwaysAutoResize);

        for (size_t viewCameraIdx = 0; viewCameraIdx < v.cameraCount(); ++viewCameraIdx) {
/*
          if (viewCameraIdx == 1) {
            ImGui::SameLine();
          }
*/

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

          static int disparityScaleSourceLevel = 0;

          float debugDisparityScale = depthMapGenerator->debugDisparityScale();
          if (ImGui::SliderFloat("Debug Disparity Scale", &debugDisparityScale, 0.0f, 2.0f)) {
            depthMapGenerator->setDebugDisparityScale(debugDisparityScale);
          }

          ImGui::SliderInt("Source Level", &disparityScaleSourceLevel, 0, disparitySurface->mipLevels() - 1);

          if (!disparityScaleTarget) {
            disparityScaleSurface = rhi()->newTexture2D(disparitySurface->width(), disparitySurface->height(), kSurfaceFormat_RGBA8);
            disparityScaleTarget = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ disparityScaleSurface }));
          }

          rhi()->beginRenderPass(disparityScaleTarget, kLoadInvalidate);
          rhi()->bindRenderPipeline(disparityScalePipeline);
          rhi()->loadTexture(ksImageTex, disparitySurface);
          DisparityScaleUniformBlock ub;
          ub.disparityScale = depthMapGenerator->disparityPrescale() * depthMapGenerator->debugDisparityScale() * (1.0f / static_cast<float>(depthMapGenerator->maxDisparity()));
          ub.sourceLevel = disparityScaleSourceLevel;
          ub.maxValidDisparityRaw = static_cast<uint32_t>(static_cast<float>(depthMapGenerator->maxDisparity() - 1) / depthMapGenerator->disparityPrescale());
          rhi()->loadUniformBlockImmediate(ksDisparityScaleUniformBlock, &ub, sizeof(ub));
          rhi()->drawFullscreenPass();
          rhi()->endRenderPass(disparityScaleTarget);

          // normalized (UV) coordinates of mouseover of any of the disparity views
          static glm::vec2 disparityHoverUV = glm::vec2(0.0f, 0.0f);
          ImGui_Image(disparityScaleSurface);
          bool hoverLeft = updateHoverPositionForLastItem(disparityHoverUV);
          drawDisparityImageCursorOverlay(disparityHoverUV);

          float disparitySample = depthMapGenerator->debugPeekDisparityUV(internalsTargetView, disparityHoverUV);
          float disparitySampleNormalized = disparitySample / static_cast<float>(disparitySurface->width());

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
          ImGui::Text("Hover UV: {%.2f, %.2f} (%d, %d)\nDisparity: %.3f\nLocal P: %.3fmm, %.3fmm, %.3fmm",
            disparityHoverUV.x, disparityHoverUV.y,
            static_cast<int>(disparityHoverUV.x * static_cast<float>(disparityScaleSurface->width())),
            static_cast<int>(disparityHoverUV.y * static_cast<float>(disparityScaleSurface->height())),
            disparitySample,
            localP.x, localP.y, localP.z);
        }


        ImGui::End();
      }

      // Service RDMA context
      rdmaContext->fireUserEvents();
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

      // Rendering
      FxRenderView renderView = sceneCamera->toRenderView(static_cast<float>(io.DisplaySize.x) / static_cast<float>(io.DisplaySize.y));



      // Note that our camera uses reversed depth projection -- we clear to 0 and use a "greater" depth-test.
      rhi()->setClearColor(glm::vec4(clear_color.x, clear_color.y, clear_color.z, clear_color.w));
      rhi()->setClearDepth(0.0f);
      rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
      rhi()->bindDepthStencilState(standardGreaterDepthStencilState);

#if 0
      { // Draw test quad
        rhi()->bindRenderPipeline(meshVertexColorPipeline);
        rhi()->bindStreamBuffer(0, meshQuadVBO);

        MeshTransformUniformBlock ub;
        ub.modelViewProjection = renderView.viewProjectionMatrix;
        rhi()->loadUniformBlockImmediate(ksMeshTransformUniformBlock, &ub, sizeof(MeshTransformUniformBlock));
        rhi()->drawPrimitives(0, 4);
      }
#endif
      for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
        CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
        if (v.isStereo)
          depthMapGenerator->renderDisparityDepthMap(viewIdx, renderView, cameraSystem->viewWorldTransform(viewIdx));
      }

      if (enableCharucoDetection) {
        std::vector<glm::vec4> pointsStaging;
        ImGui::Begin("ChAruCo");

        static int gizmoType = 1;
        static bool gizmoDepthTest = true;
        ImGui::RadioButton("Triangulated", &gizmoType, 0);
        ImGui::RadioButton("Linear remap", &gizmoType, 1);
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

            std::vector<glm::vec3> triangulatedPoints = getTriangulatedPointsForView(cameraSystem, viewIdx, proc->m_calibrationPoints[0], proc->m_calibrationPoints[1]);

            std::vector<glm::vec3> boardObjectSpacePoints;
            // build object-space point vector of charuco corners
            for (const cv::Point3f& p : s_charucoBoard->chessboardCorners) {
              boardObjectSpacePoints.push_back(glmVec3FromCV(p));
            }

            char idBuf[32];

            glm::mat4 linearRemapXf;
            float linearRemapError;
            {
              std::vector<glm::vec3> p2Points;
              // subset of board points that are visible / have been triangulated
              for (int id : ids) {
                p2Points.push_back(boardObjectSpacePoints[id]);
              }
              linearRemapError = computePointSetLinearTransform(triangulatedPoints, p2Points, linearRemapXf);

              if (linearRemapError >= 0.0) { // returns <0 on failure
                float ex, ey, ez;
                glm::extractEulerAngleYXZ(linearRemapXf, ex, ey, ez);
                ImGui::Text("Linear remap: Tx %.1f %.1f %.1f Rx %.2f %.2f %.2f Error: %.1f",
                  linearRemapXf[3][0] * 1000.0f, linearRemapXf[3][1] * 1000.0f, linearRemapXf[3][2] * 1000.0f,
                  glm::degrees(ex), glm::degrees(ey), glm::degrees(ez), linearRemapError * 1000.0f);

                sprintf(idBuf, "charuco%zu_lr", viewIdx);
                glm::vec3 offset = glm::vec3(0.0f);

                ImGui::BeginChild(idBuf, ImVec2(0, 256), /*border=*/false);
                for (size_t pointIdx = 0; pointIdx < triangulatedPoints.size(); ++pointIdx) {
                  // multiplying by 1000.0f for display in mm
                  glm::vec3 tp = triangulatedPoints[pointIdx] * 1000.0f;
                  glm::vec3 lrp = glm::vec3(linearRemapXf * glm::vec4(p2Points[pointIdx], 1.0f)) * 1000.0f;
                  offset += (lrp - tp);

                  float error = glm::length(lrp - tp);
                  ImGui::Text("[%.2u] triangulated %.1f %.1f %.1f => linear-remap %.1f %.1f %.1f || error: %.1f (%.1f %.1f %.1f)",
                    proc->m_objectIds[0][pointIdx],
                    tp.x, tp.y, tp.z,
                    lrp.x, lrp.y, lrp.z,
                    error,
                    fabs(tp.x - lrp.x), fabs(tp.y - lrp.y), fabs(tp.z - lrp.z));
                }
                offset *= (1.0f / static_cast<float>(triangulatedPoints.size()));
                ImGui::EndChild();

                ImGui::Text("Offset %.3f %.3f %.3f",
                  offset[0] * 1000.0f, offset[1] * 1000.0f, offset[2] * 1000.0f);

                if (gizmoType == 1) {
                  pointsStaging.reserve(pointsStaging.size() + boardObjectSpacePoints.size());
                  glm::mat4 viewXf = cameraSystem->viewWorldTransform(viewIdx);
                  for (const glm::vec3& p : boardObjectSpacePoints) {
                    pointsStaging.push_back(viewXf * (linearRemapXf * glm::vec4(p, 1.0f)));
                  }
                }
              }
            }

            sprintf(idBuf, "charuco%zu_disp", viewIdx);
            if (ImGui::BeginChild(idBuf, ImVec2(0, 256), /*border=*/false)) {
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


            if (gizmoType == 0) { // triangulated
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
          drawTriadGizmosForPoints(pointsBuf, pointsStaging.size(), renderView.viewProjectionMatrix, /*scale=*/-0.025f /*25mm*/);
        }

        ImGui::End();
      }



      // May modify GL state, so this should be done at the end of the renderpass.
      ImGui::Render();

      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      rhi()->endRenderPass(windowRenderTarget);

      rhi()->swapBuffers(windowRenderTarget);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  FxThreading::detail::shutdown();

  windowRenderTarget.reset();

  // XXX: We just leak the GL context to prevent a crash at shutdown due to static resource destruction
  // SDL_GL_DeleteContext(gl_context);
  // SDL_DestroyWindow(window);
  // SDL_Quit();

  return 0;
}
