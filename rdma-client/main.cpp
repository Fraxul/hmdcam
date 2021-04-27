#include "imgui_backend.h"
#include "imgui.h"
#include <stdio.h>
#include <SDL.h>
#include <cuda.h>

#include "rdma/RDMAContext.h"
#include "rdma/RDMABuffer.h"
#include "RDMACameraProvider.h"
#include "common/CameraSystem.h"
#include "common/CharucoMultiViewCalibration.h"
#include "common/FxCamera.h"
#include "common/FxThreading.h"
#include "common/DepthMapGenerator.h"
#include "common/DGPUWorkerControl.h"

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

RHIRenderPipeline::ptr meshVertexColorPipeline;
RHIBuffer::ptr meshQuadVBO;

FxAtomicString ksMeshTransformUniformBlock("MeshTransformUniformBlock");
static FxAtomicString ksImageTex("imageTex");
struct MeshTransformUniformBlock {
  glm::mat4 modelViewProjection;
};


static FxAtomicString ksDisparityScaleUniformBlock("DisparityScaleUniformBlock");
struct DisparityScaleUniformBlock {
  float disparityScale;
  int sourceLevel;
  float pad3;
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



// Main code
int main(int argc, char** argv) {

  if (argc <= 1) {
    printf("usage: %s hostname\n", argv[0]);
    return -1;
  }

  FxThreading::detail::init();

  shm = SHMSegment<DepthMapSHM>::createSegment("cuda-dgpu-worker", 16*1024*1024);
  printf("Waiting for DGPU worker...\n");
  if (!spawnAndWaitForDGPUWorker(&shm->segment()->m_workerReadySem)) {
    return 1;
  }

  const char* rdmaHost = argv[1];

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
  bool enableCharucoDetection = false;

  // CV processing init
  depthMapGenerator = new DepthMapGenerator(cameraSystem, shm);
  depthMapGenerator->setPopulateDebugTextures(true);

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
          ImGui::DragFloat3("Tx", &v.viewTranslation[0], /*speed=*/ 0.1f, /*min=*/ -10.0f, /*max=*/ 10.0f);
          ImGui::DragFloat3("Rx", &v.viewRotation[0], /*speed=*/0.1f, /*min=*/ -75.0f, /*max=*/ 75.0f, "%.3fdeg");


          ImGui::Separator();

          ImGui::PopID();
        }

        depthMapGenerator->renderIMGUI();

        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
      }

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

          static int disparityScale = 2;
          static int disparityScaleSourceLevel = 0;
          ImGui::SliderInt("Disparity Scale", &disparityScale, 1, 128);
          ImGui::SliderInt("Source Level", &disparityScaleSourceLevel, 0, disparitySurface->mipLevels() - 1);

          if (!disparityScaleTarget) {
            disparityScaleSurface = rhi()->newTexture2D(disparitySurface->width(), disparitySurface->height(), kSurfaceFormat_RGBA8);
            disparityScaleTarget = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ disparityScaleSurface }));
          }

          rhi()->beginRenderPass(disparityScaleTarget, kLoadInvalidate);
          rhi()->bindRenderPipeline(disparityScalePipeline);
          rhi()->loadTexture(ksImageTex, disparitySurface);
          DisparityScaleUniformBlock ub;
          ub.disparityScale = depthMapGenerator->m_disparityPrescale *  (1.0f / static_cast<float>(disparityScale));
          ub.sourceLevel = disparityScaleSourceLevel;
          rhi()->loadUniformBlockImmediate(ksDisparityScaleUniformBlock, &ub, sizeof(ub));
          rhi()->drawFullscreenPass();
          rhi()->endRenderPass(disparityScaleTarget);

          ImGui_Image(disparityScaleSurface);
          ImGui_Image(depthMapGenerator->leftGrayscale(internalsTargetView));
          ImGui_Image(depthMapGenerator->rightGrayscale(internalsTargetView));
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
          charucoProcessors[viewIdx]->m_calibrationPoints[0].clear();
          charucoProcessors[viewIdx]->m_calibrationPoints[1].clear();
          charucoProcessors[viewIdx]->m_objectPoints.clear();

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
          depthMapGenerator->renderDisparityDepthMap(viewIdx, renderView, v.viewTransform());
      }

      if (enableCharucoDetection) {
        std::vector<glm::vec4> pointsStaging;
        ImGui::Begin("ChAruCo");

        for (size_t viewIdx = 0; viewIdx < cameraSystem->views(); ++viewIdx) {
          if (viewIdx != 0)
            ImGui::Separator();
          ImGui::Text("View %zu", viewIdx);

          CameraSystem::View& v = cameraSystem->viewAtIndex(viewIdx);
          CharucoMultiViewCalibration* proc = charucoProcessors[viewIdx];
          if (!proc)
            continue;

          if (proc->m_calibrationPoints.empty() || proc->m_objectPoints.empty())
            continue; // nothing detected this frame

          if (v.isStereo) {
            cv::Mat points = getTriangulatedPointsForView(cameraSystem, viewIdx, proc->m_calibrationPoints[0], proc->m_calibrationPoints[1]);
            // Points are point-per-row, 1 column, CV_32FC3
            glm::mat4 viewXf = v.viewTransform();

            size_t offset = pointsStaging.size();
            std::vector<cv::Point2f> lp = flattenVector(proc->m_calibrationPoints[0]);
            std::vector<cv::Point2f> rp = flattenVector(proc->m_calibrationPoints[1]);

            pointsStaging.resize(pointsStaging.size() + points.rows);
            for (int pointIdx = 0; pointIdx < points.rows; ++pointIdx) {
              float* p = points.ptr<float>(pointIdx);
              ImGui::Text("%.4f %.4f || %.4f %.4f || %.4f %.4f %.4f",
                lp[pointIdx].x, lp[pointIdx].y,
                rp[pointIdx].x, rp[pointIdx].y,
                p[0], p[1], p[2]);
              pointsStaging[offset + pointIdx] = viewXf * glm::vec4(p[0], p[1], p[2], 1.0f);
            }

          } else {
            // TODO do something useful with markers detected in 2d views
          }

        }

        RHIBuffer::ptr pointsBuf = rhi()->newBufferWithContents(pointsStaging.data(), pointsStaging.size() * sizeof(float) * 4);

        // render points as locator gizmos
        drawTriadGizmosForPoints(pointsBuf, pointsStaging.size(), renderView.viewProjectionMatrix);

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

  SDL_GL_DeleteContext(gl_context);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
