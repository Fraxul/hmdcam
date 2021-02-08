#include "imgui_backend.h"
#include "imgui.h"
#include <stdio.h>
#include <SDL.h>
#include <cuda.h>

#include "rdma/RDMAContext.h"
#include "rdma/RDMABuffer.h"
#include "RDMACameraProvider.h"
#include "common/CameraSystem.h"
#include "FxCamera.h"
#include "OpenCVProcess.h"

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
OpenCVProcess* cvProcess;

RHIRenderPipeline::ptr meshVertexColorPipeline;
RHIRenderPipeline::ptr depthMeshPipeline;
RHIBuffer::ptr meshQuadVBO;

FxAtomicString ksMeshTransformUniformBlock("MeshTransformUniformBlock");
static FxAtomicString ksImageTex("imageTex");
struct MeshTransformUniformBlock {
  glm::mat4 modelViewProjection;
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




// Main code
int main(int argc, char** argv) {

  if (argc <= 1) {
    printf("usage: %s hostname\n", argv[0]);
    return -1;
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

  depthMeshPipeline = rhi()->compileRenderPipeline("shaders/meshTexture.vtx.glsl", "shaders/meshTexture.frag.glsl", RHIVertexLayout({
      RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "position",           0, sizeof(float) * 4),
      RHIVertexLayoutElement(1, kVertexElementTypeFloat2, "textureCoordinates", 0, sizeof(float) * 2)
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

  // CV processing init
  cvProcess = new OpenCVProcess(cameraSystem, cameraProvider, /*viewIdx=*/0);
  cvProcess->OpenCVAppStart();

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

      // Service RDMA context
      rdmaContext->fireUserEvents();
      cameraProvider->updateSurfaces();
      cvProcess->Prerender();


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
          ImGui::Text("Camera pos: %.3f %.3f %.3f", p[0], p[1], p[2]);
          ImGui::Text("Camera target: %.3f %.3f %.3f", t[0], t[1], t[2]);
        }

        ImGui::InputInt("Stereo Algorithm", &cvProcess->m_iNextStereoAlgorithm);
        ImGui::Text("Proc Frames: %d", cvProcess->m_iProcFrames);
        if (ImGui::Button("Req SCreenshot"))
          cvProcess->m_bScreenshotNext = true;


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
          if (viewCameraIdx == 1) {
            ImGui::SameLine();
          }

          size_t cameraIdx = v.cameraIndices[viewCameraIdx];
          RHISurfaceGL* glSrf = static_cast<RHISurfaceGL*>(cameraProvider->rgbTexture(cameraIdx).get());

          // TODO apply distortion correction here
          ImGui::Image((ImTextureID) static_cast<uintptr_t>(glSrf->glId()), ImVec2(glSrf->width()/v.cameraCount(), glSrf->height()/v.cameraCount()));
        }
        ImGui::End();
      }

      // Rendering
      ImGui::Render();

      FxRenderView renderView = sceneCamera->toRenderView(static_cast<float>(io.DisplaySize.x) / static_cast<float>(io.DisplaySize.y));



      // Note that our camera uses reversed depth projection -- we clear to 0 and use a "greater" depth-test.
      rhi()->setClearColor(glm::vec4(clear_color.x, clear_color.y, clear_color.z, clear_color.w));
      rhi()->setClearDepth(0.0f);
      rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
      rhi()->bindDepthStencilState(standardGreaterDepthStencilState);

      { // Draw test quad
        rhi()->bindRenderPipeline(meshVertexColorPipeline);
        rhi()->bindStreamBuffer(0, meshQuadVBO);

        MeshTransformUniformBlock ub;
        ub.modelViewProjection = renderView.viewProjectionMatrix;
        rhi()->loadUniformBlockImmediate(ksMeshTransformUniformBlock, &ub, sizeof(MeshTransformUniformBlock));
        rhi()->drawPrimitives(0, 4);
      }


      { // Draw depth map from OpenCV process
        rhi()->bindRenderPipeline(depthMeshPipeline);
        rhi()->bindStreamBuffer(0, cvProcess->m_geoDepthMapPositionBuffer);
        rhi()->bindStreamBuffer(1, cvProcess->m_geoDepthMapTexcoordBuffer);

        MeshTransformUniformBlock ub;
        ub.modelViewProjection = renderView.viewProjectionMatrix;
        rhi()->loadUniformBlockImmediate(ksMeshTransformUniformBlock, &ub, sizeof(MeshTransformUniformBlock));
        rhi()->loadTexture(ksImageTex, cvProcess->m_iTexture);

        rhi()->drawIndexedPrimitives(cvProcess->m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, cvProcess->m_geoDepthMapTristripIndexCount);
      }


      // May modify GL state, so this should be done at the end of the renderpass.
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      rhi()->endRenderPass(windowRenderTarget);

      rhi()->swapBuffers(windowRenderTarget);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  SDL_GL_DeleteContext(gl_context);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
