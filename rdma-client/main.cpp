#include "imgui_backend.h"
#include "imgui.h"
#include <stdio.h>
#include <SDL.h>
#include <cuda.h>

#include "rdma/RDMAContext.h"
#include "rdma/RDMABuffer.h"

#include "rhi/RHI.h"
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

RDMABuffer::ptr configBuffer;
std::vector<RDMABuffer::ptr> cameraRDMABuffers;
std::vector<RHISurface::ptr> cameraSurfaces;

bool cameraBuffersDirty = false;
void rdmaUserEventCallback(RDMAContext*, uint32_t userEventID, SerializationBuffer payload) {
  switch (userEventID) {
    case 1:
      cameraBuffersDirty = true;
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

  // Create camera RDMA buffers

  // Read RDMA config from server
  RDMABuffer::ptr configBuf = rdmaContext->newManagedBuffer("config", 8192, kRDMABufferUsageWriteDestination); // TODO should be read source / use RDMA read instead of relying on push
  uint32_t cameraCount, cameraWidth, cameraHeight;

  printf("Waiting to read config...");
  fflush(stdout);
  for (size_t configWait = 0;;  ++configWait) {
    printf(".");
    fflush(stdout);

    SerializationBuffer cfg(configBuf->data(), configBuf->size());
    cameraCount = cfg.get_u32();
    cameraWidth = cfg.get_u32();
    cameraHeight = cfg.get_u32();
    if (cameraCount && cameraWidth && cameraHeight) {
      printf(" Done.\nReceived config: %u cameras @ %ux%u\n", cameraCount, cameraWidth, cameraHeight);
      break;
    }

    if (configWait > 10) {
      printf(" Timed out.\n");
      return -1;
    }

    sleep(1); // TODO messy
  }

  size_t cameraRowStride = cameraWidth * 4; // TODO not sure if this will always be correct

  for (size_t cameraIdx = 0; cameraIdx < cameraCount; ++cameraIdx) {
    char key[32];
    sprintf(key, "camera%zu", cameraIdx);
    cameraRDMABuffers.push_back(rdmaContext->newManagedBuffer(std::string(key), cameraRowStride * cameraHeight, kRDMABufferUsageWriteDestination));

    cameraSurfaces.push_back(rhi()->newTexture2D(cameraWidth, cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_RGBA8)));
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
  bool show_another_window = false;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  // Main loop
  bool done = false;
  while (!done)
  {

      // Poll and handle events (inputs, window resize, etc.)
      // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
      // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
      // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
      // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
      SDL_Event event;
      while (SDL_PollEvent(&event))
      {
          ImGui_ImplSDL2_ProcessEvent(&event);
          if (event.type == SDL_QUIT)
              done = true;
          if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
              done = true;
      }

      // Service RDMA context
      rdmaContext->fireUserEvents();

      if (cameraBuffersDirty) {
        for (size_t cameraIdx = 0; cameraIdx < cameraRDMABuffers.size(); ++cameraIdx) {
          rhi()->loadTextureData(cameraSurfaces[cameraIdx], kVertexElementTypeUByte4N, cameraRDMABuffers[cameraIdx]->data());
        }

        cameraBuffersDirty = false;
      }

      // Start the Dear ImGui frame
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplSDL2_NewFrame(window);
      ImGui::NewFrame();

      windowRenderTarget->platformSetUpdatedWindowDimensions(io.DisplaySize.x, io.DisplaySize.y);

      {
          static float f = 0.0f;
          static int counter = 0;

          ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

          ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
          ImGui::Checkbox("Another Window", &show_another_window);

          ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
          ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

          if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
              counter++;
          ImGui::SameLine();
          ImGui::Text("counter = %d", counter);

          ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
          ImGui::End();
      }

      // 3. Show another simple window.
      if (show_another_window)
      {
          ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
          ImGui::Text("Hello from another window!");
          if (ImGui::Button("Close Me"))
              show_another_window = false;
          ImGui::End();
      }

      for (size_t cameraIdx = 0; cameraIdx < cameraSurfaces.size(); ++cameraIdx) {
        char windowName[32];
        sprintf(windowName, "Camera %zu", cameraIdx);
        ImGui::Begin(windowName);
        RHISurfaceGL* glSrf = static_cast<RHISurfaceGL*>(cameraSurfaces[cameraIdx].get());

        ImGui::Image((ImTextureID) static_cast<uintptr_t>(glSrf->glId()), ImVec2(glSrf->width(), glSrf->height()), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));

        ImGui::End();
      }

      // Rendering
      ImGui::Render();
      glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
      glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

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
