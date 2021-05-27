#include "imgui_backend.h"
#include "imgui.h"
#include <stdio.h>
#include <SDL.h>
#include <cuda.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <glm/gtx/transform.hpp>

#include "common/glmCvInterop.h"
#include "common/FxCamera.h"
#include "common/FxThreading.h"

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

FxCamera* sceneCamera;

RHIRenderPipeline::ptr meshVertexColorPipeline;
RHIBuffer::ptr meshQuadVBO;

FxAtomicString ksMeshTransformUniformBlock("MeshTransformUniformBlock");
static FxAtomicString ksImageTex("imageTex");
struct MeshTransformUniformBlock {
  glm::mat4 modelViewProjection;
};


void ImGui_Image(RHISurface::ptr img, const ImVec2& uv0 = ImVec2(0,0), const ImVec2& uv1 = ImVec2(1,1)) {
  ImGui::Image((ImTextureID) static_cast<uintptr_t>(static_cast<RHISurfaceGL*>(img.get())->glId()), ImVec2(img->width(), img->height()), uv0, uv1);
}


cv::Mat initCameraMatrix(const cv::Size& imageSize, double fovX_degrees) {
  cv::Matx33d K = cv::Matx33d::eye();
  K(0, 2) = (static_cast<double>(imageSize.width)  * 0.5) - 0.5; // 959.5
  K(1, 2) = (static_cast<double>(imageSize.height) * 0.5) - 0.5; // 539.5
  double aspect = (static_cast<double>(imageSize.height) / static_cast<double>(imageSize.width));

  K(0, 0) = imageSize.width / (2.0 * tan(glm::radians(fovX_degrees * 0.5)));
  K(1, 1) = imageSize.height / (2.0 * tan(glm::radians((fovX_degrees * aspect) * 0.5)));
  return cv::Mat(K);
}

// Main code
int main(int argc, char** argv) {

  FxThreading::detail::init();

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
  SDL_Window* window = SDL_CreateWindow("stereo-geometry", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1920, 1080, window_flags);
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

  cv::Size imageSize = cv::Size(1920, 1080);
  float fovX_degrees = 86.0;
  cv::Mat cameraMatrix = initCameraMatrix(imageSize, fovX_degrees);

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
        ImGui::Begin("Stereo Geometry");

        {
          glm::vec3 p = sceneCamera->position();
          glm::vec3 t = sceneCamera->targetPosition();
          if (ImGui::InputFloat3("Camera Position", &p[0]))
            sceneCamera->setPosition(p);

          if (ImGui::InputFloat3("Camera Target", &t[0]))
            sceneCamera->setTargetPosition(t);

          float fov = sceneCamera->fieldOfView();
          if (ImGui::DragFloat("Camera Horizontal FoV", &fov, /*speed=*/1.0f, /*min=*/20.0f, /*max=*/170.0f, /*format=*/"%.1fdeg")) {
            sceneCamera->setFieldOfView(fov);
          }

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
          ImGui::Separator();

        }



        bool cameraMatrixDirty = false;
        cameraMatrixDirty |= ImGui::DragFloat("FoV-X", &fovX_degrees, /*speed=*/1.0f, /*min=*/20.0f, /*max=*/170.0f, /*format=*/"%.1fdeg");

        if (cameraMatrixDirty) {
          cameraMatrix = initCameraMatrix(imageSize, fovX_degrees);
        }

        ImGui::Text("Camera Matrix:");
        for (size_t i = 0; i < 3; ++i) {
          double* rp = cameraMatrix.ptr<double>(i);
          ImGui::Text("%8.3f %8.3f %8.3f", rp[0], rp[1], rp[2]);
        }

        {
          double fx, fy, fl, ar;
          cv::Point2d pp;
          // apertureWidth / apertureHeight provided as 0.0, we only care about the FoV anyway
          cv::calibrationMatrixValues(cameraMatrix, imageSize, 0.0, 0.0, fx, fy, fl, pp, ar);
          ImGui::Text("Computed FoV: %.3f x %.3f deg", fx, fy);
        }




        ImGui::Separator();
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
      }

      // Rendering
      FxRenderView renderView = sceneCamera->toRenderView(static_cast<float>(io.DisplaySize.x) / static_cast<float>(io.DisplaySize.y));

      // Note that our camera uses reversed depth projection -- we clear to 0 and use a "greater" depth-test.
      rhi()->setClearColor(glm::vec4(clear_color.x, clear_color.y, clear_color.z, clear_color.w));
      rhi()->setClearDepth(0.0f);
      rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
      rhi()->bindDepthStencilState(standardGreaterDepthStencilState);

#if 1
      { // Draw test quad
        rhi()->bindRenderPipeline(meshVertexColorPipeline);
        rhi()->bindStreamBuffer(0, meshQuadVBO);

        MeshTransformUniformBlock ub;
        ub.modelViewProjection = renderView.viewProjectionMatrix * glm::translate(glm::vec3(0.0f, 0.0f, -5.0f));
        rhi()->loadUniformBlockImmediate(ksMeshTransformUniformBlock, &ub, sizeof(MeshTransformUniformBlock));
        rhi()->drawPrimitives(0, 4);
      }
#endif

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

