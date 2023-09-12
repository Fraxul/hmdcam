#include "imgui_backend.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "InputListener.h"
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


FxAtomicString ksTexture("sTexture");

void ImGui_ImplInputListener_Init() {
  // ImGuiIO& io = ImGui::GetIO();
}

void ImGui_ImplInputListener_NewFrame() {
  ImGuiIO& io = ImGui::GetIO();

  io.AddKeyEvent(ImGuiKey_UpArrow, testButton(kButtonUp));
  io.AddKeyEvent(ImGuiKey_DownArrow, testButton(kButtonDown));
  io.AddKeyEvent(ImGuiKey_LeftArrow, testButton(kButtonLeft));
  io.AddKeyEvent(ImGuiKey_RightArrow, testButton(kButtonRight));
  io.AddKeyEvent(ImGuiKey_Space, testButton(kButtonOK));
  io.AddKeyEvent(ImGuiKey_Escape, testButton(kButtonBack));
}


static RHISurface::ptr imguiFontAtlas;
static RHIBlendState::ptr imguiBlendState;
static RHIRenderPipeline::ptr imguiPipeline;

void ImGui_ImplFxRHI_Init() {
  ImGuiIO& io = ImGui::GetIO();
  io.BackendRendererName = "FxEngine";
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

  // Generate font atlas
  unsigned char* pixels;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  imguiFontAtlas = rhi()->newTexture2D(width, height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  rhi()->loadTextureData(imguiFontAtlas, kVertexElementTypeUByte4N, pixels);

  // ImTextureID is a raw pointer. We rely on the static variable to keep the font atlas alive.
  io.Fonts->SetTexID(&(*imguiFontAtlas));

  if (!imguiBlendState) {
    // imgui blending mode, from the GL3 sample backend
    imguiBlendState = rhi()->compileBlendState(RHIBlendStateDescriptorElement(kBlendSourceAlpha, kBlendOneMinusSourceAlpha, kBlendOne, kBlendOneMinusSourceAlpha));
  }
  if (!imguiPipeline) {
    imguiPipeline = rhi()->compileRenderPipeline("shaders/imgui.vtx.glsl", "shaders/imgui.frag.glsl", RHIVertexLayout({
      RHIVertexLayoutElement(0, kVertexElementTypeFloat2,  "Position", offsetof(ImDrawVert, pos), sizeof(ImDrawVert)),
      RHIVertexLayoutElement(0, kVertexElementTypeFloat2,  "UV",       offsetof(ImDrawVert, uv),  sizeof(ImDrawVert)),
      RHIVertexLayoutElement(0, kVertexElementTypeUByte4N, "Color",    offsetof(ImDrawVert, col), sizeof(ImDrawVert))
    }), kPrimitiveTopologyTriangleList);
  }

}

void ImGui_ImplFxRHI_NewFrame() {
  // Nothing to do here
}

void ImGui_ImplFxRHI_RenderDrawData(RHIRenderTarget::ptr renderTarget, ImDrawData* draw_data) {
  ImDrawData* drawData = ImGui::GetDrawData();
  if (drawData->TotalVtxCount && drawData->TotalIdxCount) {

    ImDrawVert* vertexData = new ImDrawVert[drawData->TotalVtxCount];
    ImDrawIdx* indexData = new ImDrawIdx[drawData->TotalIdxCount];

    size_t vertexBase = 0;
    size_t indexBase = 0;
    std::vector<ImDrawCmd> drawCommands;

    for (int cmdListIdx = 0; cmdListIdx < drawData->CmdListsCount; ++cmdListIdx) {
      ImDrawList* cmdList = drawData->CmdLists[cmdListIdx];

      memcpy(vertexData + vertexBase, cmdList->VtxBuffer.Data, sizeof(ImDrawVert) * cmdList->VtxBuffer.Size);
      memcpy(indexData + indexBase, cmdList->IdxBuffer.Data, sizeof(ImDrawIdx) * cmdList->IdxBuffer.Size);

      // Generate draw commands, applying the additional base offset to VtxOffset and IdxOffset to account for the single-buffer strategy

      drawCommands.reserve(drawCommands.size() + cmdList->CmdBuffer.Size);

      for (int drawCmdIdx = 0; drawCmdIdx < cmdList->CmdBuffer.Size; ++drawCmdIdx) {
        ImDrawCmd cmd = cmdList->CmdBuffer[drawCmdIdx];
        cmd.VtxOffset += vertexBase;
        cmd.IdxOffset += indexBase;
        drawCommands.push_back(cmd);
      }
      vertexBase += cmdList->VtxBuffer.Size;
      indexBase += cmdList->IdxBuffer.Size;
    }

    if (!drawCommands.empty()) {
      RHIBuffer::ptr uiVertexBuffer = rhi()->newBufferWithContents(vertexData, drawData->TotalVtxCount * sizeof(ImDrawVert), kBufferUsageCPUWriteOnly);
      RHIBuffer::ptr uiIndexBuffer = rhi()->newBufferWithContents(indexData, drawData->TotalIdxCount * sizeof(ImDrawIdx), kBufferUsageCPUWriteOnly);

      rhi()->bindDepthStencilState(disabledDepthStencilState);
      rhi()->bindBlendState(imguiBlendState);
      rhi()->bindRenderPipeline(imguiPipeline);

      ImVec2 clip_off = draw_data->DisplayPos; // (0,0) unless using multi-viewports
      ImVec2 clip_scale = draw_data->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

      // ortho projection matrix
      glm::mat4 mvp = glm::ortho<float>(/*left=*/0.0f, /*right=*/renderTarget->width() / drawData->FramebufferScale.x, /*bottom=*/renderTarget->height() / drawData->FramebufferScale.y, /*top=*/0.0f);
      rhi()->loadUniformBlockImmediate(ksUILayerUniformBlock, &mvp, sizeof(glm::mat4));

      for (const ImDrawCmd& drawCmd : drawCommands) {
        // Project scissor/clipping rectangles into framebuffer space
        RHIRect scissor = RHIRect::ltrb(
          (drawCmd.ClipRect.x - clip_off.x) * clip_scale.x,
          (drawCmd.ClipRect.y - clip_off.y) * clip_scale.y,
          (drawCmd.ClipRect.z - clip_off.x) * clip_scale.x,
          (drawCmd.ClipRect.w - clip_off.y) * clip_scale.y);
        // Coordsys correction for scissor rect
        scissor.y = renderTarget->height() - (scissor.y + scissor.height);

        if (scissor.x < renderTarget->width() && scissor.y < renderTarget->height() && scissor.width > 0 && scissor.height > 0) {
          // Setup per-command draw state
          rhi()->setScissorRect(scissor);
          rhi()->loadTexture(ksTexture, static_cast<RHISurface*>(drawCmd.TextureId), linearClampSampler);

          rhi()->bindStreamBuffer(0, uiVertexBuffer, /*offsetBytes=*/drawCmd.VtxOffset * sizeof(ImDrawVert));
          rhi()->drawIndexedPrimitives(uiIndexBuffer, kIndexBufferTypeUInt16, drawCmd.ElemCount, drawCmd.IdxOffset);
        }
      }
    }

    delete[] vertexData;
    delete[] indexData;
  }
}

