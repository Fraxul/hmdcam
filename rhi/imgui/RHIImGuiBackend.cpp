#include "rhi/imgui/RHIImGuiBackend.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


static FxAtomicString ksTexture("sTexture");

static RHISurface::ptr imguiFontAtlas;
static RHIBlendState::ptr imguiBlendState;
static RHIRenderPipeline::ptr imguiPipeline;

static constexpr size_t kVertexCountIncrement = (1 << 16);
static constexpr size_t kIndexCountIncrement = (1 << 18);


struct FrameData {
  // Host-mapped buffers.

  RHIBuffer::ptr vertexBuffer;
  RHIBuffer::ptr indexBuffer;

  // Ensure buffers exist and are large enough to hold the specified vertex and index counts.
  void ensureBufferSize(ImDrawData* drawData) {
    int TotalVtxCount = drawData->TotalVtxCount;
    int TotalIdxCount = drawData->TotalIdxCount;
    if (!vertexBuffer || vertexBuffer->size() < (sizeof(ImDrawVert) * TotalVtxCount)) {
      size_t newVertexBufferSize = sizeof(ImDrawVert) * ((TotalVtxCount + (kVertexCountIncrement - 1)) & (~(kVertexCountIncrement - 1)));
      // fprintf(stderr, "RHIImGuiBackend::FrameData(%p): new vertex buffer size %zu bytes\n", this, newVertexBufferSize);
      vertexBuffer = rhi()->newEmptyBuffer(newVertexBufferSize, kBufferUsageCPUWriteOnly);
      vertexBuffer->map(kBufferMapWriteOnly);
    }

    if (!indexBuffer || indexBuffer->size() < (sizeof(ImDrawIdx) * TotalIdxCount)) {
      size_t newIndexBufferSize = sizeof(ImDrawIdx) * ((TotalIdxCount + (kIndexCountIncrement - 1)) & (~(kIndexCountIncrement - 1)));
      // fprintf(stderr, "RHIImGuiBackend::FrameData(%p): new index buffer size %zu bytes\n", this, newIndexBufferSize);
      indexBuffer = rhi()->newEmptyBuffer(newIndexBufferSize, kBufferUsageCPUWriteOnly);
      indexBuffer->map(kBufferMapWriteOnly);
    }
  }

  void releaseResources() {
    vertexBuffer.reset();
    indexBuffer.reset();
  }
};

static constexpr size_t kFrameDataRingSize = 4;
static FrameData frameDataRing[kFrameDataRingSize];
static uint32_t frameDataRingIndex = 0;

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
    // clang-format off
    imguiPipeline = rhi()->compileRenderPipeline("shaders/imgui.vtx.glsl", "shaders/imgui.frag.glsl", RHIVertexLayout({
      RHIVertexLayoutElement(0, kVertexElementTypeFloat2,  "Position", offsetof(ImDrawVert, pos), sizeof(ImDrawVert)),
      RHIVertexLayoutElement(0, kVertexElementTypeFloat2,  "UV",       offsetof(ImDrawVert, uv),  sizeof(ImDrawVert)),
      RHIVertexLayoutElement(0, kVertexElementTypeUByte4N, "Color",    offsetof(ImDrawVert, col), sizeof(ImDrawVert))
    }), kPrimitiveTopologyTriangleList);
    // clang-format on
  }
}

void ImGui_ImplFxRHI_NewFrame() {
  // Advance frame data ring buffer cursor
  frameDataRingIndex += 1;
  if (frameDataRingIndex >= kFrameDataRingSize)
    frameDataRingIndex = 0;
}

void ImGui_ImplFxRHI_Shutdown() {
  // Release before the RHI device is destroyed. Static destructors fire
  // after main(); by that point shutdownRHI() may have already torn the
  // device down, leading to crashes in the RHI object dtors.
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->SetTexID(nullptr);
  imguiFontAtlas.reset();
  imguiBlendState.reset();
  imguiPipeline.reset();
  for (size_t i = 0; i < kFrameDataRingSize; ++i) {
    frameDataRing[i].releaseResources();
  }
}

void ImGui_ImplFxRHI_RenderDrawData(RHIRenderTarget::ptr renderTarget, ImDrawData* draw_data) {
  ImDrawData* drawData = ImGui::GetDrawData();
  if (drawData->TotalVtxCount && drawData->TotalIdxCount) {

    FrameData& frameData = frameDataRing[frameDataRingIndex];
    frameData.ensureBufferSize(drawData);

    ImDrawVert* vertexData = reinterpret_cast<ImDrawVert*>(frameData.vertexBuffer->data());
    ImDrawIdx* indexData = reinterpret_cast<ImDrawIdx*>(frameData.indexBuffer->data());

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
      rhi()->bindDepthStencilState(disabledDepthStencilState);
      rhi()->bindBlendState(imguiBlendState);
      rhi()->bindRenderPipeline(imguiPipeline);

      ImVec2 clip_off = draw_data->DisplayPos; // (0,0) unless using multi-viewports
      ImVec2 clip_scale = draw_data->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

      // ortho projection matrix for Vulkan top-is-zero orientation.
      glm::mat4 mvp = glm::ortho<float>(/*left=*/ 0.0f, /*right=*/ renderTarget->width() / drawData->FramebufferScale.x, /*top=*/ 0.0f, /*bottom=*/ renderTarget->height() / drawData->FramebufferScale.y);
      rhi()->loadUniformBlockImmediate(ksUILayerUniformBlock, &mvp, sizeof(glm::mat4));

      for (const ImDrawCmd& drawCmd : drawCommands) {
        // Project scissor/clipping rectangles into framebuffer space
        RHIRect scissor = RHIRect::ltrb(
          (drawCmd.ClipRect.x - clip_off.x) * clip_scale.x,
          (drawCmd.ClipRect.y - clip_off.y) * clip_scale.y,
          (drawCmd.ClipRect.z - clip_off.x) * clip_scale.x,
          (drawCmd.ClipRect.w - clip_off.y) * clip_scale.y);

        if (scissor.x < renderTarget->width() && scissor.y < renderTarget->height() && scissor.width > 0 && scissor.height > 0) {
          // Setup per-command draw state
          rhi()->setScissorRect(scissor);
          rhi()->loadTexture(ksTexture, static_cast<RHISurface*>(drawCmd.TextureId), linearClampSampler);

          rhi()->bindStreamBuffer(0, frameData.vertexBuffer, /*offsetBytes=*/ drawCmd.VtxOffset * sizeof(ImDrawVert));
          rhi()->drawIndexedPrimitives(frameData.indexBuffer, kIndexBufferTypeUInt16, drawCmd.ElemCount, drawCmd.IdxOffset);
        }
      }
    }
  }
}
