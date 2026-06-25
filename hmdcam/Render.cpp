#include "Render.h"
#include "RenderBackend.h"
#include "RenderBackendVKDirect.h"
#include "RenderBackendWayland.h"
#include "rhi/vk/RHIVK.h"
#include "common/Timing.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/GLCommon.h"
#include "rhi/cuda/RHICUDA.h"

#include "xrt/xrt_instance.h"
#include "xrt/xrt_space.h"
#include "xrt/xrt_system.h"
#include "xrt/xrt_device.h"
#include "math/m_api.h"
#include "util/u_distortion_mesh.h"
#include "util/u_device.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <vector>
#include <sys/time.h>

// extern bool vive_watchman_enable; // hacky; symbol added in xrt/drivers/vive/vive_device.c to disable watchman thread (since we don't use lighthouse tracking)

RHIRenderTarget::ptr windowRenderTarget;

FxAtomicString ksMVPUniformBlock("MVPUniformBlock");
FxAtomicString ksModelSpaceClippedQuadUniformBlock("ModelSpaceClippedQuadUniformBlock");
FxAtomicString ksSolidQuadUniformBlock("SolidQuadUniformBlock");

RHIRenderPipeline::ptr solidQuadPipeline;

RHISurface::ptr disabledMaskTex;

RHIRenderPipeline::ptr mesh1chDistortionPipeline;
RHIRenderPipeline::ptr mesh3chDistortionPipeline;

// Lens aperture mask: the distortion mesh fills the whole display rectangle, but
// the lens only sees a circular region, so the eye-target corners are hidden even
// where the distortion footprint covers them. This annulus (rect minus the FOV-
// derived aperture ellipse) masks them.
RHIRenderPipeline::ptr hiddenAreaMeshPipeline;
RHIBuffer::ptr apertureMaskVertexBuffer[2];
uint32_t apertureMaskVertexCount[2] = {0, 0};

FxAtomicString ksOverlayTex("overlayTex");
FxAtomicString ksMaskTex("maskTex");

// combined eye render target (pre distortion)
RHISurface::ptr eyeTex;
RHISurface::ptr eyeDepthRenderbuffer;
RHIRenderTarget::ptr eyeRT;
RHIRect eyeViewports[2]; // viewports in the eye RT
RHIRect eyePostDistortionViewports[2]; // viewports on the HMD window surface

// distortion parameter buffers
RHIBuffer::ptr meshDistortionVertexBuffer, meshDistortionIndexBuffer;
struct MeshDistortionUniformBlock {
  glm::vec2 uvOffset;
  glm::vec2 uvScale;
};
static FxAtomicString ksMeshDistortionUniformBlock("MeshDistortionUniformBlock");

// HMD info/state
struct xrt_instance* xrtInstance = NULL;
struct xrt_device* xrtHMDevice = NULL;
struct xrt_system_devices* xrtSystemDevices = NULL;
struct xrt_space_overseer* xrtSpaceOverseer = nullptr;
struct xrt_input* xrtHeadDetectInput = nullptr; // optional, can be null

bool isDummyHMD = false;
unsigned int hmd_width, hmd_height;
unsigned int eye_width, eye_height;
glm::mat4 eyeProjection[2];
glm::mat4 eyeView[2];

RenderBackend* renderBackend = NULL;

// -----------

UserPresenceState RenderGetUserPresenceState() {
  if (xrtHeadDetectInput && xrtHeadDetectInput->active) {
    return xrtHeadDetectInput->value.boolean ? kUserPresenceState_Present : kUserPresenceState_NotPresent;
  }
  return kUserPresenceState_Unknown;
}

// Build the lens aperture mask per eye. The aperture is a circle in tangent
// space centered on the optical axis (boresight, tangent 0,0) and inscribed in
// the FOV (radius = nearest half-FOV tangent). Under the eye projection (see
// recomputeHMDParameters: tangent (tx,ty) -> NDC (a11*tx - a31, -a22*ty - a32))
// it becomes an ellipse centered at (-a31, -a32) with half-axes (a11*r, a22*r).
// We mask the annulus between that ellipse and the eye-target rectangle edge.
// Skipped on the dummy/desktop HMD, which has no real lens.
static void buildHiddenAreaMesh(struct xrt_hmd_parts* hmd) {
  if (!isDummyHMD) {
    for (uint32_t eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
      const struct xrt_fov& fov = hmd->distortion.fov[eyeIndex];
      const float tanLeft = tanf(fov.angle_left), tanRight = tanf(fov.angle_right);
      const float tanDown = tanf(fov.angle_down), tanUp = tanf(fov.angle_up);
      const float tanWidth = tanRight - tanLeft;
      const float tanHeight = tanUp - tanDown;
      const float radiusTangent = std::min(std::min(-tanLeft, tanRight), std::min(-tanDown, tanUp));
      if (!(tanWidth > 0.0f) || !(tanHeight > 0.0f) || !(radiusTangent > 0.0f)) {
        printf("buildHiddenAreaMesh: degenerate FOV for eye %u; skipping aperture mask.\n", eyeIndex);
        continue;
      }

      const float a11 = 2.0f / tanWidth;
      const float a22 = 2.0f / tanHeight;
      const float a31 = (tanRight + tanLeft) / tanWidth;
      const float a32 = (tanUp + tanDown) / tanHeight;

      const glm::vec2 center(-a31, -a32); // boresight in NDC
      const glm::vec2 radii(a11 * radiusTangent, a22 * radiusTangent); // ellipse half-axes in NDC

      // Walk the eye-target rectangle ([-1,1] NDC) perimeter -- corners included,
      // so the masked corners are covered exactly. For each boundary point, cast a
      // ray inward to the ellipse center and clip it to the ellipse to get the
      // matching inner-ring point.
      const uint32_t edgeSubdivisions = 64;
      const glm::vec2 corners[4] = {glm::vec2(-1.0f, -1.0f), glm::vec2(1.0f, -1.0f), glm::vec2(1.0f, 1.0f), glm::vec2(-1.0f, 1.0f)};
      std::vector<glm::vec2> rectRing;
      rectRing.reserve(4 * edgeSubdivisions);
      for (uint32_t edge = 0; edge < 4; ++edge) {
        const glm::vec2& a = corners[edge];
        const glm::vec2& b = corners[(edge + 1) & 3];
        for (uint32_t j = 0; j < edgeSubdivisions; ++j)
          rectRing.push_back(a + ((b - a) * (static_cast<float>(j) / static_cast<float>(edgeSubdivisions))));
      }

      std::vector<glm::vec2> strip;
      strip.reserve((rectRing.size() + 1) * 2);
      for (size_t k = 0; k <= rectRing.size(); ++k) {
        const glm::vec2& outer = rectRing[k % rectRing.size()];
        const glm::vec2 d = outer - center;
        // Scale d down onto the ellipse boundary: q is how many ellipse-radii out
        // the rectangle point sits, so center + d/q lands on the ellipse.
        const float q = sqrtf(((d.x / radii.x) * (d.x / radii.x)) + ((d.y / radii.y) * (d.y / radii.y)));
        const glm::vec2 inner = (q > 1.0f) ? (center + (d / q)) : outer; // q >= 1 since the ellipse is inside the rectangle
        strip.push_back(inner);
        strip.push_back(outer);
      }

      apertureMaskVertexBuffer[eyeIndex] = rhi()->newBufferWithContents(strip.data(), strip.size() * sizeof(glm::vec2));
      apertureMaskVertexCount[eyeIndex] = static_cast<uint32_t>(strip.size());
    }
    printf("Aperture mask: %u strip verts/eye.\n", apertureMaskVertexCount[0]);
  }

  // clang-format off
  hiddenAreaMeshPipeline = rhi()->compileRenderPipeline(
    "shaders/hiddenAreaMesh.vtx.glsl",
    "shaders/hiddenAreaMesh.frag.glsl",
    RHIVertexLayout({ RHIVertexLayoutElement(0, kVertexElementTypeFloat2, "ndcPosition", 0, sizeof(glm::vec2)) }),
    kPrimitiveTopologyTriangleStrip);
  // clang-format on
}

bool RenderInit(ERenderBackend backendType) {
  // Monado setup -- this needs to occur before EGL initialization because we might need to send a command to turn on the HMD display.
  struct xrt_hmd_parts* hmd = NULL;
  {
    // vive_watchman_enable = false; // Skip Watchman initialization, we don't (can't) use lighthouse tracking here.

    int ret;

    ret = xrt_instance_create(NULL, &xrtInstance);
    if (ret != 0) {
      printf("xrt_instance_create() failed: %d\n", ret);
      return false;
    }

    ret = xrt_instance_create_system(xrtInstance, &xrtSystemDevices, &xrtSpaceOverseer, /*compositor=*/ NULL);
    if (ret != 0) {
      printf("xrt_instance_create_system() failed: %d\n", ret);
      return false;
    }

    if (xrtSystemDevices->roles.head != NULL && xrtSystemDevices->roles.head->device_type == XRT_DEVICE_TYPE_HMD && xrtSystemDevices->roles.head->hmd != nullptr) {
      xrtHMDevice = xrtSystemDevices->roles.head;
    } else {
      // Fallback to selecting the first enumerated HMD
      for (size_t i = 0; i < xrtSystemDevices->xdev_count; i++) {
        if (xrtSystemDevices->xdevs[i] == NULL) {
          continue;
        }

        if (xrtHMDevice == NULL && xrtSystemDevices->xdevs[i]->device_type == XRT_DEVICE_TYPE_HMD && xrtSystemDevices->xdevs[i]->hmd != nullptr) {
          printf("Selected HMD device: %s\n", xrtSystemDevices->xdevs[i]->str);
          xrtHMDevice = xrtSystemDevices->xdevs[i];
          break;
        }
      }
    }

    hmd = xrtHMDevice->hmd;
    assert(hmd);
    if (strstr(xrtHMDevice->str, "Simulated HMD")) {
      isDummyHMD = true;
    }

    // Dump HMD info
    printf("HMD screen: %d x %d, %lu ns nominal frame interval (%.3f FPS)\n", hmd->screens[0].w_pixels, hmd->screens[0].h_pixels, hmd->screens[0].nominal_frame_interval_ns, 1000000000.0 / static_cast<double>(hmd->screens[0].nominal_frame_interval_ns));
    printf("Viewports:\n");
    for (int viewportIdx = 0; viewportIdx < 2; ++viewportIdx) {
      printf("[%d] %u x %u pixels @ %u, %u\n", viewportIdx, hmd->views[viewportIdx].viewport.w_pixels, hmd->views[viewportIdx].viewport.h_pixels, hmd->views[viewportIdx].viewport.x_pixels, hmd->views[viewportIdx].viewport.y_pixels);
    }

    // Try to find the head presence detect input
    for (size_t inputIdx = 0; inputIdx < xrtHMDevice->input_count; ++inputIdx) {
      if (xrtHMDevice->inputs[inputIdx].name == XRT_INPUT_GENERIC_HEAD_DETECT) {
        printf("XRT_INPUT_GENERIC_HEAD_DETECT supported for proximity sensor.\n");
        xrtHeadDetectInput = &(xrtHMDevice->inputs[inputIdx]);
        break;
      }
    }

    // Setup global state
    hmd_width = hmd->screens[0].w_pixels;
    hmd_height = hmd->screens[0].h_pixels;
  }

  // Construct RenderBackendVKDirect for native Vulkan rendering. earlyInit is a no-op
  // on VKDirect, createGLContext becomes a no-op, createPresentation skips
  // VKGLSyncData and uses RHIWindowRenderTargetVK.
  auto* vkBackend = new RenderBackendVKDirect();
  renderBackend = vkBackend;
  vkBackend->earlyInit();
  vkBackend->createGLContext(); // Only creates an EGL context.

  initRHIVulkan();

  vkBackend->createPresentation();
  windowRenderTarget = renderBackend->windowRenderTarget();

  // Attach the swap-frame source to RHIVK so it can drive the per-frame
  // command buffer + acquire/present cycle.
  static_cast<RHIVK*>(rhi())->setFrameSource(vkBackend);


  // Set up shared resources

  solidQuadPipeline = rhi()->compileRenderPipeline("shaders/solidQuad.vtx.glsl", "shaders/solidQuad.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);

  {
    // clang-format off
    RHIVertexLayout vtx({
      RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "position_Ruv", 0, 16)
    });
    RHIShaderDescriptor desc(
      "shaders/meshDistortion.vtx.glsl",
      "shaders/meshDistortion.frag.glsl",
      vtx);
    desc.setFlag("CHROMA_CORRECTION", false);

    mesh1chDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
    // clang-format on
  }

  {
    // clang-format off
    RHIVertexLayout vtx({
      RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "position_Ruv",  0, 32),
      RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "Guv_Buv",      16, 32),
    });

    RHIShaderDescriptor desc(
      "shaders/meshDistortion.vtx.glsl",
      "shaders/meshDistortion.frag.glsl",
      vtx);
    desc.setFlag("CHROMA_CORRECTION", true);
    mesh3chDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
    // clang-format on
  }

  {
    uint8_t* maskData = new uint8_t[8 * 8];
    memset(maskData, 0xff, 8 * 8);
    disabledMaskTex = rhi()->newTexture2D(8, 8, RHISurfaceDescriptor(kSurfaceFormat_R8));
    rhi()->loadTextureData(disabledMaskTex, kVertexElementTypeUByte1N, maskData);
    delete[] maskData;
  }


  // Set up distortion models
  {
    printf("Distortion models: %s%s%s\n",
      hmd->distortion.models & XRT_DISTORTION_MODEL_NONE ? "None " : "",
      hmd->distortion.models & XRT_DISTORTION_MODEL_MESHUV ? "MeshUV " : "",
      hmd->distortion.models & XRT_DISTORTION_MODEL_COMPUTE ? "Compute " : "");

    if (!(hmd->distortion.models & XRT_DISTORTION_MODEL_MESHUV)) {
      if (!((hmd->distortion.models & XRT_DISTORTION_MODEL_NONE) || (hmd->distortion.models & XRT_DISTORTION_MODEL_COMPUTE))) {
        printf("HMD does not report any usable distortion models (MeshUV, Compute, or None)\n");
        return false;
      }
      printf("Generating HMD MeshUV distortion from Compute function\n");
      u_distortion_mesh_fill_in_compute(xrtHMDevice);
    }

    printf("Distortion mesh data:\n");
    printf("vertices=%u stride=%u uv_channels_count=%u index_counts={%u, %u} index_offsets={%u, %u} index_count_total=%u\n",
      hmd->distortion.mesh.vertex_count, hmd->distortion.mesh.stride, hmd->distortion.mesh.uv_channels_count,
      hmd->distortion.mesh.index_counts[0], hmd->distortion.mesh.index_counts[1],
      hmd->distortion.mesh.index_offsets[0], hmd->distortion.mesh.index_offsets[1],
      hmd->distortion.mesh.index_count_total);

    // Upload vertex and index buffers for distortion
    meshDistortionVertexBuffer = rhi()->newBufferWithContents(hmd->distortion.mesh.vertices, hmd->distortion.mesh.vertex_count * hmd->distortion.mesh.stride);
    meshDistortionIndexBuffer = rhi()->newBufferWithContents(hmd->distortion.mesh.indices, hmd->distortion.mesh.index_count_total * sizeof(uint32_t));

    // Compute post-distortion viewports
    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
      eyePostDistortionViewports[eyeIndex] = RHIRect::xywh(
        hmd->views[eyeIndex].viewport.x_pixels,
        hmd->views[eyeIndex].viewport.y_pixels,
        hmd->views[eyeIndex].viewport.w_pixels,
        hmd->views[eyeIndex].viewport.h_pixels);
    }

    // Eye target dimensions are 1.5x the per-eye viewport resolution, rounded up to the next 16 pixel block
    eye_width = (((hmd->views[0].viewport.w_pixels * 3) / 2) + 0xf) & ~0xfUL;
    eye_height = (((hmd->views[0].viewport.h_pixels * 3) / 2) + 0xf) & ~0xfUL;
    printf("Eye target dimensions: %u x %u\n", eye_width, eye_height);

  } // Monado distortion setup

  // Set up uniform buffers for HMD distortion passes
  recomputeHMDParameters();

  printf("Screen dimensions: %u x %u\n", windowRenderTarget->width(), windowRenderTarget->height());
  if (isDummyHMD) {
    // Resize the dummy HMD eye RTs to match the attached display.
    hmd_width = windowRenderTarget->width();
    hmd_height = windowRenderTarget->height();
    // Use 1:1 eye targets since we have no distortion to compensate for
    eye_width = hmd_width / 2;
    eye_height = hmd_height;
    // Reset the viewports
    // clang-format off
    eyePostDistortionViewports[0] = RHIRect::xywh(        0, 0, eye_width, eye_height);
    eyePostDistortionViewports[1] = RHIRect::xywh(eye_width, 0, eye_width, eye_height);
    // clang-format on
    printf("Dummy HMD Eye target dimensions: %u x %u\n", eye_width, eye_height);
  }

  if (!(windowRenderTarget->width() == hmd_width && windowRenderTarget->height() == hmd_height)) {
    printf("WARNING: Screen and HMD dimensions don't match; check system configuration.\n");
  }

  // Create FBOs and viewports for eye rendering (pre distortion)
  eyeTex = rhi()->newTexture2D(eye_width * 2, eye_height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  eyeDepthRenderbuffer = rhi()->newRenderbuffer2D(eye_width * 2, eye_height, RHISurfaceDescriptor(kSurfaceFormat_Depth32f));
  eyeRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({eyeTex}, eyeDepthRenderbuffer));
  // clang-format off
  eyeViewports[0] = RHIRect::xywh(        0, 0, eye_width, eye_height);
  eyeViewports[1] = RHIRect::xywh(eye_width, 0, eye_width, eye_height);
  // clang-format on

  // Precompute the hidden-area mesh from the distortion grid.
  buildHiddenAreaMesh(xrtHMDevice->hmd);

  return true;
}

void RenderShutdown() {
  // Release OpenGL resources
  delete renderBackend;
  renderBackend = NULL;

  if (xrtSystemDevices)
    xrt_system_devices_destroy(&xrtSystemDevices); // also destroys owned devices

  if (xrtSpaceOverseer)
    xrt_space_overseer_destroy(&xrtSpaceOverseer);

  if (xrtInstance)
    xrt_instance_destroy(&xrtInstance);
}

void recomputeHMDParameters() {
  float zNear = 0.005f;

  // from renderer_get_view_projection (compositor/main/comp_renderer.c)
  struct xrt_vec3 eye_relation = {
    0.063000f, /* TODO: get actual ipd_meters */
    0.0f,
    0.0f,
  };

  for (uint32_t eyeIdx = 0; eyeIdx < 2; eyeIdx++) {
    // clang-format off
    struct xrt_fov* fov = &xrtHMDevice->hmd->distortion.fov[eyeIdx];

    // from comp_layer_renderer_set_fov
    const float tan_left = tanf(fov->angle_left);
    const float tan_right = tanf(fov->angle_right);

    const float tan_down = tanf(fov->angle_down);
    const float tan_up = tanf(fov->angle_up);

    const float tan_width = tan_right - tan_left;
    const float tan_height = tan_up - tan_down;

    const float a11 = 2.0f / tan_width;
    const float a22 = 2.0f / tan_height;

    const float a31 = (tan_right + tan_left) / tan_width;
    const float a32 = (tan_up + tan_down) / tan_height;
    
    /*
    self->mat_projection[eye] = (struct xrt_matrix_4x4) {
      .v = {
        a11, 0, 0, 0,
        0, a22, 0, 0,
        a31, a32, a33, -1,
        0, 0, a43, 0,
      }
    };*/

    // Right-handed infinite-Z far plane, Vulkan Y-down NDC, reverse-Z [0,1].
    eyeProjection[eyeIdx] = glm::mat4(
       a11,  0.0f,  0.0f,   0.0f,
      0.0f,  -a22,  0.0f,   0.0f,
       a31,   a32,  0.0f,  -1.0f,
      0.0f,  0.0f,  zNear,  0.0f);

    struct xrt_pose eye_pose;
    u_device_get_view_pose(&eye_relation, eyeIdx, &eye_pose);

    xrt_matrix_4x4 eye_view;
    math_matrix_4x4_view_from_pose(&eye_pose, &eye_view);

    const float* v = eye_view.v;
    eyeView[eyeIdx] = glm::mat4(
      v[ 0], v[ 1], v[ 2], v[ 3],
      v[ 4], v[ 5], v[ 6], v[ 7],
      v[ 8], v[ 9], v[10], v[11],
      v[12], v[13], v[14], v[15]);
    // clang-format on
  }

  for (size_t i = 0; i < 2; ++i) {
    // clang-format off
    printf("Eye %zu projection matrix:\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n\n", i,
      eyeProjection[i][0][0], eyeProjection[i][0][1], eyeProjection[i][0][2], eyeProjection[i][0][3],
      eyeProjection[i][1][0], eyeProjection[i][1][1], eyeProjection[i][1][2], eyeProjection[i][1][3],
      eyeProjection[i][2][0], eyeProjection[i][2][1], eyeProjection[i][2][2], eyeProjection[i][2][3],
      eyeProjection[i][3][0], eyeProjection[i][3][1], eyeProjection[i][3][2], eyeProjection[i][3][3]);
    // clang-format on
  }
}

void RenderBeginHMDEyeTargetRenderPass() {
  rhi()->setClearDepth(0.0f);
  rhi()->beginRenderPass(eyeRT, kLoadClear);
  rhi()->setViewports(eyeViewports, 2);

  // Apply the hidden-area mesh: write near-plane depth (1.0, reverse-Z) into the
  // border region the distortion pass never samples. The scene then renders with
  // standardGreaterDepthStencilState, so its fragments are rejected there (depth >
  // 1.0 is unsatisfiable), saving shading and bandwidth. standardGreaterDepthStencilState
  // here lets the mesh's 1.0 beat the 0.0 depth clear while writing depth.
  if (apertureMaskVertexCount[0] || apertureMaskVertexCount[1]) {
    rhi()->bindRenderPipeline(hiddenAreaMeshPipeline);
    rhi()->bindDepthStencilState(standardGreaterDepthStencilState);
    rhi()->bindBlendState(disabledBlendState);
    rhi()->setCullState(kCullDisabled); // strip winding alternates around the annulus
    // Depth-only: only the near-plane depth write matters; the fragment shader's
    // red output is debug visualization. Disabling color writes skips that color
    // traffic. Flip to true to see the masked region painted red again.
    rhi()->setColorWriteEnabled(false);
    for (uint32_t eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
      rhi()->setViewport(eyeViewports[eyeIndex]);
      rhi()->bindStreamBuffer(0, apertureMaskVertexBuffer[eyeIndex]);
      rhi()->drawPrimitives(0, apertureMaskVertexCount[eyeIndex]);
    }
    // Restore color writes for the scene geometry that renders into eyeRT next.
    rhi()->setColorWriteEnabled(true);
  }
}

void renderHMDFrame() {
  // Record the distortion pass into the current frame's command buffer.
  // Caller is responsible for calling rhi()->swapBuffers(windowRenderTarget)
  // after this returns (and after any post-distortion bookkeeping like
  // ending a distortion timer query, which must happen before swap so its
  // timestamp write lands in the CB that swap submits).
  rhi()->beginRenderPass(windowRenderTarget, kLoadInvalidate);

  if (xrtHMDevice->hmd->distortion.mesh.uv_channels_count == 1) {
    rhi()->bindRenderPipeline(mesh1chDistortionPipeline);
  } else {
    rhi()->bindRenderPipeline(mesh3chDistortionPipeline);
  }

  rhi()->bindStreamBuffer(0, meshDistortionVertexBuffer);
  rhi()->loadTexture(ksImageTex, eyeTex, linearClampSampler);

  // Run distortion passes
  for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {

    rhi()->setViewport(eyePostDistortionViewports[eyeIndex]);

    MeshDistortionUniformBlock ub;
    ub.uvOffset = glm::vec2(eyeIndex == 0 ? 0.0f : 0.5f, 0.0f);
    ub.uvScale = glm::vec2(0.5f, 1.0f);
    rhi()->loadUniformBlockImmediate(ksMeshDistortionUniformBlock, &ub, sizeof(ub));

    rhi()->drawIndexedPrimitives(meshDistortionIndexBuffer, kIndexBufferTypeUInt32, xrtHMDevice->hmd->distortion.mesh.index_counts[eyeIndex], xrtHMDevice->hmd->distortion.mesh.index_offsets[eyeIndex]);
  }

  rhi()->endRenderPass(windowRenderTarget);
}
