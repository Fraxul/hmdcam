#include "rhi/RHISurface.h"

size_t rhiSurfaceFormatSize(RHISurfaceFormat format) {
  switch (format) {
    case kSurfaceFormat_Invalid:
      assert(false && "rhiSurfaceFormatSize: kSurfaceFormat_Invalid");

    default:
    case kSurfaceFormat_Depth32f_Stencil8: // should never be uploading this type
      assert(false && "rhiSurfaceFormatSize: unhandled enum");

    case kSurfaceFormat_R8:
    case kSurfaceFormat_R8i:
    case kSurfaceFormat_R8ui:
    case kSurfaceFormat_Stencil8:
      return 1;

    case kSurfaceFormat_R16:
    case kSurfaceFormat_R16f:
    case kSurfaceFormat_R16i:
    case kSurfaceFormat_R16ui:
    case kSurfaceFormat_RG8i:
    case kSurfaceFormat_RG8ui:
    case kSurfaceFormat_Depth16:
      return 2;

    case kSurfaceFormat_sRGB8_A8:
    case kSurfaceFormat_RGBA8:
    case kSurfaceFormat_RGB10_A2:
    case kSurfaceFormat_RG16:
    case kSurfaceFormat_R32f:
    case kSurfaceFormat_R32i:
    case kSurfaceFormat_R32ui:
    case kSurfaceFormat_RG16i:
    case kSurfaceFormat_RG16ui:
    case kSurfaceFormat_RGBA8i:
    case kSurfaceFormat_RGBA8ui:
    case kSurfaceFormat_Depth32f:
      return 4;

    case kSurfaceFormat_RGB16f:
    case kSurfaceFormat_RGB16s:
      return 6;

    case kSurfaceFormat_RGBA16f:
    case kSurfaceFormat_RGBA16s:
    case kSurfaceFormat_RG32f:
    case kSurfaceFormat_RG32i:
    case kSurfaceFormat_RG32ui:
    case kSurfaceFormat_RGBA16i:
    case kSurfaceFormat_RGBA16ui:
      return 8;

    case kSurfaceFormat_RGBA32i:
    case kSurfaceFormat_RGBA32ui:
      return 16;
  };
}

bool rhiSurfaceFormatHasDepth(RHISurfaceFormat format) {
  switch (format) {
    case kSurfaceFormat_Depth16:
    case kSurfaceFormat_Depth32f:
    case kSurfaceFormat_Depth32f_Stencil8:
      return true;

    default:
      return false;
  }
}

bool rhiSurfaceFormatHasStencil(RHISurfaceFormat format) {
  switch (format) {
    case kSurfaceFormat_Depth32f_Stencil8:
    case kSurfaceFormat_Stencil8:
      return true;

    default:
      return false;
  }
}

RHISurface::~RHISurface() {

}

RHISampler::~RHISampler() {

}

