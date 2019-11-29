#include "rhi/RHISurface.h"

size_t rhiSurfaceFormatSize(RHISurfaceFormat format) {
  switch (format) {
    case kSurfaceFormat_Invalid:
      assert(false && "rhiSurfaceFormatSize: kSurfaceFormat_Invalid");

    default:
    case kSurfaceFormat_Depth32f_Stencil8: // should never be uploading this type
      assert(false && "rhiSurfaceFormatSize: unhandled enum");

    case kSurfaceFormat_R8:
    case kSurfaceFormat_Stencil8:
      return 1;

    case kSurfaceFormat_R16f:
      return 2;

    case kSurfaceFormat_sRGB8_A8:
    case kSurfaceFormat_RGBA8:
    case kSurfaceFormat_RGB10_A2:
    case kSurfaceFormat_R32f:
    case kSurfaceFormat_Depth32f:
      return 4;

    case kSurfaceFormat_RGB16f:
    case kSurfaceFormat_RGB16s:
      return 6;

    case kSurfaceFormat_RGBA16f:
    case kSurfaceFormat_RGBA16s:
      return 8;
  };
}

bool rhiSurfaceFormatHasDepth(RHISurfaceFormat format) {
  switch (format) {
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

