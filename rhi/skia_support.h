#pragma once
#include "rhi/RHIShader.h"
#include <SkColorPriv.h>

static inline RHIVertexElementType SkN32TexelFormat() {
#if defined(SK_PMCOLOR_IS_RGBA)
  return kVertexElementTypeUByte4N;
#elif defined(SK_PMCOLOR_IS_BGRA)
  #error SKN32TexelFormat(): BGRA is not supported on ES3 platform
  //return kVertexElementTypeUByte4N_BGRA;
#else
  #error SKN32TexelFormat(): SK_PMCOLOR_IS_... macro not defined
#endif
}

