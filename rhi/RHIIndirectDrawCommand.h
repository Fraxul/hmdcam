#pragma once
#include <stdint.h>

struct RHIDrawIndexedIndirectCommand {
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t baseIndex;
  int32_t vertexOffset;
  uint32_t baseInstance;
};

struct RHIDrawIndirectCommand {
  uint32_t vertexCount;
  uint32_t instanceCount;
  uint32_t baseVertex;
  uint32_t baseInstance;
};
