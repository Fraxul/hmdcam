#include "rhi/RHIBuffer.h"

RHIBuffer::~RHIBuffer() {
  assert(m_data == NULL && "buffer mapped at destruction");
}
