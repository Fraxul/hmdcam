#include "rhi/RHIBuffer.h"

RHIBuffer::RHIBuffer() : m_data(NULL), m_size(0), m_usageMode(kBufferUsageGPUPrivate) {

}

RHIBuffer::~RHIBuffer() {
  assert(m_data == NULL && "buffer mapped at destruction");
}


