#pragma once
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <boost/noncopyable.hpp>
#include <infiniband/verbs.h>

class RDMAConnection;
class RDMAContext;
struct ibv_mr;

enum RDMABufferUsage {
  kRDMABufferUsageWriteSource,
  kRDMABufferUsageWriteDestination,
  kRDMABufferUsageReadSource,
  kRDMABufferUsageReadDestination
};

class RDMABuffer : public boost::intrusive_ref_counter<RDMABuffer>, public boost::noncopyable {
public:
  typedef boost::intrusive_ptr<RDMABuffer> ptr;


  RDMABuffer(RDMAContext* ctx, const std::string& key, char* bufferData, size_t bufferSize, RDMABufferUsage usage);


  virtual ~RDMABuffer();

  RDMAContext* context() const { return m_context; }

  char* data() const { return m_bufferData; }
  size_t size() const { return m_bufferSize; }

  const std::string& key() const { return m_key; }

  uint32_t local_key() const { return m_mr->lkey; }
  uint64_t remote_addr() const { return reinterpret_cast<uint64_t>(m_mr->addr); }
  uint32_t remote_key() const { return m_mr->rkey; }

  RDMABufferUsage usage() const { return m_usage; }

protected:
  friend class RDMAContext;
  RDMAContext* m_context;

  std::string m_key;

  char* m_bufferData;
  size_t m_bufferSize;

  struct ibv_mr* m_mr;
  RDMABufferUsage m_usage;
};

class RDMAManagedBuffer : public RDMABuffer {
public:
  RDMAManagedBuffer(RDMAContext*, const std::string& key, size_t bufferSize, RDMABufferUsage);
  virtual ~RDMAManagedBuffer();
};

class RDMACUDAManagedBuffer : public RDMABuffer {
public:
  RDMACUDAManagedBuffer(RDMAContext*, const std::string& key, size_t bufferSize, RDMABufferUsage);
  virtual ~RDMACUDAManagedBuffer();
};

