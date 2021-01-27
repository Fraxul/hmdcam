#pragma once
#include "SerializationBuffer.h"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <boost/noncopyable.hpp>
#include <map>

class RDMAContext;
struct ibv_qp;
struct rdma_cm_id;
struct ibv_mr;

struct RDMAPublishedBufferInfo {
  RDMAPublishedBufferInfo() : remote_addr(0), remote_key(0) {}
  RDMAPublishedBufferInfo(uint64_t remote_addr_, uint32_t remote_key_) : remote_addr(remote_addr_), remote_key(remote_key_) {}

  uint64_t remote_addr;
  uint32_t remote_key;
};


class RDMAConnection : public boost::intrusive_ref_counter<RDMAConnection>, public boost::noncopyable {
public:
  typedef boost::intrusive_ptr<RDMAConnection> ptr;

  RDMAConnection(RDMAContext* ctx, struct rdma_cm_id* id);

  virtual ~RDMAConnection();

  bool sendMessage(uint32_t opcode, SerializationBuffer payload);

  void didEstablishConnection();
  void didCompleteReceive(size_t bytesReceived);
  void didCompleteSend();
  void postReceives();

  RDMAContext* context() const { return m_ctx; }

  RDMAContext* m_ctx;
  struct ibv_qp* m_qp;
  struct rdma_cm_id* m_id;

  struct ibv_mr* m_recvMr;

  size_t m_bufferSize;
  char* m_recvBuffer;

  std::map<std::string, RDMAPublishedBufferInfo> m_publishedBuffers;
protected:
  bool internalSendMessage(const char* payload, size_t length);
};

