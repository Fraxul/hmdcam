#pragma once
#include "RDMABuffer.h"
#include "RDMAConnection.h"
#include "SerializationBuffer.h"

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <set>
#include <vector>

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <boost/noncopyable.hpp>

struct ibv_context;
struct ibv_pd;
struct ibv_cq;
struct ibv_comp_channel;
struct rdma_cm_id;
struct rdma_event_channel;

class RDMAContext {
public:
  static RDMAContext* createServerContext();
  static RDMAContext* createClientContext(const char* host);

  RDMABuffer::ptr newCUDAManagedBuffer(const std::string& key, size_t length, RDMABufferUsage usage);
  RDMABuffer::ptr newManagedBuffer(const std::string& key, size_t length, RDMABufferUsage usage);
  RDMABuffer::ptr newUnmanagedBuffer(const std::string& key, char* data, size_t length, RDMABufferUsage usage);

  void asyncFlushWriteBuffer(RDMABuffer::ptr);
  void asyncSendUserEvent(uint32_t userEventID, SerializationBuffer payload);


  bool hasPeerConnections() const { return !(m_peerConnections.empty()); }


  void fireUserEvents(); // Call from whatever thread the event callback function needs to run on

  void setUserEventCallback(std::function<void(RDMAContext*, uint32_t, SerializationBuffer)> callbackFn) { m_userEventCallbackFn = callbackFn; }

  struct ibv_context* m_ctx;
  struct ibv_pd* m_pd;
  struct ibv_cq* m_cq;
  struct ibv_comp_channel* m_comp_channel;

protected:
  RDMAContext(struct ibv_context*);
  friend class RDMAConnection;

  void processMessage(RDMAConnection* source, uint32_t opcode, SerializationBuffer payload);
  void internalRegisterBuffer(RDMABuffer::ptr buf);
  void sendBufferAdvertisement(RDMABuffer::ptr buf);
  void broadcastMessage(uint32_t opcode, const SerializationBuffer&);
  bool internalFlushWriteBuffer(RDMABuffer::ptr src, RDMAConnection* destConn, const RDMAPublishedBufferInfo& destBuffer);

  std::function<void(RDMAContext*, uint32_t, SerializationBuffer)> m_userEventCallbackFn;

  struct UserEvent_t {
    UserEvent_t() {}
    UserEvent_t(uint32_t i, const SerializationBuffer& b) : userEventID(i), payload(b) {}

    uint32_t userEventID;
    SerializationBuffer payload;
  };

  std::vector<UserEvent_t> m_queuedUserEvents;
  std::mutex m_userEventQueueLock;


  std::set<RDMAConnection::ptr> m_peerConnections;
  std::mutex m_peerConnectionsLock;

  static void* cqPollThreadEntryPoint(void* x) { reinterpret_cast<RDMAContext*>(x)->cqPollThreadFn(); return NULL; }
  void cqPollThreadFn();
  pthread_t m_cqPollThread;

  // client parts
  RDMAConnection::ptr m_clientConnection;
  struct rdma_cm_id* m_server;
  struct rdma_event_channel* m_clientEventChannel;
  pthread_t m_clientEventThread;
  static void* clientEventThreadEntryPoint(void* x) { reinterpret_cast<RDMAContext*>(x)->clientEventThreadFn(); return NULL; }
  void clientEventThreadFn();

  // server parts
  void initServer();
  static void* listenerThreadEntryPoint(void* x) { reinterpret_cast<RDMAContext*>(x)->listenerThreadFn(); return NULL; }
  void listenerThreadFn();
  pthread_t m_listenerThread;
  struct rdma_cm_id* m_listener;
  struct rdma_event_channel* m_serverEventChannel;

  // ---

  std::map<std::string, RDMABuffer::ptr> m_buffers;
};

