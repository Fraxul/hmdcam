#include "RDMAContext.h"
#include "RDMABuffer.h"
#include "RDMAConnection.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <dirent.h>



#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

static const uint16_t rdmaPort = 55443;
static const int rdmaConnectionTimeoutMs = 1000;

enum ContextOpcode {
  OP_ADVERTISE_BUFFERS = 1,
  OP_USER_EVENT = 2
};


RDMAContext::RDMAContext(struct ibv_context* verbs) : m_ctx(verbs) {
  m_pd = ibv_alloc_pd(m_ctx);
  m_comp_channel = ibv_create_comp_channel(m_ctx);
  m_cq = ibv_create_cq(m_ctx, 32, NULL, m_comp_channel, 0);
  TEST_NZ(ibv_req_notify_cq(m_cq, 0));

  pthread_create(&m_cqPollThread, NULL, &cqPollThreadEntryPoint, static_cast<void*>(this));
}

RDMABuffer::ptr RDMAContext::newCUDAManagedBuffer(const std::string& key, size_t length, RDMABufferUsage usage) {
  RDMABuffer::ptr buf = new RDMACUDAManagedBuffer(this, key, length, usage);
  internalRegisterBuffer(buf);
  return buf;
}

RDMABuffer::ptr RDMAContext::newManagedBuffer(const std::string& key, size_t length, RDMABufferUsage usage) {
  RDMABuffer::ptr buf = new RDMAManagedBuffer(this, key, length, usage);
  internalRegisterBuffer(buf);
  return buf;
}

RDMABuffer::ptr RDMAContext::newUnmanagedBuffer(const std::string& key, char* data, size_t length, RDMABufferUsage usage) {
  RDMABuffer::ptr buf = new RDMABuffer(this, key, data, length, usage);
  internalRegisterBuffer(buf);
  return buf;
}

void RDMAContext::internalRegisterBuffer(RDMABuffer::ptr buf) {
  RDMABuffer::ptr& bp = m_buffers[buf->key()];
  if (bp.get()) {
    die("RDMAContext::newUnmanagedBuffer(): buffer with key \"%s\" already exists", buf->key().c_str());
  }
  bp = buf;

  int ibv_access = 0;
  switch (buf->usage()) {
    case kRDMABufferUsageWriteSource:       ibv_access = IBV_ACCESS_LOCAL_WRITE;  break;
    case kRDMABufferUsageWriteDestination:  ibv_access = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE; break;
    case kRDMABufferUsageReadSource:        ibv_access = IBV_ACCESS_REMOTE_READ;  break;
    case kRDMABufferUsageReadDestination:   ibv_access = IBV_ACCESS_LOCAL_WRITE;  break; // TODO not sure if this is correct
    default: assert(false && "invalid RDMABufferUsage");
  };
  buf->m_mr = ibv_reg_mr(m_pd, buf->data(), buf->size(), ibv_access);
  assert(buf->m_mr);

  if (buf->usage() == kRDMABufferUsageWriteDestination || buf->usage() == kRDMABufferUsageReadSource) {
    // Advertise network-accessible buffers
    sendBufferAdvertisement(buf);
  }
}

bool RDMAContext::internalFlushWriteBuffer(RDMABuffer::ptr src, RDMAConnection* destConn, const RDMAPublishedBufferInfo& destBuffer) {
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));

  wr.wr_id = 0; // (uintptr_t)id;

  sge.addr = (uintptr_t) src->data();
  sge.length = src->size();
  sge.lkey = src->local_key();

  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.wr.rdma.remote_addr = destBuffer.remote_addr;
  wr.wr.rdma.rkey = destBuffer.remote_key;

  int res = ibv_post_send(destConn->m_qp, &wr, &bad_wr);
  if (res != 0) {
    printf("RDMAContext(%p)::internalFlushWriteBuffer(buffer %p : \"%.*s\"): ibv_post_send(destConn = %p) failed: %s\n",
      this, src.get(), (int) src->key().size(), src->key().c_str(), destConn, strerror(res));
    return false;
  }

  return true;
}

void RDMAContext::asyncFlushWriteBuffer(RDMABuffer::ptr buf) {
  // For each peer connection, check to see if we have previously matched a remote buffer sharing the same key, and issue an RDMA write
  std::lock_guard<std::mutex> guard(m_peerConnectionsLock);

  for (auto conn_it = m_peerConnections.begin(); conn_it != m_peerConnections.end(); ++conn_it) {
    RDMAConnection* c = conn_it->get();
    auto buf_it = c->m_publishedBuffers.find(buf->key());
    if (buf_it != c->m_publishedBuffers.end()) {
      const RDMAPublishedBufferInfo& remoteBuf = buf_it->second;


      if (!internalFlushWriteBuffer(buf, c, remoteBuf)) {
        printf("RDMAContext(%p)::asyncFlushWriteBuffer(buf=%p) to peer connection %p remote_addr 0x%lx remote_key 0x%x failed\n",
          this, buf.get(), c, remoteBuf.remote_addr, remoteBuf.remote_key);
      }
    }
  }
}

void RDMAContext::asyncSendUserEvent(uint32_t userEventID, SerializationBuffer eventPayload) {
  SerializationBuffer b;
  b.put_u32(userEventID);
  b.put_buffer(eventPayload);

  broadcastMessage(OP_USER_EVENT, b);
}

void RDMAContext::sendBufferAdvertisement(RDMABuffer::ptr buf) {
  SerializationBuffer advPayload;
  advPayload.put_u32(1); // one buffer
  advPayload.put_u16_prefixed_string(buf->key());
  advPayload.put_u64(buf->remote_addr());
  advPayload.put_u32(buf->remote_key());

  broadcastMessage(OP_ADVERTISE_BUFFERS, advPayload);
}

void RDMAContext::broadcastMessage(uint32_t opcode, const SerializationBuffer& payload) {
  std::lock_guard<std::mutex> guard(m_peerConnectionsLock);
  for (auto conn_it = m_peerConnections.begin(); conn_it != m_peerConnections.end(); ++conn_it) {
    if (!(*conn_it)->sendMessage(opcode, payload)) {
      printf("RDMAContext(%p)::broadcastMessage() to peer connection %p failed\n",
        this, conn_it->get());
    }
  }
}

void RDMAContext::processMessage(RDMAConnection* source, uint32_t opcode, SerializationBuffer payload) {
  switch (opcode) {
    case OP_ADVERTISE_BUFFERS: {
      uint32_t bufferCount = payload.get_u32();
      for (uint32_t i = 0; i < bufferCount; ++i) {
        std::string key = payload.get_u16_prefixed_string();
        uint64_t remote_addr = payload.get_u64();
        uint32_t remote_key = payload.get_u32();
        // printf("RDMAContext: RDMAConnection(%p): received buffer advertisement key \"%s\" remote_addr=0x%lx remote_key=0x%x\n", source, key.c_str(), remote_addr, remote_key);
        source->m_publishedBuffers[key] = RDMAPublishedBufferInfo(remote_addr, remote_key);

        // Search for a matching local buffer
        auto buf_it = m_buffers.find(key);
        if (buf_it != m_buffers.end()) {
          RDMABuffer::ptr b = buf_it->second;
          if (b->usage() == kRDMABufferUsageWriteSource) {
            // Found a matching source-write buffer -- sync it to the client
            // printf("RDMAContext: RDMAConnection(%p): advertised buffer matched local write-source buffer, doing immediate flush\n", source);
            internalFlushWriteBuffer(b, source, RDMAPublishedBufferInfo(remote_addr, remote_key));
          }
        }
      }
    } break;

    case OP_USER_EVENT: {
      uint32_t userEventID = payload.get_u32();
      SerializationBuffer userEventPayload = payload.consume_buffer(payload.remaining());
      // printf("RDMAContext: RDMAConnection(%p): received user event type 0x%x with 0x%zx byte payload\n", source, userEventID, userEventPayload.size());

      // Queue user event for client
      {
        std::lock_guard<std::mutex> guard(m_userEventQueueLock);
        m_queuedUserEvents.push_back(UserEvent_t(userEventID, userEventPayload));
      }

    } break;

    default:
      printf("RDMAContext: RDMAConnection(%p): Unhandled message opcode 0x%x with %zu byte payload\n", source, opcode, payload.size());
  };
}

void RDMAContext::fireUserEvents() {
  if (!m_userEventCallbackFn) {
    die("RDMAContext::fireUserEvents(): no event callback function registered");
  }

  std::vector<UserEvent_t> q;

  {
    std::lock_guard<std::mutex> guard(m_userEventQueueLock);
    m_queuedUserEvents.swap(q);
  }

  for (size_t i = 0; i < q.size(); ++i) {
    m_userEventCallbackFn(this, q[i].userEventID, q[i].payload);
  }
}

void RDMAContext::cqPollThreadFn() {

  struct ibv_cq *cq;
  struct ibv_wc wc;
  void* ctx;

  while (1) {
    TEST_NZ(ibv_get_cq_event(m_comp_channel, &cq, &ctx));
    ibv_ack_cq_events(cq, 1);
    TEST_NZ(ibv_req_notify_cq(cq, 0));

    while (ibv_poll_cq(cq, 1, &wc)) {
      if (wc.status != IBV_WC_SUCCESS) {
        printf("CQ poll: bad event status %s\n", ibv_wc_status_str(wc.status));
      }

      if (wc.opcode & IBV_WC_RECV) {
        RDMAConnection* conn = reinterpret_cast<RDMAConnection*>(wc.wr_id);
        conn->didCompleteReceive(wc.byte_len);
      } else if (wc.opcode == IBV_WC_SEND) {
        RDMAConnection* conn = reinterpret_cast<RDMAConnection*>(wc.wr_id);
        conn->didCompleteSend();
      }

    }
  }
}


void RDMAContext::listenerThreadFn() {

  struct rdma_cm_event *ev = NULL;
  while (rdma_get_cm_event(m_serverEventChannel, &ev) == 0) {

    switch (ev->event) {
      case RDMA_CM_EVENT_CONNECT_REQUEST: {
          RDMAConnection* conn = new RDMAConnection(this, ev->id);
          // printf("rdmaListenerThread: accept created RDMAConnection(%p)\n", conn);
          rdma_accept(ev->id, NULL);
          {
            std::lock_guard<std::mutex> guard(m_peerConnectionsLock);
            m_peerConnections.insert(conn);
          }
          rdma_ack_cm_event(ev);

        } break;

      case RDMA_CM_EVENT_ESTABLISHED: {
          RDMAConnection* conn = reinterpret_cast<RDMAConnection*>(ev->id->context);
          conn->didEstablishConnection();
          rdma_ack_cm_event(ev);

        } break;

      case RDMA_CM_EVENT_DISCONNECTED: {
          RDMAConnection* conn = reinterpret_cast<RDMAConnection*>(ev->id->context);
          // printf("rdmaListenerThread: disconnect destroyed RDMAConnection(%p)\n", conn);
          // Event must be acknowledged before we can destroy the connection, otherwise it'll hang in rdma_destroy_id
          rdma_ack_cm_event(ev);
          {
            std::lock_guard<std::mutex> guard(m_peerConnectionsLock);
            m_peerConnections.erase(conn);
          }
        } break;

      default:
        printf("rdmaListenerThread: unhandled event type %s (%d)\n", rdma_event_str(ev->event), ev->event);
        rdma_ack_cm_event(ev);
        break;
    };

  }

  printf("rdmaListenerThread: Shutting down due to error response (%s) from rdma_get_cm_event\n", strerror(errno));

  rdma_destroy_id(m_listener);
  rdma_destroy_event_channel(m_serverEventChannel);
}

int ibv_port_active_width_to_multiplier(uint8_t active_width) {
  // LUT from  https://www.rdmamojo.com/2012/07/21/ibv_query_port/
  switch (active_width) {
    default:
    case 1: return 1;
    case 2: return 4;
    case 4: return 8;
    case 8: return 12;
  };
}

int ibv_port_active_speed_to_mbps(uint8_t active_speed) {
  // LUT from https://www.rdmamojo.com/2012/07/21/ibv_query_port/
  switch (active_speed) {
    default:
    case  1: return 2500;
    case  2: return 5000;
    case  4: return 10000;
    case  8: return 10000;
    case 16: return 14000;
    case 32: return 25000;
  };
}

float ibv_port_speed_gbps(uint8_t active_width, uint8_t active_speed) {
  return static_cast<float>(ibv_port_active_width_to_multiplier(active_width) * ibv_port_active_speed_to_mbps(active_speed)) / 1000.0;
}

/*static*/ RDMAContext* RDMAContext::createServerContext() {

  ibv_context* selectedDevice = NULL;
  {
    int deviceCount;
    struct ibv_context** devices = rdma_get_devices(&deviceCount);
    if (!deviceCount) {
      rdma_free_devices(devices);
      printf("No RDMA devices available\n");
      return NULL;
    }

    printf("RDMA devices (%d)\n", deviceCount);
    ibv_context* firstIBContext = NULL;
    ibv_context* firstActivePortsContext = NULL;

    for (int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
      struct ibv_context* dCtx = devices[deviceIdx];
      struct ibv_device_attr ibDeviceAttr;
      memset(&ibDeviceAttr, 0, sizeof(ibv_device_attr));
      ibv_query_device(dCtx, &ibDeviceAttr);

      printf("  [%d] %s (%#.16llx), %u ports:\n", deviceIdx, ibv_get_device_name(dCtx->device), ibv_get_device_guid(dCtx->device), ibDeviceAttr.phys_port_cnt);

      for (uint8_t ibPortIdx = 1; ibPortIdx <= ibDeviceAttr.phys_port_cnt; ++ibPortIdx) {
        struct ibv_port_attr port;
        if (ibv_query_port(dCtx, ibPortIdx, &port) != 0) {
          printf("    [%u] (ibv_query_port() failed: %s)\n", ibPortIdx, strerror(errno));
          continue;
        }
        const char* linkLayerNames[] = { "Unspecified", "InfiniBand", "Ethernet" };
        printf("    [%u] State: %s LL: %s Speed: %.1f Gbps \n", ibPortIdx, ibv_port_state_str(port.state), linkLayerNames[port.link_layer], ibv_port_speed_gbps(port.active_width, port.active_speed));

        if (port.state >= IBV_PORT_ACTIVE) {
          if ((port.link_layer != IBV_LINK_LAYER_ETHERNET) && (!firstIBContext)) {
            firstIBContext = dCtx;
          }
          if (!firstActivePortsContext)
            firstActivePortsContext = dCtx;
        }
      }
    }
    if (firstIBContext) {
      printf("Selected device %s (first device with active InfiniBand ports)\n", ibv_get_device_name(firstIBContext->device));
      selectedDevice = firstIBContext;
    } else if (firstActivePortsContext) {
      printf("Selected device %s (first device with any active port)\n", ibv_get_device_name(firstActivePortsContext->device));
      selectedDevice = firstActivePortsContext;
    } else {
      selectedDevice = devices[0];
      printf("Selected device %s (first device)\n", ibv_get_device_name(selectedDevice->device));
    }
    rdma_free_devices(devices);
  }

  assert(selectedDevice);

  RDMAContext* ctx = new RDMAContext(selectedDevice);
  ctx->initServer();
  return ctx;
}

void RDMAContext::initServer() {

  // struct sockaddr_in6 addr;
  struct sockaddr_in addr;

  memset(&addr, 0, sizeof(addr));
  // addr.sin6_family = AF_INET6;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(rdmaPort);

  TEST_Z(m_serverEventChannel = rdma_create_event_channel());
  TEST_NZ(rdma_create_id(m_serverEventChannel, &m_listener, NULL, RDMA_PS_TCP));
  TEST_NZ(rdma_bind_addr(m_listener, (struct sockaddr *)&addr));

  m_listener->verbs = m_ctx; // TODO not sure this is the right way to bind to a specific device

  TEST_NZ(rdma_listen(m_listener, /*backlog=*/4));

  // Dump addresses associated with the interface we're bound to
  {
    struct ifaddrs *ifaddr = NULL;
    getifaddrs(&ifaddr);

    std::vector<std::string> deviceNames;
    deviceNames.push_back(ibv_get_device_name(m_ctx->device));

    // Get related net devices from sysfs
    {
      char path[IBV_SYSFS_PATH_MAX + 256];
      snprintf(path, IBV_SYSFS_PATH_MAX + 256, "%s/device/net", m_ctx->device->ibdev_path);
      DIR* dir = opendir(path);
      if (dir) {
        struct dirent* de;
        while ((de = readdir(dir))) {
          if (de->d_name[0] == '.' || de->d_name[0] == '\0')
            continue;

          deviceNames.push_back(std::string(de->d_name));
        }
        closedir(dir);
      }
    }

    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == NULL)
        continue;

      // Check for a prefix match on the device name
      bool didMatchDevice = false;
      for (size_t deviceNameIdx = 0; deviceNameIdx < deviceNames.size(); ++deviceNameIdx) {
        if (strstr(ifa->ifa_name, deviceNames[deviceNameIdx].c_str()) == ifa->ifa_name) {
          didMatchDevice = true;
          break;
        }
      }

      if (!didMatchDevice)
        continue;

      int family = ifa->ifa_addr->sa_family;

      if (family == AF_INET || family == AF_INET6) {
        char host[NI_MAXHOST];
        int s = getnameinfo(ifa->ifa_addr,
          (family == AF_INET) ? sizeof(struct sockaddr_in) :
                                sizeof(struct sockaddr_in6),
          host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
        if (s != 0) {
          printf("(getnameinfo() failed: %s)\n", gai_strerror(s));
        } else {
          printf("%s address: %s\n", ifa->ifa_name, host);
        }
      }
    }

    freeifaddrs(ifaddr);
  }

  printf("RDMA listening on port %u\n", rdmaPort);

  pthread_create(&m_listenerThread, NULL, &RDMAContext::listenerThreadEntryPoint, this);
}

/*static*/ RDMAContext* RDMAContext::createClientContext(const char* host) {
  RDMAContext* ctx = NULL;

  char portStr[8];
  sprintf(portStr, "%d", rdmaPort);

  struct addrinfo *addr;
  struct rdma_cm_id* server;

  TEST_NZ(getaddrinfo(host, portStr, NULL, &addr));
  TEST_NZ(rdma_create_id(NULL, &server, NULL, RDMA_PS_TCP)); // Create in synchronous mode while connecting

  // Resolve address
  TEST_NZ(rdma_resolve_addr(server, NULL, addr->ai_addr, rdmaConnectionTimeoutMs));
  if (server->event->event == RDMA_CM_EVENT_ADDR_ERROR) {
    die("RDMAContext::createClientContext(%s): RDMA_CM_EVENT_ADDR_ERROR\n", host);
  } else if (server->event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
    die("RDMAContext::createClientContext(%s): rdma_resolve_addr(): unexpected event %s\n", host, rdma_event_str(server->event->event));
  }
  freeaddrinfo(addr);

  // Should have enough information to build the context now


  ctx = new RDMAContext(server->verbs);
  ctx->m_server = server;
  ctx->m_clientConnection = new RDMAConnection(ctx, server);
  ctx->m_peerConnections.insert(ctx->m_clientConnection);

  ctx->m_clientEventChannel = rdma_create_event_channel();

  TEST_NZ(rdma_resolve_route(server, rdmaConnectionTimeoutMs));
  if (server->event->event == RDMA_CM_EVENT_ROUTE_ERROR) {
    die("RDMAContext::createClientContext(%s): RDMA_CM_EVENT_ROUTE_ERROR\n", host);
  } else if (server->event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
    die("RDMAContext::createClientContext(%s): rdma_resolve_route(): unexpected event %s\n", host, rdma_event_str(server->event->event));
  }

  // Connect
  struct rdma_conn_param cm_params;
  memset(&cm_params, 0, sizeof(cm_params));

  cm_params.initiator_depth = cm_params.responder_resources = 1;
  cm_params.rnr_retry_count = 7; // infinite retry on receiver-not-ready

  TEST_NZ(rdma_connect(server, &cm_params));
  if (server->event->event != RDMA_CM_EVENT_ESTABLISHED) {
    die("RDMAContext::createClientContext(%s): rdma_connect(): unexpected event %s\n", host, rdma_event_str(server->event->event));
  }

  TEST_NZ(rdma_migrate_id(server, ctx->m_clientEventChannel));
  pthread_create(&ctx->m_clientEventThread, NULL, &RDMAContext::clientEventThreadEntryPoint, ctx);

  return ctx;
}

void RDMAContext::clientEventThreadFn() {
  struct rdma_cm_event *ev = NULL;
  while (rdma_get_cm_event(m_clientEventChannel, &ev) == 0) {
    printf("RDMAContext(%p)::clientEventThread: event %s\n", this, rdma_event_str(ev->event));
    rdma_ack_cm_event(ev);
  }

  printf("RDMAContext::clientEventThread: Shutting down due to error response (%s) from rdma_get_cm_event\n", strerror(errno));

  rdma_destroy_event_channel(m_clientEventChannel);
}

