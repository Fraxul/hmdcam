#include "RDMAConnection.h"
#include "RDMAContext.h"
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

RDMAConnection::RDMAConnection(RDMAContext* ctx, struct rdma_cm_id* id) : m_ctx(ctx), m_id(id) {
  ibv_qp_init_attr qp_attr;
  memset(&qp_attr, 0, sizeof(ibv_qp_init_attr));

  qp_attr.send_cq = m_ctx->m_cq;
  qp_attr.recv_cq = m_ctx->m_cq;
  qp_attr.qp_type = IBV_QPT_RC;

  qp_attr.cap.max_send_wr = 32;
  qp_attr.cap.max_recv_wr = 32;
  qp_attr.cap.max_send_sge = 4;
  qp_attr.cap.max_recv_sge = 4;

  rdma_create_qp(m_id, m_ctx->m_pd, &qp_attr);
  m_id->context = this;
  m_qp = m_id->qp;

  m_bufferSize = 4096;
  m_recvBuffer = new char[m_bufferSize];

  TEST_Z(m_recvMr = ibv_reg_mr(m_ctx->m_pd, m_recvBuffer, m_bufferSize, IBV_ACCESS_LOCAL_WRITE));

  postReceives();
}

RDMAConnection::~RDMAConnection() {
  rdma_destroy_qp(m_id);

  ibv_dereg_mr(m_recvMr);

  rdma_destroy_id(m_id);

  delete[] m_recvBuffer;
}


bool RDMAConnection::sendMessage(uint32_t opcode, SerializationBuffer payload) {
  SerializationBuffer b;
  b.put_u32(opcode);
  b.put_u32(payload.size());
  b.append(payload);

  return internalSendMessage(b.data(), b.size());
}

bool RDMAConnection::internalSendMessage(const char* payload, size_t length) {
  struct ibv_send_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;
  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  wr.wr_id = reinterpret_cast<uintptr_t>(this);
  wr.opcode = IBV_WR_SEND;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

  sge.addr = (uintptr_t)payload;
  sge.length = length;

  int res = ibv_post_send(m_qp, &wr, &bad_wr);
  if (res != 0) {
    printf("RDMAConnection(%p)::internalSendMessage(%zu bytes): ibv_post_send() failed: %s\n",
      this, length, strerror(res));
    return false;
  }
  return true;
}

void RDMAConnection::didEstablishConnection() {
  // printf("RDMAConnection(%p)::didEstablishConnection()\n", this);
}

void RDMAConnection::didCompleteReceive(size_t bytesReceived) {
  // printf("RDMAConnection(%p)::didCompleteReceive(%zu bytes)\n", this, bytesReceived);

  SerializationBuffer b(m_recvBuffer, bytesReceived);
  uint32_t opcode = b.get_u32();
  uint32_t payloadLength = b.get_u32();
  SerializationBuffer payload = b.consume_buffer(payloadLength);

  context()->processMessage(this, opcode, payload);

  postReceives();
}

void RDMAConnection::didCompleteSend() {
  // printf("RDMAConnection(%p)::didCompleteSend()\n", this);
}

void RDMAConnection::postReceives() {
  struct ibv_recv_wr wr, *bad_wr = NULL;
  struct ibv_sge sge;

  wr.wr_id = reinterpret_cast<uintptr_t>(this);
  wr.next = NULL;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)m_recvBuffer;
  sge.length = m_bufferSize;
  sge.lkey = m_recvMr->lkey;

  TEST_NZ(ibv_post_recv(m_qp, &wr, &bad_wr));
}

