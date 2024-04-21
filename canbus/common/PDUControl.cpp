#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include "common/PDUControl.h"
#include "socketcan.h"

const char* interfaceName = "can0";

static void* canardMemAllocate(CanardInstance* const ins, const size_t amount) {
  (void)ins;
  return malloc(amount);
}

static void canardMemFree(CanardInstance* const ins, void* const pointer) {
  (void)ins;
  free(pointer);
}

PDUControl::PDUControl() {

  m_fd = socketcanOpen(interfaceName, false);
  if (m_fd < 0) {
    fprintf(stderr, "PDUControl: Cannot open CAN interface \"%s\": %s\n", interfaceName, strerror(errno));
    return;
  }

  m_canard = canardInit(&canardMemAllocate, &canardMemFree);
  m_canard.node_id = 1;


  // Create subscriptions
  canardRxSubscribe(&m_canard, CanardTransferKindMessage, /*port_id=*/ 7509U, /*extent=*/ 12, CANARD_DEFAULT_TRANSFER_ID_TIMEOUT_USEC, &m_heartbeatSubscription);
  canardRxSubscribe(&m_canard, CanardTransferKindMessage, /*port_id=*/ 101, /*extent=*/ 16, CANARD_DEFAULT_TRANSFER_ID_TIMEOUT_USEC, &m_powerInfoSubscription);

  pthread_create(&m_thread, NULL, canReadThreadThunk, this);
}

PDUControl::~PDUControl() {
  if (m_fd >= 0) {
    pthread_cancel(m_thread);
    void* retval;
    pthread_join(m_thread, &retval);
    ::close(m_fd);
  }
}


void PDUControl::canReadThread() {
  while (true) {
    pthread_testcancel();

    CanardFrame rxFrame{};
    CanardMicrosecond timestamp_us{};
    char payload_buffer[64];

    int16_t res = socketcanPop(m_fd, &rxFrame, &timestamp_us, sizeof(payload_buffer), payload_buffer, 1'000'000, /*loopback=*/ nullptr);
    // fprintf(stderr, "socketcanPop(): %d\n", res);
    if (res == 0) {
      continue; // timed out
    } else if (res < 0) {
      fprintf(stderr, "canReadThread(): socketcanPop errno=%d: %s\n", -res, strerror(-res));
      continue;
    }

    CanardRxTransfer transfer;
    CanardRxSubscription* outSubscription = nullptr;
    
    res = canardRxAccept((CanardInstance* const)&m_canard, timestamp_us, &rxFrame, /*redundant_iface_index=*/ 0, &transfer, &outSubscription);
    // fprintf(stderr, "canardRxAccept(): %d\n", res);
    if (res != 1) {
      continue; // the frame received is not a valid transfer
    }

    fprintf(stderr, "prio=%d kind=%d port=%d remoteNode=%d txID=%d ts=%lu payload=%zu bytes\n  ",
      transfer.metadata.priority, transfer.metadata.transfer_kind, transfer.metadata.port_id, transfer.metadata.remote_node_id, transfer.metadata.transfer_id, transfer.timestamp_usec, transfer.payload_size);

    for (size_t i = 0; i < transfer.payload_size; ++i) {
      fprintf(stderr, "%02x ", reinterpret_cast<uint8_t*>(transfer.payload)[i]);
    }
    fprintf(stderr, "\n");
  }

}

