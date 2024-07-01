#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include "common/PDUControl.h"
#include "socketcan.h"
#include "Buffer.h"

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

bool PDUControl::State::deserialize(const CanardRxTransfer& transfer) {
  try {
    Buffer b(reinterpret_cast<const char*>(transfer.payload), transfer.payload_size);

    systemVoltage_mV = b.get_u16_le();
    stateOfCharge_pct = b.get_u8();
    batteryVoltage_mV = b.get_u16_le();
    batteryChargeCurrent_mA = b.get_s16_le();
    batteryPower_mW = b.get_s16_le();
    chargerPowerInput_mV = b.get_u16_le();
    chargerPowerInput_mA = b.get_u16_le();

    messageTimestamp = transfer.timestamp_usec;
  } catch (const std::exception& ex) {
    printf("PDUControl::State::deserialize: %s\n", ex.what());
    return false;
  }

  return true;
}

static inline uint64_t currentCANTimestampUs() {
  // CAN frames are timestamped with CLOCK_REALTIME in microseconds
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ((ts.tv_sec * 1000000UL) + (ts.tv_nsec / 1000UL));
}

size_t PDUControl::State::toString(char* buf, size_t bufLen) const {
  size_t p = snprintf(buf, bufLen, "%umV (%u%%) %dmA %dmW", batteryVoltage_mV, stateOfCharge_pct, batteryChargeCurrent_mA, batteryPower_mW);
  if (chargerPowerInput_mV) {
    p += snprintf(buf + p, bufLen - p, " (In: %umV %umA)", chargerPowerInput_mV, chargerPowerInput_mA);
  }
  unsigned int dataAge_ms  = (currentCANTimestampUs() - messageTimestamp) / 1000000UL;
  if (dataAge_ms > 5000) {
    p += snprintf(buf + p, bufLen - p, " (%ums ago)", dataAge_ms);
  }
  return p;
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

    if (outSubscription == &m_heartbeatSubscription) {
    } else if (outSubscription == &m_powerInfoSubscription) {
      // Quick-and-dirty DSDL-less deserialization
      State msg_state;
      if (msg_state.deserialize(transfer)) {
        m_state = msg_state;
      }
/*
      char buf[256];
      m_state.toString(buf, sizeof(buf));
      printf("%s\n", buf);
*/
    } else {
      fprintf(stderr, "prio=%d kind=%d port=%d remoteNode=%d txID=%d ts=%lu payload=%zu bytes\n  ",
        transfer.metadata.priority, transfer.metadata.transfer_kind, transfer.metadata.port_id, transfer.metadata.remote_node_id, transfer.metadata.transfer_id, transfer.timestamp_usec, transfer.payload_size);

      for (size_t i = 0; i < transfer.payload_size; ++i) {
        fprintf(stderr, "%02x ", reinterpret_cast<uint8_t*>(transfer.payload)[i]);
      }
      fprintf(stderr, "\n");
    }
  }

}

