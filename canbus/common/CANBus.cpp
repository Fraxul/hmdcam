#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

#include "common/CANBus.h"
#include "socketcan.h"
#include "common/SerializationBuffer.h"

#include <boost/thread/lock_guard.hpp>

const char* interfaceName = "can0";

static CANBus* s_canbus;
CANBus* canbus() {
  if (!s_canbus)
    s_canbus = new CANBus();

  return s_canbus;
}

static void* canardMemAllocate(CanardInstance* const ins, const size_t amount) {
  (void)ins;
  return malloc(amount);
}

static void canardMemFree(CanardInstance* const ins, void* const pointer) {
  (void)ins;
  free(pointer);
}

void heartbeatMessageHandler(SerializationBuffer&, const CanardTransferMetadata&, uint64_t) {

}

CANBus::CANBus() {

  m_fd = socketcanOpen(interfaceName, /*CAN-FD mode=*/ false);
  if (m_fd < 0) {
    fprintf(stderr, "CANBus: Cannot open CAN interface \"%s\": %s\n", interfaceName, strerror(errno));
    return;
  }
  // Test reading a frame from the device.
  // We may be able to open it, but socketcanPop will return an error if it's not ready for reading.
  {
    CanardFrame rxFrame{};
    CanardMicrosecond timestamp_us{};
    char payload_buffer[64];
    int16_t res = socketcanPop(m_fd, &rxFrame, &timestamp_us, sizeof(payload_buffer), payload_buffer, 1'000'000, /*loopback=*/ nullptr);
    if (res < 0) {
      fprintf(stderr, "CANBus(): Initial test read from CAN socket failed with errno=%d: %s\n", -res, strerror(-res));
      ::close(m_fd);
      m_fd = -1;
      return;
    }
  }

  // Rx setup
  m_canard = canardInit(&canardMemAllocate, &canardMemFree);
  m_canard.node_id = 1;

  // Default heartbeat subscription
  addMessageSubscription(7509U, heartbeatMessageHandler, /*extent=*/ 12);

  // Tx setup
  m_txQueue = canardTxInit(/*capacity (packets)=*/ 128, /*mtu_bytes=*/ CANARD_MTU_CAN_CLASSIC);

  // Worker threads
  m_rxThread = boost::thread(boost::bind(&CANBus::canRxThread, this));
  m_txThread = boost::thread(boost::bind(&CANBus::canTxThread, this));
}

CANBus::~CANBus() {
  m_rxThread.interrupt();
  m_rxThread.join();

  m_txThread.interrupt();
  m_txThread.join();

  if (m_fd >= 0) {
    ::close(m_fd);
  }

  for (SubscriptionData* sd : m_subscriptions) {
    delete sd;
  }

  for (const auto& p : m_txMetadata) {
    delete p.second;
  }

  // Drain Tx queue
  while (CanardTxQueueItem* item = canardTxPop(&m_txQueue, canardTxPeek(&m_txQueue))) {
    canardMemFree(&m_canard, item);
  }

}

void CANBus::addMessageSubscription(CanardPortID port_id, std::function<void(SerializationBuffer&, const CanardTransferMetadata&, uint64_t)> handler, size_t maxMessageLength, uint64_t messageTimeoutUs) {
  if (m_fd < 0)
    return; // CAN service not available, just ignore subscription requests

  // Accessing the canard subscription data requires holding m_subscriptionLock
  boost::lock_guard<boost::mutex> l(m_subscriptionLock);

  SubscriptionData* sub = new SubscriptionData();
  sub->m_handler = handler;
  canardRxSubscribe(&m_canard, CanardTransferKindMessage, /*port_id=*/ port_id, /*extent=*/ maxMessageLength, /*timeout_us=*/ messageTimeoutUs, sub);

  m_subscriptions.push_back(sub);
}

void CANBus::canTxThread() {
  pthread_setname_np(pthread_self(), "canTxThread");

  while (true) {
    if (boost::this_thread::interruption_requested())
      break;

    boost::unique_lock<boost::mutex> l(m_txQueueLock);

    const CanardTxQueueItem* peek = nullptr;
    while (peek == nullptr) {
      peek = canardTxPeek(&m_txQueue);
      if (peek)
        break;

      m_txQueueFilled.wait(l);
    }

    // Takes ownership of peek item, but does not delete it yet.
    CanardTxQueueItem* item = canardTxPop(&m_txQueue, peek);

    uint64_t now = currentCANTimestampUs();

    bool transferOK = false;

    // Check deadline and compute timeout
    if (item->tx_deadline_usec > now) {
      uint64_t timeout_usec = item->tx_deadline_usec - now;

      int res = socketcanPush(m_fd, &item->frame, timeout_usec);
      if (res < 0) {
        fprintf(stderr, "canTxThread(): socketcanPush errno=%d: %s\n", -res, strerror(-res));
      }

      // Free item once we're done with it
      canardMemFree(&m_canard, item);
      transferOK = true;
    }

    if (!transferOK) {
      // Item timed out before it could be transmitted.
      // Free this node and remove/free any subsequent nodes in its next_in_transfer chain
      do {
        CanardTxQueueItem* to_free = item;
        item = item->next_in_transfer ? canardTxPop(&m_txQueue, item->next_in_transfer) : nullptr;
        canardMemFree(&m_canard, to_free);
      } while (item != nullptr);
    }
  }
}

void CANBus::canRxThread() {
  pthread_setname_np(pthread_self(), "canRxThread");

  while (true) {
    if (boost::this_thread::interruption_requested())
      break;

    CanardFrame rxFrame{};
    CanardMicrosecond timestamp_us{};
    char payload_buffer[CANARD_MTU_MAX];

    int16_t res = socketcanPop(m_fd, &rxFrame, &timestamp_us, sizeof(payload_buffer), payload_buffer, 1'000'000, /*loopback=*/ nullptr);
    // fprintf(stderr, "socketcanPop(): %d\n", res);
    if (res == 0) {
      continue; // timed out
    } else if (res < 0) {
      fprintf(stderr, "canRxThread(): socketcanPop errno=%d: %s\n", -res, strerror(-res));
      continue;
    }

    CanardRxTransfer transfer;
    CanardRxSubscription* outSubscription = nullptr;
    {
      // Accessing the canard subscription data requires holding m_subscriptionLock
      boost::lock_guard<boost::mutex> l(m_subscriptionLock);

      res = canardRxAccept((CanardInstance* const)&m_canard, timestamp_us, &rxFrame, /*redundant_iface_index=*/ 0, &transfer, &outSubscription);
      // fprintf(stderr, "canardRxAccept(): %d\n", res);
      if (res != 1) {
        continue; // the frame received is not a valid transfer
      }

      // SubscriptionData derives from CanardRxSubscription, so we can up-cast to get our extra data (handler fn)
      SubscriptionData* subData = static_cast<SubscriptionData*>(outSubscription);

      if (subData) {
        // Subscription is registered, so call the handler.
        SerializationBuffer b = SerializationBuffer::withStringRef(reinterpret_cast<const char*>(transfer.payload), transfer.payload_size);
        subData->m_handler(b, transfer.metadata, timestamp_us);
      } else {
        fprintf(stderr, "prio=%d kind=%d port=%d remoteNode=%d txID=%d ts=%lu payload=%zu bytes\n  ",
          transfer.metadata.priority, transfer.metadata.transfer_kind, transfer.metadata.port_id, transfer.metadata.remote_node_id, transfer.metadata.transfer_id, transfer.timestamp_usec, transfer.payload_size);

        for (size_t i = 0; i < transfer.payload_size; ++i) {
          fprintf(stderr, "%02x ", reinterpret_cast<uint8_t*>(transfer.payload)[i]);
        }
        fprintf(stderr, "\n");
      }

      // Reassembled payload is dynamically allocated and must be freed
      canardMemFree(&m_canard, transfer.payload);
    }
  }

}

CanardTransferMetadata* CANBus::getTransferMetadata(CanardPortID port_id) {
  CanardTransferMetadata*& md = m_txMetadata[port_id];
  if (md == nullptr) {
    // Initialize metadata on first use
    md = new CanardTransferMetadata();

    md->priority = CanardPriorityNominal;
    md->transfer_kind = CanardTransferKindMessage;
    md->port_id = port_id;
    md->remote_node_id = CANARD_NODE_ID_UNSET;
    md->transfer_id = 0;
  }
  return md;
}

void CANBus::transmitMessage(CanardPortID port_id, SerializationBuffer& b) {
  // Hold Tx queue lock while adding frames
  boost::lock_guard<boost::mutex> l(m_txQueueLock);

  CanardTransferMetadata* md = getTransferMetadata(port_id);

  // canardTxPush references the m_canard instance that is mostly used for receiving, but does not require
  // holding any receive-related locks -- the m_canard instance is only used for its memory allocation functions.
  canardTxPush(&m_txQueue, &m_canard,
    /*tx_deadline_usec=*/ currentCANTimestampUs() + CANARD_DEFAULT_TRANSFER_ID_TIMEOUT_USEC,
    md,
    b.size(), b.data());

  // Increment and wrap transfer_id
  md->transfer_id = (md->transfer_id + 1U) & ((1U << CANARD_TRANSFER_ID_BIT_LENGTH) - 1U);

  // Wake Tx thread
  m_txQueueFilled.notify_one();
}

