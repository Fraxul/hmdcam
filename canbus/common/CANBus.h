#pragma once
#include <stdint.h>
#include "common/SerializationBuffer.h"
#include "canard.h"
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include <sys/time.h>
#include <time.h>

#include <list>

class CANBus;
CANBus* canbus();

class CANBus {
public:

  CANBus();
  ~CANBus();

  bool isOpen() const { return (m_fd >= 0); }

  static uint64_t currentCANTimestampUs() {
    // CAN frames are timestamped with CLOCK_REALTIME in microseconds
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((ts.tv_sec * 1000000UL) + (ts.tv_nsec / 1000UL));
  }

  // Setting maxMessageLength larger than the actual message is fine, but a shorter maxMessageLength will cause message truncation.
  void addMessageSubscription(CanardPortID port_id, std::function<void(SerializationBuffer&, const CanardTransferMetadata&, uint64_t)> handler, size_t maxMessageLength = 256, uint64_t messageTimeoutUs = CANARD_DEFAULT_TRANSFER_ID_TIMEOUT_USEC);

  void transmitMessage(CanardPortID port_id, SerializationBuffer& b);

protected:

  // CanardRxSubscription object can't be moved while active, so SubscriptionData objects must be passed around by pointer.
  struct SubscriptionData : public CanardRxSubscription, boost::noncopyable {

    std::function<void(SerializationBuffer&, const CanardTransferMetadata&, uint64_t)> m_handler;
  };


  void canRxThread();
  void canTxThread();

  // Socket
  int m_fd = -1;

  // Rx data
  CanardInstance m_canard;
  boost::thread m_rxThread;
  boost::mutex m_subscriptionLock;
  std::list<SubscriptionData*> m_subscriptions;


  // Tx data
  boost::thread m_txThread;
  CanardTxQueue m_txQueue;
  boost::mutex m_txQueueLock; // Protects m_txQueue and m_txQueueFilled
  boost::condition_variable m_txQueueFilled;
  std::map<CanardPortID, CanardTransferMetadata*> m_txMetadata;
  CanardTransferMetadata* getTransferMetadata(CanardPortID);

};

