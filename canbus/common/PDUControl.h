#pragma once
#include <stdint.h>
#include <string>
#include <vector>

#include "canard.h"

class PDUControl {
public:

  PDUControl();
  ~PDUControl();

  bool isOpen() const;

protected:
  void canReadThread();
  static void* canReadThreadThunk(void* pThis) { reinterpret_cast<PDUControl*>(pThis)->canReadThread(); return NULL; }

  int m_fd = -1;
  pthread_t m_thread;

  CanardInstance m_canard;

  CanardRxSubscription m_heartbeatSubscription;
  CanardRxSubscription m_powerInfoSubscription;
};


