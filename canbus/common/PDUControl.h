#pragma once
#include <stdint.h>
#include <string>
#include <vector>

#include <string.h>
#include "canard.h"

class PDUControl {
public:

  PDUControl();
  ~PDUControl();

  bool isOpen() const { return (m_fd >= 0); }

  struct State {
    State() { memset(this, 0, sizeof(State)); }

    uint16_t systemVoltage_mV;
    uint8_t stateOfCharge_pct;
    uint16_t batteryVoltage_mV;
    int16_t batteryChargeCurrent_mA;
    int16_t batteryPower_mW;

    uint16_t chargerPowerInput_mV;
    uint16_t chargerPowerInput_mA;


    uint64_t messageTimestamp;

    bool deserialize(const CanardRxTransfer& transfer);
    size_t toString(char* buf, size_t bufLen) const;
    bool valid() const { return (messageTimestamp != 0); }
  };

  struct State m_state;

protected:
  void canReadThread();
  static void* canReadThreadThunk(void* pThis) { reinterpret_cast<PDUControl*>(pThis)->canReadThread(); return NULL; }

  int m_fd = -1;
  pthread_t m_thread;

  CanardInstance m_canard;

  CanardRxSubscription m_heartbeatSubscription;
  CanardRxSubscription m_powerInfoSubscription;
};


