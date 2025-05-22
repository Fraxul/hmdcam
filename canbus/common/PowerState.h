#pragma once
#include <stdint.h>
#include <string.h>
#include "common/SerializationBuffer.h"
#include "CANBus.h"

struct PowerState {
  static constexpr uint16_t kPortId = 101;

  uint16_t systemVoltage_mV = 0;
  uint8_t stateOfCharge_pct = 0;
  uint16_t batteryVoltage_mV = 0;
  int16_t batteryChargeCurrent_mA = 0;
  int16_t batteryPower_mW = 0;

  uint16_t chargerPowerInput_mV = 0;
  uint16_t chargerPowerInput_mA = 0;

  uint64_t messageTimestamp = 0;

  void handleMessage(SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_usec) {
    try {
      systemVoltage_mV = b.get_u16_le();
      stateOfCharge_pct = b.get_u8();
      batteryVoltage_mV = b.get_u16_le();
      batteryChargeCurrent_mA = b.get_s16_le();
      batteryPower_mW = b.get_s16_le();
      chargerPowerInput_mV = b.get_u16_le();
      chargerPowerInput_mA = b.get_u16_le();

      messageTimestamp = timestamp_usec;
    } catch (const std::exception& ex) {
      messageTimestamp = 0; // Flag contents as invalid
      printf("CANBus::State::deserialize: %s\n", ex.what());
    }
  }

  size_t toString(char* buf, size_t bufLen) const {
    size_t p = snprintf(buf, bufLen, "%umV (%u%%) %dmA %dmW", batteryVoltage_mV, stateOfCharge_pct, batteryChargeCurrent_mA, batteryPower_mW);
    if (chargerPowerInput_mV) {
      p += snprintf(buf + p, bufLen - p, " (In: %umV %umA)", chargerPowerInput_mV, chargerPowerInput_mA);
    }
    unsigned int dataAge_ms  = (CANBus::currentCANTimestampUs() - messageTimestamp) / 1000000UL;
    if (dataAge_ms > 5000) {
      p += snprintf(buf + p, bufLen - p, " (%ums ago)", dataAge_ms);
    }
    return p;
  }

  bool valid() const { return (messageTimestamp != 0); }
};

