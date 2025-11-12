#pragma once
#include <stdint.h>
#include <string.h>
#include "common/SerializationBuffer.h"
#include "CANBus.h"

struct GloveControllerPacket {
  static constexpr uint16_t kLeftPortId = 301;
  static constexpr uint16_t kRightPortId = 302;

  int32_t accel_milliG[3] = {0, 0, 0};
  int32_t gyro_milliDPS[3] = {0, 0, 0};
  uint8_t buttonState = 0;

  uint64_t messageTimestamp = 0;

  void handleMessage(SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_usec) {
    try {
      for (size_t i = 0; i < 3; ++i)
        accel_milliG[i] = b.get_s32_le();

      for (size_t i = 0; i < 3; ++i)
        gyro_milliDPS[i] = b.get_s32_le();

      buttonState = b.get_u8();

      if (b.remaining() != 0) {
        printf("GloveControllerPacket::handleMessage: serialization mismatch, %zu bytes remaining in buffer\n", b.remaining());
        messageTimestamp = 0; // Flag contents as invalid
        return;
      }

      messageTimestamp = timestamp_usec;
    } catch (const std::exception& ex) {
      messageTimestamp = 0; // Flag contents as invalid
      printf("GloveControllerPacket::handleMessage: %s\n", ex.what());
    }
  }

  size_t toString(char* buf, size_t bufLen) const {
    size_t p = snprintf(buf, bufLen, "XL=[%.3f, %.3f, %.3f] Gyro=[%.3f, %.3f, %.3f] Btn=%x",
      static_cast<float>(accel_milliG[0]) / 1000.0f,
      static_cast<float>(accel_milliG[1]) / 1000.0f,
      static_cast<float>(accel_milliG[2]) / 1000.0f,
      static_cast<float>(gyro_milliDPS[0]) / 1000.0f,
      static_cast<float>(gyro_milliDPS[1]) / 1000.0f,
      static_cast<float>(gyro_milliDPS[2]) / 1000.0f,
      buttonState);

    unsigned int dataAge_ms  = (CANBus::currentCANTimestampUs() - messageTimestamp) / 1000000UL;
    if (dataAge_ms > 5000) {
      p += snprintf(buf + p, bufLen - p, " (%ums ago)", dataAge_ms);
    }
    return p;
  }

  bool valid() const { return (messageTimestamp != 0); }
};

