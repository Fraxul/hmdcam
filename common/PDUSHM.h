#pragma once
#include <stdint.h>

struct PDUInfo {
  PDUInfo(size_t segmentLength) { memset(this, 0, sizeof(PDUInfo)); }

  uint64_t lastUpdateTimeMs; // based on CLOCK_MONOTONIC, milliseconds
  char statusLines[1024];


  char dataSegmentEndMarker[64];

  static constexpr size_t dataSegmentStart() {
    return 0;
  }
  static constexpr size_t dataSegmentSize() {
    return offsetof(PDUInfo, dataSegmentEndMarker);
  }

  static constexpr size_t controlSegmentStart() {
    return offsetof(PDUInfo, dataSegmentEndMarker) + 64;
  } 
  static constexpr size_t controlSegmentSize() {
    return sizeof(PDUInfo) - controlSegmentStart();
  }

  // control segment
  uint32_t clearRequested;
};

