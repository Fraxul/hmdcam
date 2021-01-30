#pragma once
#include <stddef.h>
#include <stdint.h>

static inline uint32_t floorLogTwo(uint32_t v) {
  uint32_t pos = 0;
  if (v >= 1<<16) { v >>= 16; pos += 16; }
  if (v >= 1<< 8) { v >>=  8; pos +=  8; }
  if (v >= 1<< 4) { v >>=  4; pos +=  4; }
  if (v >= 1<< 2) { v >>=  2; pos +=  2; }
  if (v >= 1<< 1) {           pos +=  1; }
  return (v == 0) ? 0 : pos;
}

static inline uint32_t leadingZeros(uint32_t x) {
  if (x == 0) return 32;
  return 31 - floorLogTwo(x);
}

static inline uint32_t ceilLogTwo(uint32_t x) {
  uint32_t mask = (leadingZeros(x) << 26) >> 31;
  return (32 - leadingZeros(x - 1)) & (~mask);
}

static inline uint32_t nextPowerOfTwo(uint32_t x) {
  return 1 << ceilLogTwo(x);
}


