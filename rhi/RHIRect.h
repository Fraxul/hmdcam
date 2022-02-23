#pragma once
#include <stddef.h>
#include <stdint.h>

struct RHIRect {
  RHIRect() : x(0), y(0), width(0), height(0) {}
  static RHIRect xywh(uint32_t x, uint32_t y, uint32_t w, uint32_t h) { return RHIRect(x, y, w, h); }
  static RHIRect ltrb(uint32_t l, uint32_t t, uint32_t r, uint32_t b) {
    uint32_t _x = ((l < r) ? l : r);
    uint32_t _y = ((t < b) ? t : b);
    return RHIRect(_x, _y, ((l > r) ? l : r) - _x, ((t > b) ? t : b)- _y);
  }
  static RHIRect sized(uint32_t w, uint32_t h) { return RHIRect(0, 0, w, h); }

  bool empty() const { return (width == 0) && (height == 0); }

  uint32_t x, y;
  uint32_t width, height;

  uint32_t left() const { return x; }
  uint32_t top() const { return y; }
  uint32_t right() const { return x + width; }
  uint32_t bottom() const { return y + height; }
protected:
  RHIRect(uint32_t x_, uint32_t y_, uint32_t w_, uint32_t h_) : x(x_), y(y_), width(w_), height(h_) {}
};

