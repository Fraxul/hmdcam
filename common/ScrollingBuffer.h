#pragma once
#include <vector>
#include <stdint.h>

template <typename T> class ScrollingBuffer {
public:
  ScrollingBuffer(size_t maxSize = 1024) : m_maxSize(maxSize), m_offset(0) {
    m_data.reserve(m_maxSize);
  }

  size_t offset() const { return m_offset; }
  const T* data() const { return m_data.data(); }
  size_t maxSize() const { return m_maxSize; }
  size_t size() const { return m_data.size(); }
  static size_t stride() { return sizeof(T); }

  void push_back(const T& value) {
    if (m_data.size() < m_maxSize) {
      m_data.push_back(value);
    } else {
      m_data[m_offset] = value;
      m_offset = (m_offset + 1) % m_maxSize;
    }
  }

  void clear() {
    m_data.clear();
    m_offset = 0;
  }

protected:
  std::vector<T> m_data;
  size_t m_maxSize;
  size_t m_offset;
};

