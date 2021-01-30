#pragma once
#include <stddef.h>
#include <stdint.h>

class FxAtomicString {
public:

  FxAtomicString(const char*);
  FxAtomicString() : m_ref(0) {}

  FxAtomicString(const FxAtomicString& right) : m_ref(right.m_ref) {}
  FxAtomicString& operator=(const FxAtomicString& right) {
    m_ref = right.m_ref;
    return *this;
  }

  // returns a non-empty FxAtomicString only if the input string is already in the string pool
  static FxAtomicString toAtomicStringIfValid(const char*);

  operator const char*() const { return m_ref; }
  bool operator==(const FxAtomicString& right) const { return m_ref == right.m_ref; }
  bool operator!=(const FxAtomicString& right) const { return m_ref != right.m_ref; }
  bool operator==(const char* right) const;
  // Only for container use (LessThanComparable concept) -- not a proper lexicographical comparison.
  bool operator<(const FxAtomicString& right) const { return m_ref < right.m_ref; }
  operator bool() const { return (bool) m_ref; }
  size_t length() const;
  const char* c_str() const { return m_ref; }
  const char* data() const { return m_ref; }
  void clear() { m_ref = NULL; }

protected:
  const char* m_ref;
};

namespace boost {
  size_t hash_value(const FxAtomicString&);
}

