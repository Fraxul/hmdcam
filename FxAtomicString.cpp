#include "FxAtomicString.h"
#include <map>
#include <string>
#include <string.h>
#include <boost/functional/hash.hpp>

std::map<std::string, FxAtomicString>& atomicStringPool() {
  static std::map<std::string, FxAtomicString>* s_atomicStringPool = NULL;
  if (!s_atomicStringPool)
    s_atomicStringPool = new std::map<std::string, FxAtomicString>();
  return *s_atomicStringPool;
}

FxAtomicString::FxAtomicString(const char* value) {
  std::string s(value);
  std::map<std::string, FxAtomicString>::iterator it = atomicStringPool().find(s);

  if (it == atomicStringPool().end()) {
    it = atomicStringPool().insert(std::make_pair(s, FxAtomicString())).first;
    it->second.m_ref = it->first.c_str();
  }

  m_ref = it->second;
}

/*static*/ FxAtomicString FxAtomicString::toAtomicStringIfValid(const char* value) {
  FxAtomicString res;

  std::map<std::string, FxAtomicString>::iterator it = atomicStringPool().find(value);
  if (it != atomicStringPool().end()) {
    res.m_ref = it->first.c_str();
  }

  return res;
}

bool FxAtomicString::operator==(const char* right) const {
  return (strcmp(m_ref, right) == 0);
}

size_t FxAtomicString::length() const {
  return strlen(m_ref);
}

namespace boost {
  size_t hash_value(const FxAtomicString& str) {
    size_t hv = 0;
    for (const char* p = str.data(); *p; ++p) {
      boost::hash_combine(hv, *p);
    }
    return hv;
  }
}

