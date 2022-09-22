#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>
#include <typeinfo>

template <typename T> bool readEnvironmentVariable(const char* envVarName, T& outVar) {
  char* e = getenv(envVarName);
  if (e) {
    try {
      outVar = boost::lexical_cast<T>(e);
      return true;
    } catch (const std::exception& ex) {
      fprintf(stderr, "readEnvironmentVariable(%s): couldn't cast value \"%s\" to type %s: %s\n", envVarName, e, typeid(T).name(), ex.what());
    }
  }
  return false;
}

