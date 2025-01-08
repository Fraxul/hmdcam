#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
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

template <typename T> bool readEnvironmentVariableVector(const char* envVarName, std::vector<T>& outVar) {
  char* e = getenv(envVarName);
  if (e) {
    try {
      std::vector<std::string> parts;
      boost::split(parts, e, boost::is_any_of(" ,"));
      outVar.resize(parts.size());
      for (size_t i = 0; i < parts.size(); ++i) {
        outVar[i] = boost::lexical_cast<T>(parts[i]);
      }
      return true;
    } catch (const std::exception& ex) {
      fprintf(stderr, "readEnvironmentVariableVector(%s): couldn't cast value \"%s\" to vector of type %s: %s\n", envVarName, e, typeid(T).name(), ex.what());
    }
  }
  return false;
}

