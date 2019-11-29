#pragma once
#include <stddef.h>
#include <stdint.h>

// Windows doesn't define ssize_t
#ifdef  _WIN64
  typedef __int64    ssize_t;
#elif defined(_WIN32)
  typedef _W64 int   ssize_t;
#endif
