#include "common/PosixFileUtil.h"
#include <cerrno>
#include <unistd.h>

bool writeFully(int fd, const void* data, size_t length) {
  const unsigned char* p = reinterpret_cast<const unsigned char*>(data);
  while (length) {
    ssize_t res = write(fd, p, length);
    if (res < 0) {
      if (errno == EINTR)
        continue; // Interrupted before any bytes were written; retry.
      return false;
    }
    p += res;
    length -= res;
  }
  return true;
}
