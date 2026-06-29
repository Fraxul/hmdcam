#pragma once
#include <cstddef>

// Writes the entire buffer to fd, retrying short writes and EINTR. Returns true on success.
bool writeFully(int fd, const void* data, size_t length);
