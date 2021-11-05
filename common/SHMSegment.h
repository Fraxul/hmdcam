#pragma once
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/core/noncopyable.hpp>

template <typename T> class SHMSegment : public boost::noncopyable {
public:

  ~SHMSegment() {
    munmap(m_segment, m_size);
    close(m_fd);
  }

  static SHMSegment<T>* createSegment(const char* name, size_t size) {
    int shm_fd = shm_open(name, O_RDWR | O_CREAT, 0600);
    if (shm_fd < 0) {
      perror("shm_open");
      return NULL;
    }
    ftruncate(shm_fd, size);

    SHMSegment* res = new SHMSegment(shm_fd, size);
    new(res->m_segment) T(size);

    return res; 
  }

  static SHMSegment<T>* openSegment(const char* name) {
    int shm_fd = shm_open(name, O_RDWR, 0600);
    if (shm_fd < 0) {
      perror("shm_open");
      return NULL;
    }

    struct stat statbuf;
    if (0 != fstat(shm_fd, &statbuf)) {
      perror("fstat");
      return NULL;
    }

    size_t size = statbuf.st_size;
    // fprintf(stderr, "openSegment(%s): %zu bytes\n", name, size);
    
    return new SHMSegment<T>(shm_fd, size);
  }


  T* segment() const { return m_segment; }

  size_t segmentSize() const { return m_size; }


  void flush(size_t offset, size_t length) {
    msync(reinterpret_cast<char*>(m_segment) + offset, length, MS_SYNC);
  }

protected:
  SHMSegment(int fd, size_t length) : m_fd(fd), m_size(length) {
    void* res = mmap(NULL, m_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE | MAP_LOCKED, m_fd, 0);
    if (res == MAP_FAILED) {
      perror("SHMSegment::<ctor> mmap");
      abort();
    }
    m_segment = reinterpret_cast<T*>(res);
  }

  T* m_segment;
  int m_fd;
  size_t m_size;
};
