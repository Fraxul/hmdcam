#pragma once
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>

class mmfile {
public:
  mmfile(const char* fn) {
    m_fd = open(fn, O_RDONLY);
    if (m_fd == -1) throw std::runtime_error(std::string("mmfile: unable to open file ") + std::string(fn));
    struct stat statbuf;
    fstat(m_fd, &statbuf);
    m_flen = statbuf.st_size;
    m_fbase = (char*) mmap(NULL, m_flen, PROT_READ, MAP_SHARED, m_fd, 0);
  }

  ~mmfile() {
    this->close();
  }

  void close() {
    if (m_fd != -1) {
      munmap(m_fbase, m_flen);
      ::close(m_fd);
      m_fd = -1;
    }
  }

  char* data() { return m_fbase; }
  size_t size() { return m_flen; }

  char* m_fbase;
  size_t m_flen;
  int m_fd;
private:
  mmfile(const mmfile& right); // noncopyable
  mmfile& operator=(const mmfile& right); // nonassignable
};

