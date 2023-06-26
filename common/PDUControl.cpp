#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <poll.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <glob.h>

#include "common/PDUControl.h"

const char* portPattern = "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_*";

PDUControl::PDUControl() {

}

PDUControl::~PDUControl() {
  if (m_fd >= 0) {
    ::close(m_fd);
  }

}

bool PDUControl::tryOpenSerial() {
  {
    glob_t portGlob;
    memset(&portGlob, 0, sizeof(portGlob));
    int globRes = glob(portPattern, /*flags=*/ 0, /*errFunc=*/ nullptr, &portGlob);
    if (globRes != 0 || portGlob.gl_pathc == 0) {
      fprintf(stderr, "PDUControl::tryOpenSerial(): Couldn't find any matches for port pattern \"%s\"\n", portPattern);
      return false;
    }

    m_serialPort = portGlob.gl_pathv[0];
    fprintf(stderr, "PDUControl::tryOpenSerial(): Trying port \"%s\"\n", m_serialPort.c_str());

    m_fd = open(m_serialPort.c_str(), O_RDWR | O_NONBLOCK);
    if (m_fd < 0) {
      fprintf(stderr, "PDUControl: error opening %s: %s\n", m_serialPort.c_str(), strerror(errno));
      return -1;
    }

    // Set baudrate to 115,200
    struct termios options;
    if (tcgetattr(m_fd, &options) != 0) {
      perror("PDUControl::tryOpenSerial(): tcgetattr");
      goto fail;
    }

    cfmakeraw(&options);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    if (tcsetattr(m_fd, TCSANOW, &options) != 0) {
      perror("PDUControl::tryOpenSerial(): tcsetattr");
      goto fail;
    }

    return true;
  }

fail:
  ::close(m_fd);
  m_fd = -1;
  return false;
}

void PDUControl::drainSerialInput() {
  struct pollfd pfd;
  pfd.fd = m_fd;
  pfd.events = POLLIN;

  while (true) {
    pfd.revents = 0;

    int pollRes = poll(&pfd, 1, 100);
    if (pollRes <= 0)
      break;

    char buf[1024];
    ssize_t n = read(m_fd, buf, 1024);
    if (n < 0) {
      if (errno == EAGAIN)
        continue;

      perror("PDUControl::drainSerialInput(): read()");
      return; // read error
    }

    if (n > 0)
      m_partialLineBuf.append(buf, buf + n);

    // split by lines
    while (true) {
      auto loc = m_partialLineBuf.find_first_of("\r\n");
      if (loc == std::string::npos)
        break;

      // add non-empty lines
      if (loc > 0) {
        std::string line = std::string(m_partialLineBuf.begin(), m_partialLineBuf.begin() + loc);
        //printf("LINE: \"%s\"\n", line.c_str());
        m_lines.push_back(line);
      }

      // consume consecutive linebreaks
      while ((m_partialLineBuf[loc] == '\r' || m_partialLineBuf[loc] == '\n') && loc < m_partialLineBuf.size())
        ++loc;

      // shrink input buffer
      if (loc == m_partialLineBuf.size()) {
        m_partialLineBuf.clear();
        break; // done -- consumed entire buffer
      } else {
        std::string rest = std::string(m_partialLineBuf.begin() + loc, m_partialLineBuf.end());
        m_partialLineBuf.swap(rest);
      }
    }
  }
}

bool starts_with(const std::string& s, const char* test) {
  if (s.size() < strlen(test))
    return false;

  return (memcmp(s.data(), test, strlen(test)) == 0);
}

bool ends_with(const std::string& s, const char* test) {
  size_t l = strlen(test);
  if (s.size() < l)
    return false;

  return (memcmp(s.data() + (s.size() - l), test, l) == 0);
}

bool safe_write(int fd, const char* data, size_t len) {
  while (len) {
    ssize_t res = write(fd, data, len);
    if (res < 0)
      return false;

    data += res;
    len -= res;
  }
  return true;
}

std::vector<std::string> PDUControl::execCommand(const std::string& cmd) {
  std::vector<std::string> outResponseLines;

  // Throw away anything remaining on the input buffer
  drainSerialInput();
  m_lines.clear();

  safe_write(m_fd, cmd.c_str(), cmd.size());
  safe_write(m_fd, "\n", 1);

  while (true) {
    drainSerialInput();

    if (starts_with(m_partialLineBuf, "\x1b[35m[")) {
      // ANSI color codes for the prompt means the command is done.
      // Throw away the partial line and return
      m_partialLineBuf.clear();
      break; 
    }
  }

  if (m_lines.size()) {
    std::vector<std::string>::iterator it = m_lines.begin();

    // Skip the echo-back of the command that we just sent.
    // (this should always be the first response line, but this is a sanity check)
    if (ends_with(m_lines[0], cmd.c_str())) {
      ++it;
    }

    outResponseLines.insert(outResponseLines.end(), it, m_lines.end());

#if 0 // debug - dump response lines
    printf("Response lines (%zu):\n", outResponseLines.size());
    for (const auto& line : outResponseLines) {
      printf("  \"%s\"\n", line.c_str());
    }
#endif

  }
  return outResponseLines;
}

