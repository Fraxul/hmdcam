#pragma once
#include <stdint.h>
#include <string>
#include <vector>

class PDUControl {
public:

  PDUControl();
  ~PDUControl();

  std::vector<std::string> execCommand(const std::string& cmd);

  bool isOpen() const { return m_fd >= 0; }

  bool tryOpenSerial();

protected:
  int m_fd = -1;
  std::string m_serialPort;

  void drainSerialInput();

  std::vector<std::string> m_lines;

  std::string m_partialLineBuf;
};

