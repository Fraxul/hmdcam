#include "PDUComms.h"
#include "common/PDUControl.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "imgui_backend.h"


const size_t kPDUStatusLineSize = 1024;
char pduStatusLine[kPDUStatusLineSize];


void drawPDUStatusLine() {
  ImGui::Text("%s", pduStatusLine);
}

void drawPDUCommandMenu() {

}

void* PDUCommsThread(void*) {
  pthread_setname_np(pthread_self(), "PDU-Comms");
  memset(pduStatusLine, 0, sizeof(pduStatusLine));

  PDUControl pduControl;

  while (true) {
    // Make sure the port is open
    if (!pduControl.isOpen()) {
      if (!pduControl.tryOpenSerial()) {
        strcpy(pduStatusLine, "PDU comms failure");
        sleep(10);
        continue;
      }
    }


    // Update status line
    {
      auto statusLines = pduControl.execCommand("pwr");
      std::string lineBuf;
      for (const auto& line : statusLines) {
        lineBuf += line + "\n";
      }
      strncpy(pduStatusLine, lineBuf.c_str(), kPDUStatusLineSize);
    }


    sleep(2); // rate-limit.
  }
}

void startPDUCommsThread() {
  pthread_t thread;
  pthread_create(&thread, NULL, PDUCommsThread, NULL);
}


