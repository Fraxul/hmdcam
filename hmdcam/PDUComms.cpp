#include "PDUComms.h"
#include "PDUControl.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "imgui_backend.h"


const size_t kPDUStatusLineSize = 1024;
char pduStatusLine[kPDUStatusLineSize];
PDUControl* pduControl = nullptr;

void drawPDUStatusLine() {
  if (!pduControl)
    return;

  if (pduControl->m_state.valid()) {
    pduControl->m_state.toString(pduStatusLine, kPDUStatusLineSize);
    ImGui::TextUnformatted(pduStatusLine);
  }
}

void drawPDUCommandMenu() {

}

void startPDUCommsThread() {
  pduControl = new PDUControl();
}


