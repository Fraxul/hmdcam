#include "PDUComms.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "imgui_backend.h"


const size_t kPDUStatusLineSize = 1024;
char pduStatusLine[kPDUStatusLineSize];

void drawPDUStatusLine() {
  // ImGui::Text("%s", pduStatusLine);
}

void drawPDUCommandMenu() {

}

void startPDUCommsThread() {

}


