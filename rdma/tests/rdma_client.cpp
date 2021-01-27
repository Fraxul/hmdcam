#include "RDMAContext.h"
#include <stdio.h>
#include <signal.h>
#include <boost/scoped_ptr.hpp>


volatile bool quitRequested = false;

void signalHandler(int sig) {
  quitRequested = true;
}


bool buf1Ready = false;
bool buf2Ready = false;

void userEventCallback(RDMAContext* ctx, uint32_t eventID, SerializationBuffer payload) {
  printf("User event received: id=%u payload=\"%.*s\"\n", eventID, (int) payload.size(), payload.data());

  switch (eventID) {
    case 1:
      buf1Ready = true;
      break;

    case 2:
      buf2Ready = true;
      break;
  };
}

int main(int argc, char* argv[]) {
  if (argc < 1) {
    printf("usage: %s hostname\n", argv[0]);
    return -1;
  }

  char* host = argv[1];

  boost::scoped_ptr<RDMAContext> ctx(RDMAContext::createClientContext(host));
  ctx->setUserEventCallback(userEventCallback);
  RDMABuffer::ptr buf1 = ctx->newManagedBuffer("buf1", 4096, kRDMABufferUsageWriteDestination);
  RDMABuffer::ptr buf2 = ctx->newManagedBuffer("buf2", 4096, kRDMABufferUsageWriteDestination);
  memset(buf1->data(), 0, buf1->size());
  memset(buf2->data(), 0, buf2->size());

  signal(SIGINT, signalHandler);

  while (!quitRequested) {
    ctx->fireUserEvents();

    if (buf1Ready) {
      printf("buf1 ready, contents: %s\n", buf1->data());
      buf1Ready = false;
    }

    if (buf2Ready) {
      printf("buf2 ready, contents: %s\n", buf2->data());
      buf2Ready = false;
    }


    {
      struct timespec ts;
      ts.tv_sec = 0;
      ts.tv_nsec = 500 * 1000000UL;
      nanosleep(&ts, NULL);
    }
  }

  return 0;
}
