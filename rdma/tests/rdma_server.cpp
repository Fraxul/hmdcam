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
}

int main(int argc, char* argv[]) {

  boost::scoped_ptr<RDMAContext> ctx(RDMAContext::createServerContext());

  ctx->setUserEventCallback(userEventCallback);
  RDMABuffer::ptr buf1 = ctx->newManagedBuffer("buf1", 4096, kRDMABufferUsageWriteSource);
  RDMABuffer::ptr buf2 = ctx->newManagedBuffer("buf2", 4096, kRDMABufferUsageWriteSource);
  memset(buf1->data(), 0, buf1->size());
  memset(buf2->data(), 0, buf2->size());

  signal(SIGINT, signalHandler);

  uint64_t iteration = 0;
  while (!quitRequested) {
    iteration += 1;

    ctx->fireUserEvents();

    if (iteration & 1) {
      sprintf(buf1->data(), "buf1 iteration %lu", iteration);
      ctx->asyncFlushWriteBuffer(buf1);
      ctx->asyncSendUserEvent(1, SerializationBuffer("buf1"));
    } else {
      sprintf(buf2->data(), "buf2 iteration %lu", iteration);
      ctx->asyncFlushWriteBuffer(buf2);
      ctx->asyncSendUserEvent(2, SerializationBuffer("buf2"));
    }

    {
      struct timespec ts;
      ts.tv_sec = 1;
      ts.tv_nsec = 0;
      //ts.tv_nsec = 500 * 1000000UL;
      nanosleep(&ts, NULL);
    }
  }

  return 0;
}

