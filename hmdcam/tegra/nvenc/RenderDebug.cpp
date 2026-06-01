#include "Render.h"
#include "common/Timing.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"

#include "MpegTSMuxer.h"
#include "NvEncSession.h"

#include <atomic>
#include <string>

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

// Remote-debug video stream.
//
// This application renders to an HMD, which is awkward to observe while
// debugging, so we expose a side-channel video stream of the debug surface to a
// desktop running ffplay or VLC. The requirements are deliberately narrow: a
// single client on the LAN, one H.264 stream, no audio, and the lowest practical
// latency from encoder output to client display. We do not need robustness to
// network loss -- if the client falls behind, dropping frames at the encoder
// input is the correct response.
//
// The NvEnc encoder emits an Annex-B H.264 byte stream (one access unit per
// callback, SPS/PPS inline at every IDR). We wrap that in a minimal MPEG-TS
// container (MpegTSMuxer) and write it to a single accepted TCP connection. TS is
// what makes the stream self-describing: its PAT/PMT tables and 0x47-sync framing
// let players autodetect the format with no client-side demuxer hints, so it
// "just works" in both ffplay and VLC:
//
//   ffplay -fflags nobuffer -flags low_delay tcp://<device-ip>:8554
//   vlc --network-caching=50 tcp://<device-ip>:8554
//
// The encoder runs only while a client is connected (it is expensive on a
// battery-powered embedded target). Starting it on connect also guarantees the
// first access unit the client sees is a fresh SPS+PPS+IDR with the PAT/PMT
// tables, so the decoder joins cleanly without any mid-stream-resync handling.

namespace {

constexpr uint16_t kDebugStreamPort = 8554;
constexpr uint64_t kRenderIntervalNs = 33333333; // 30fps

NvEncSession* s_nvencSession = nullptr;
std::string s_debugURL;

// The currently-connected client socket, or -1 when no client is connected.
// Written by the accept loop (DebugStream thread); read by the MpegTSMuxer sink,
// which runs on the encoder capture-plane DQ thread. Atomic so the hot path can
// check it without taking a lock. The fd is only ever closed after the delivery
// callback has been unregistered (see the teardown sequence in the accept loop),
// so the sink can never observe a closed-then-reused fd.
std::atomic<int> s_clientFd{-1};

// Write the whole buffer, looping over partial sends. MSG_NOSIGNAL keeps a write
// to a dropped client from raising SIGPIPE (the process installs no handler).
// The socket is blocking and has no send timeout, so send() never returns
// EWOULDBLOCK: a slow-but-alive client simply blocks us here, which is the
// intended backpressure path (see the muxer sink below). A negative/zero return
// therefore means the client is genuinely gone; we abandon the write and let the
// accept loop's recv() observe the same closure and tear the session down.
void writeAll(int fd, const void* data, size_t length) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  size_t remaining = length;
  while (remaining) {
    ssize_t n = send(fd, p, remaining, MSG_NOSIGNAL);
    if (n > 0) {
      p += static_cast<size_t>(n);
      remaining -= static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR)
      continue;
    return; // EPIPE / ECONNRESET / etc.: client gone, abandon the rest
  }
}

// The muxer batches a whole access unit (plus PSI tables at keyframes) and hands
// it to this sink as one buffer, so we issue a single socket write per frame. The
// sink reads the live client fd at call time; the muxer itself is socket-agnostic.
MpegTSMuxer s_tsMuxer([](const uint8_t* data, size_t length) {
  int fd = s_clientFd.load(std::memory_order_acquire);
  if (fd >= 0)
    writeAll(fd, data, length);
});

void* debugStreamServerThreadEntryPoint(void*) {
  pthread_setname_np(pthread_self(), "DebugStream");

  int listenFd = socket(AF_INET, SOCK_STREAM, 0);
  if (listenFd < 0) {
    fprintf(stderr, "DebugStream: socket() failed: %s\n", strerror(errno));
    return nullptr;
  }

  int one = 1;
  setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(kDebugStreamPort);

  if (bind(listenFd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    fprintf(stderr, "DebugStream: bind to port %u failed: %s\n", kDebugStreamPort, strerror(errno));
    close(listenFd);
    return nullptr;
  }

  // Backlog of 1: this is a single-client debug stream. A second connection
  // attempt while a client is connected waits in the backlog (or is refused) and
  // is accepted only once the current client disconnects.
  if (listen(listenFd, 1) < 0) {
    fprintf(stderr, "DebugStream: listen() failed: %s\n", strerror(errno));
    close(listenFd);
    return nullptr;
  }

  printf("DebugStream: H.264/MPEG-TS stream listening on tcp/%u\n", kDebugStreamPort);
  printf("Recommended client configuration for low-latency streaming:\n");
  printf("  ffplay -fflags nobuffer -flags low_delay tcp://<device-ip>:%u\n", kDebugStreamPort);
  printf("  vlc --network-caching=50 tcp://<device-ip>:%u\n", kDebugStreamPort);

  while (true) {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int fd = accept(listenFd, reinterpret_cast<struct sockaddr*>(&clientAddr), &clientAddrLen);
    if (fd < 0) {
      if (errno == EINTR)
        continue;
      fprintf(stderr, "DebugStream: accept() failed: %s\n", strerror(errno));
      break;
    }

    char clientIp[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &clientAddr.sin_addr, clientIp, sizeof(clientIp));
    printf("DebugStream: client connected from %s; starting encoder\n", clientIp);

    // Disable Nagle so each frame's packets hit the wire immediately rather than
    // being coalesced for throughput -- coalescing trades away exactly the latency
    // we are trying to minimize.
    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    // Start a fresh TS stream for this client, then publish the fd, register the
    // delivery callback, and start the encoder. Registering before start() means
    // we capture the very first access unit (a fresh encoder emits SPS+PPS+IDR
    // first), which the muxer pairs with the PAT/PMT so the client decoder syncs
    // immediately.
    s_tsMuxer.reset();
    s_clientFd.store(fd, std::memory_order_release);
    size_t callbackId = s_nvencSession->registerEncodedFrameDeliveryCallback(
      [](const char* data, size_t length, struct timeval& presentationTime) {
        if (s_clientFd.load(std::memory_order_acquire) < 0)
          return;
        s_tsMuxer.writeAccessUnit(reinterpret_cast<const uint8_t*>(data), length, presentationTime);
      });
    s_nvencSession->start();

    // Park until the client goes away. ffplay/VLC never send anything on this
    // socket, so recv() blocks until EOF (clean close -> returns 0) or a
    // reset/error (returns < 0). A slow-but-alive client does NOT unblock this
    // recv(); it blocks the send() in writeAll instead, which is the desired
    // backpressure path.
    char drain[256];
    while (true) {
      ssize_t n = recv(fd, drain, sizeof(drain), 0);
      if (n > 0)
        continue; // unexpected client->server data; ignore it
      if (n < 0 && errno == EINTR)
        continue;
      break; // EOF (0) or error (< 0): the client is gone
    }

    printf("DebugStream: client disconnected; stopping encoder\n");

    // Teardown order is load-bearing. unregister() takes the same m_callbackLock
    // that the DQ thread holds across its entire dispatch, so once it returns we
    // are guaranteed no encoder thread is still inside the muxer/writeAll touching
    // fd. Only then is it safe to clear the published fd, close it, and stop the
    // encoder.
    s_nvencSession->unregisterEncodedFrameDeliveryCallback(callbackId);
    s_clientFd.store(-1, std::memory_order_release);
    close(fd);
    s_nvencSession->stop();
  }

  close(listenFd);
  return nullptr;
}

} // namespace

void RenderInitDebugSurface(uint32_t width, uint32_t height) {
  s_nvencSession = new NvEncSession(width, height);
  //s_nvencSession->setBitrate(bitrate);
  //s_nvencSession->setFramerate(fps_n, fps_d);
  s_nvencSession->setFramerate(30, 1); // TODO derive this from the screen's framerate.

  {
    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname) - 1) != 0)
      strcpy(hostname, "<device-ip>");
    char buf[320];
    snprintf(buf, sizeof(buf), "tcp://%s:%u  (H.264/MPEG-TS -- ffplay or vlc tcp://...)", hostname, kDebugStreamPort);
    s_debugURL = buf;
  }

  // Run the stream server asynchronously; it owns the encoder lifecycle, starting
  // and stopping it as a client connects and disconnects.
  pthread_t serverThread;
  pthread_create(&serverThread, nullptr, &debugStreamServerThreadEntryPoint, nullptr);
}

bool RenderDebugSubsystemEnabled() {
  // True once RenderInitDebugSurface has created the NvEncSession. main.cpp
  // only calls that on the VK RHI backend (NvEncSession renders into a
  // VK-imported NvBufSurface). Per-frame paths use this to skip the work when
  // the subsystem isn't up.
  return s_nvencSession != nullptr;
}

RHISurface::ptr renderAcquireDebugSurface() {
  assert(s_nvencSession != nullptr);

  // Only render the debug surface while a client is connected (isRunning() is
  // true only between the start()/stop() the accept loop drives), and rate-limit
  // submission to the stream's frame interval.
  static uint64_t lastFrameSubmissionTimeNs = 0;
  if (s_nvencSession->isRunning()) {
    uint64_t now = currentTimeNs();
    if (lastFrameSubmissionTimeNs + kRenderIntervalNs <= now) {
      lastFrameSubmissionTimeNs = now;

      return s_nvencSession->acquireSurface(); // might be NULL anyway if the encoder isn't ready
    }
  }

  return RHISurface::ptr();
}

void renderSubmitDebugSurface(RHISurface::ptr debugSurface) {
  s_nvencSession->submitSurface(debugSurface);
}

const char* renderDebugURL() {
  return s_debugURL.c_str();
}
