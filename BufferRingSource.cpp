#include "BufferRingSource.h"
#include "NvEncSession.h"
#include <GroupsockHelper.hh> // for "gettimeofday()"

const size_t kBufferCount = 32;
const size_t kBufferReserveSize = 16384;

BufferRingSource* BufferRingSource::createNew(UsageEnvironment& env, NvEncSession* session) {
  return new BufferRingSource(env, session);
}

EventTriggerId BufferRingSource::eventTriggerId = 0;

BufferRingSource::BufferRingSource(UsageEnvironment& env, NvEncSession* session) : FramedSource(env), m_nvencSession(session) {

  // Any instance-specific initialization of the device would be done here:
  pthread_mutex_init(&m_qLock, NULL);
  for (size_t i = 0; i < kBufferCount; ++i) {
    std::string* b = new std::string();
    b->reserve(kBufferReserveSize);
    m_emptyBuffers.push_back(b);
  }

  // We arrange here for our "deliverFrame" member function to be called
  // whenever the next frame of data becomes available from the device.
  //
  // If the device can be accessed as a readable socket, then one easy way to do this is using a call to
  //     envir().taskScheduler().turnOnBackgroundReadHandling( ... )
  // (See examples of this call in the "liveMedia" directory.)
  //
  // If, however, the device *cannot* be accessed as a readable socket, then instead we can implement it using 'event triggers':
  // Create an 'event trigger' for this device (if it hasn't already been done):
  if (eventTriggerId == 0) {
    eventTriggerId = envir().taskScheduler().createEventTrigger(deliverFrame0);
  }

  {
    BufferRingSource* source = this;
    TaskScheduler* scheduler = &envir().taskScheduler();
    m_nvencSessionCallbackId = session->registerEncodedFrameDeliveryCallback([source, scheduler](const char* data, size_t length) {
      source->asyncDeliverFrame(scheduler, data, length);
    });
  }
  session->start();

}

BufferRingSource::~BufferRingSource() {
  // Any instance-specific 'destruction' (i.e., resetting) of the device would be done here:

  m_nvencSession->unregisterEncodedFrameDeliveryCallback(m_nvencSessionCallbackId);
  m_nvencSession->stop();

  while (!m_filledBuffers.empty()) {
    delete m_filledBuffers.front();
    m_filledBuffers.pop_front();
  }

  while (!m_emptyBuffers.empty()) {
    delete m_emptyBuffers.front();
    m_emptyBuffers.pop_front();
  }
}

void BufferRingSource::doGetNextFrame() {
  // This function is called (by our 'downstream' object) when it asks for new data.

  // Note: If, for some reason, the source device stops being readable (e.g., it gets closed), then you do the following:
#if 0
  if (0 /* the source stops being readable */ /*%%% TO BE WRITTEN %%%*/) {
    handleClosure();
    return;
  }
#endif

  pthread_mutex_lock(&m_qLock);
  bool framesAvailable = !m_filledBuffers.empty();
  pthread_mutex_unlock(&m_qLock);

  // If a new frame of data is immediately available to be delivered, then do this now:
  if (framesAvailable) {
    deliverFrame();
  }

  // No new data is immediately available to be delivered.  We don't do anything more here.
  // Instead, our event trigger must be called (e.g., from a separate thread) when new data becomes available.
}

void BufferRingSource::deliverFrame0(void* clientData) {
  ((BufferRingSource*)clientData)->deliverFrame();
}

void BufferRingSource::deliverFrame() {
  // This function is called when new frame data is available from the device.
  // We deliver this data by copying it to the 'downstream' object, using the following parameters (class members):
  // 'in' parameters (these should *not* be modified by this function):
  //     fTo: The frame data is copied to this address.
  //         (Note that the variable "fTo" is *not* modified.  Instead,
  //          the frame data is copied to the address pointed to by "fTo".)
  //     fMaxSize: This is the maximum number of bytes that can be copied
  //         (If the actual frame is larger than this, then it should
  //          be truncated, and "fNumTruncatedBytes" set accordingly.)
  // 'out' parameters (these are modified by this function):
  //     fFrameSize: Should be set to the delivered frame size (<= fMaxSize).
  //     fNumTruncatedBytes: Should be set iff the delivered frame would have been
  //         bigger than "fMaxSize", in which case it's set to the number of bytes
  //         that have been omitted.
  //     fPresentationTime: Should be set to the frame's presentation time
  //         (seconds, microseconds).  This time must be aligned with 'wall-clock time' - i.e., the time that you would get
  //         by calling "gettimeofday()".
  //     fDurationInMicroseconds: Should be set to the frame's duration, if known.
  //         If, however, the device is a 'live source' (e.g., encoded from a camera or microphone), then we probably don't need
  //         to set this variable, because - in this case - data will never arrive 'early'.
  // Note the code below.

  if (!isCurrentlyAwaitingData()) return; // we're not ready for the data yet

  std::string* b = NULL;
  pthread_mutex_lock(&m_qLock);
  if (!m_filledBuffers.empty()) {
    b = m_filledBuffers.front();
    m_filledBuffers.pop_front();
  }
  pthread_mutex_unlock(&m_qLock);

  if (!b) {
    //printf("BufferRingSource::deliverFrame(): no buffers\n");
    return;
  }

  const char* newFrameDataStart = b->data();
  size_t newFrameSize = b->size();

  // Deliver the data here:
  if (newFrameSize > fMaxSize) {
    fFrameSize = fMaxSize;
    fNumTruncatedBytes = newFrameSize - fMaxSize;
  } else {
    fFrameSize = newFrameSize;
  }
  gettimeofday(&fPresentationTime, NULL); // If you have a more accurate time - e.g., from an encoder - then use that instead.
  // If the device is *not* a 'live source' (e.g., it comes instead from a file or buffer), then set "fDurationInMicroseconds" here.
  memmove(fTo, newFrameDataStart, fFrameSize);

  //printf("BufferRingSource::deliverFrame(): size=%zu fNumTruncatedBytes=%u\n", fFrameSize, fNumTruncatedBytes);

  if (fNumTruncatedBytes) {
    // Still have data left in this buffer, return it to the front of the queue for later
    memmove(const_cast<char*>(b->data()), b->data() + (b->size() - fNumTruncatedBytes), fNumTruncatedBytes);
    b->resize(fNumTruncatedBytes);
  }

  pthread_mutex_lock(&m_qLock);
  if (fNumTruncatedBytes) {
    m_filledBuffers.push_front(b);
  } else {
    m_emptyBuffers.push_back(b);
  }
  pthread_mutex_unlock(&m_qLock);

  // After delivering the data, inform the reader that it is now available:
  FramedSource::afterGetting(this);
}

// Called by an external client (and off the TaskScheduler thread) to deliver a buffer of frame data.
void BufferRingSource::asyncDeliverFrame(TaskScheduler* taskScheduler, const char* data, size_t length) {
  //printf("BufferRingSource(%p)::asyncDeliverFrame(%p, %zu)\n", this, data, length);

  if (data == NULL || !length)
    return;

  // TODO replace the buffer pool with a single large ring buffer?

  pthread_mutex_lock(&m_qLock);
  std::string* b = NULL;

  if (!m_emptyBuffers.empty()) {
    b = m_emptyBuffers.front();
    m_emptyBuffers.pop_front();
    m_buffersOverflowing = false;
  } else {
    // Have to steal a buffer off of the filled buffers queue, it isn't getting drained fast enough
    b = m_filledBuffers.front();
    m_filledBuffers.pop_front();

    if (!m_buffersOverflowing) {
      printf("BufferRingSource::asyncDeliverFrame: Buffer queue is full and old buffers are being overwritten\n");
    }
    m_buffersOverflowing = true;
  }

  b->resize(length);
  memcpy(const_cast<char*>(b->data()), data, length);
  m_filledBuffers.push_back(b);

  pthread_mutex_unlock(&m_qLock);

  taskScheduler->triggerEvent(eventTriggerId, this);
}

