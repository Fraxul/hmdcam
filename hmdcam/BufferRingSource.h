#pragma once
#include "FramedSource.hh"
#include <pthread.h>
#include <deque>
#include <string>

class NvEncSession;

class BufferRingSource: public FramedSource {
public:
  static BufferRingSource* createNew(UsageEnvironment& env, NvEncSession*);

public:
  static EventTriggerId eventTriggerId;
  // Note that this is defined here to be a static class variable, because this code is intended to illustrate how to
  // encapsulate a *single* device - not a set of devices.
  // You can, however, redefine this to be a non-static member variable.

  // Intended to be called from outside the TaskScheduler environment -- delivers a frame.
  void asyncDeliverFrame(TaskScheduler* taskScheduler, const char* data, size_t length, struct timeval& timestamp);

protected:
  BufferRingSource(UsageEnvironment& env, NvEncSession*);
  // called only by createNew(), or by subclass constructors
  virtual ~BufferRingSource();

private:
  // redefined virtual functions:
  virtual void doGetNextFrame();
  //virtual void doStopGettingFrames(); // optional

private:
  static void deliverFrame0(void* clientData);
  void deliverFrame();

private:
  pthread_mutex_t m_qLock;
  struct Buffer {
    std::string payload;
    struct timeval pts;
  };

  std::deque<Buffer*> m_filledBuffers;
  std::deque<Buffer*> m_emptyBuffers;
  bool m_buffersOverflowing;
  NvEncSession* m_nvencSession;
  size_t m_nvencSessionCallbackId;
};

