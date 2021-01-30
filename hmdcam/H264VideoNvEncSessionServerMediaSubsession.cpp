/**********
This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version. (See <http://www.gnu.org/copyleft/lesser.html>.)

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
more details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
**********/
// "liveMedia"
// Copyright (c) 1996-2020 Live Networks, Inc.  All rights reserved.
// A 'ServerMediaSubsession' object that creates new, unicast, "RTPSink"s
// on demand, from a H264 video file.
// Implementation

#include "H264VideoNvEncSessionServerMediaSubsession.h"
#include "H264VideoRTPSink.hh"
#include "H264VideoStreamDiscreteFramer.hh"

#include "NvEncSession.h"
#include "BufferRingSource.h"

H264VideoNvEncSessionServerMediaSubsession*
H264VideoNvEncSessionServerMediaSubsession::createNew(UsageEnvironment& env, NvEncSession* source) {
  return new H264VideoNvEncSessionServerMediaSubsession(env, source);
}

H264VideoNvEncSessionServerMediaSubsession::H264VideoNvEncSessionServerMediaSubsession(UsageEnvironment& env, NvEncSession* source)
  : OnDemandServerMediaSubsession(env, /*reuseFirstSource=*/ true),
    fAuxSDPLine(NULL), fDoneFlag(0), fDummyRTPSink(NULL), fNvEncSession(source) {
}

H264VideoNvEncSessionServerMediaSubsession::~H264VideoNvEncSessionServerMediaSubsession() {
  delete[] fAuxSDPLine;
}

static void afterPlayingDummy(void* clientData) {
  H264VideoNvEncSessionServerMediaSubsession* subsess = (H264VideoNvEncSessionServerMediaSubsession*)clientData;
  subsess->afterPlayingDummy1();
}

void H264VideoNvEncSessionServerMediaSubsession::afterPlayingDummy1() {
  // Unschedule any pending 'checking' task:
  envir().taskScheduler().unscheduleDelayedTask(nextTask());
  // Signal the event loop that we're done:
  setDoneFlag();
}

static void checkForAuxSDPLine(void* clientData) {
  H264VideoNvEncSessionServerMediaSubsession* subsess = (H264VideoNvEncSessionServerMediaSubsession*)clientData;
  subsess->checkForAuxSDPLine1();
}

void H264VideoNvEncSessionServerMediaSubsession::checkForAuxSDPLine1() {
  nextTask() = NULL;

  char const* dasl;
  if (fAuxSDPLine != NULL) {
    // Signal the event loop that we're done:
    setDoneFlag();
  } else if (fDummyRTPSink != NULL && (dasl = fDummyRTPSink->auxSDPLine()) != NULL) {
    fAuxSDPLine = strDup(dasl);
    fDummyRTPSink = NULL;

    // Signal the event loop that we're done:
    setDoneFlag();
  } else if (!fDoneFlag) {
    // try again after a brief delay:
    int uSecsToDelay = 100000; // 100 ms
    nextTask() = envir().taskScheduler().scheduleDelayedTask(uSecsToDelay,
			      (TaskFunc*)checkForAuxSDPLine, this);
  }
}

char const* H264VideoNvEncSessionServerMediaSubsession::getAuxSDPLine(RTPSink* rtpSink, FramedSource* inputSource) {
  if (fAuxSDPLine != NULL) return fAuxSDPLine; // it's already been set up (for a previous client)

  if (fDummyRTPSink == NULL) { // we're not already setting it up for another, concurrent stream
    // Note: For H264 video files, the 'config' information ("profile-level-id" and "sprop-parameter-sets") isn't known
    // until we start reading the file.  This means that "rtpSink"s "auxSDPLine()" will be NULL initially,
    // and we need to start reading data from our file until this changes.
    fDummyRTPSink = rtpSink;

    // Start reading the file:
    fDummyRTPSink->startPlaying(*inputSource, afterPlayingDummy, this);

    // Check whether the sink's 'auxSDPLine()' is ready:
    checkForAuxSDPLine(this);
  }

  envir().taskScheduler().doEventLoop(&fDoneFlag);

  return fAuxSDPLine;
}

FramedSource* H264VideoNvEncSessionServerMediaSubsession::createNewStreamSource(unsigned /*clientSessionId*/, unsigned& estBitrate) {
  estBitrate = fNvEncSession->bitrate() / 1000; // kbps, estimate
  if (!estBitrate)
    estBitrate = 8000;

  // Create a framer for the Video Elementary Stream:
  // TODO: this needs to be H264VideoStreamDiscreteFramer to use our PTSes, which means we need to strip the start codes and probably separate the SPS and PPS (since they're delivered in the same buffer)
  return H264VideoStreamDiscreteFramer::createNew(envir(), BufferRingSource::createNew(envir(), fNvEncSession));
}

RTPSink* H264VideoNvEncSessionServerMediaSubsession
::createNewRTPSink(Groupsock* rtpGroupsock,
		   unsigned char rtpPayloadTypeIfDynamic,
		   FramedSource* /*inputSource*/) {
  return H264VideoRTPSink::createNew(envir(), rtpGroupsock, rtpPayloadTypeIfDynamic);
}
