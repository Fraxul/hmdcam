#pragma once
#include <functional>
#include <stdint.h>
#include <sys/time.h>
#include <vector>

// MpegTSMuxer: a minimal MPEG-2 Transport Stream muxer for a single H.264 video
// elementary stream, used by the remote-debug video stream (RenderDebug.cpp).
//
// We stream raw H.264 to the desktop for debugging, but a bare Annex-B
// elementary stream has no container structure for a player to autodetect, so
// VLC (and other players that prioritize their TS/PS demuxers) fail to recognize
// it. Wrapping the stream in MPEG-TS gives it the PAT/PMT tables and 0x47-sync
// packet framing those players probe for, so playback "just works" with no
// client-side demuxer hints -- in both VLC and ffplay.
//
// This is deliberately the smallest muxer that produces a conformant single-
// program stream: one program, one AVC video elementary stream, PCR carried in
// the video PID. No audio, no PSI versioning, no SDT. Timestamps come from the
// monotonic per-frame PTS the encoder synthesizes (NvEncSession), anchored to
// zero at the first access unit.
class MpegTSMuxer {
public:
  // Sink for fully-formed output bytes. The muxer batches all TS packets for one
  // access unit (plus any PSI tables) and calls the sink once per access unit, so
  // the caller issues a single socket write per frame.
  using PacketSink = std::function<void(const uint8_t* data, size_t length)>;

  explicit MpegTSMuxer(PacketSink sink);

  // Reset per-stream state (continuity counters, PTS anchor). Call when a new
  // client connects so each session starts from a clean, self-consistent stream.
  void reset();

  // Packetize and emit one complete H.264 access unit in Annex-B byte-stream form
  // (start-code delimited NALs), as delivered by NvEncSession. presentationTime is
  // the encoder's synthesized monotonic PTS for this frame.
  void writeAccessUnit(const uint8_t* accessUnit, size_t length, const struct timeval& presentationTime);

private:
  void appendTables();
  void appendPSISection(uint16_t pid, uint8_t& continuityCounter, const uint8_t* section, size_t sectionLength);
  uint64_t presentationTicks(const struct timeval& presentationTime);

  static bool accessUnitIsKeyframe(const uint8_t* accessUnit, size_t length);
  static void writePTSField(uint8_t* out, uint64_t pts90k);
  static void writePCRField(uint8_t* out, uint64_t pcrBase90k);
  static uint32_t crc32Mpeg(const uint8_t* data, size_t length);

  PacketSink m_sink;
  std::vector<uint8_t> m_out; // reused per access unit to avoid per-frame allocation

  uint8_t m_patContinuity = 0;
  uint8_t m_pmtContinuity = 0;
  uint8_t m_videoContinuity = 0;

  uint64_t m_ptsAnchorMicroseconds = 0;
  bool m_hasPtsAnchor = false;

  uint64_t m_lastTablesTicks = 0; // PCR ticks at the last PAT/PMT emission
  bool m_tablesEmitted = false; // false until the first PAT/PMT of this stream
};
