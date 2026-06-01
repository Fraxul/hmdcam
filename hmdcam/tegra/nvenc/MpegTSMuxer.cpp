#include "MpegTSMuxer.h"
#include <string.h>

namespace {

constexpr size_t kTSPacketSize = 188;

// PID assignments for our single-program stream.
constexpr uint16_t kPidPAT = 0x0000; // Program Association Table (fixed by spec)
constexpr uint16_t kPidPMT = 0x1000; // Program Map Table
constexpr uint16_t kPidVideo = 0x0100; // H.264 elementary stream; also carries the PCR

constexpr uint8_t kStreamTypeH264 = 0x1B; // ISO/IEC 14496-10 AVC video in a PMT
constexpr uint8_t kStreamIdVideo = 0xE0; // PES stream_id for the first video stream
constexpr uint16_t kProgramNumber = 1;

// Access Unit Delimiter NAL (type 9, primary_pic_type = 7). We prepend one to
// each access unit. PES packet boundaries (PUSI) already delimit access units,
// but an explicit AUD helps the H.264 packetizers in some players (notably VLC)
// find access-unit boundaries unambiguously, which improves first-frame sync.
constexpr uint8_t kAccessUnitDelimiter[6] = {0x00, 0x00, 0x00, 0x01, 0x09, 0xF0};

// Present each frame this many 90 kHz ticks after its PCR. The PCR is the frame's
// real (capture-time) clock value; offsetting the PTS forward gives the client
// decoder a buffering cushion, so normal delivery jitter (notably keyframe bursts)
// does not make pictures arrive after their presentation deadline. Without it,
// players that schedule strictly against the clock -- VLC in particular -- declare
// pictures late and periodically re-buffer a full second. 3000 ticks is ~37 ms,
// about one frame at 30 fps. This lead is the dominant contributor to
// glass-to-glass latency: lower it for less latency, raise it for more jitter
// tolerance. (ffplay tolerates a zero cushion, but VLC does not.)
constexpr uint64_t kPresentationLeadTicks = 3000;

// Re-emit the PAT/PMT at least this often (in 90 kHz ticks) so a client always
// sees the tables promptly even when keyframes are rare. With periodic
// intra-refresh the only IDR is the first frame, so a keyframe-only resend would
// transmit the tables just once; this time-based resend keeps the stream
// self-describing regardless of keyframe cadence. 45000 ticks = 500 ms.
constexpr uint64_t kTablesResendIntervalTicks = 45000;

} // namespace

MpegTSMuxer::MpegTSMuxer(PacketSink sink) :
  m_sink(std::move(sink)) {
}

void MpegTSMuxer::reset() {
  m_patContinuity = 0;
  m_pmtContinuity = 0;
  m_videoContinuity = 0;
  m_ptsAnchorMicroseconds = 0;
  m_hasPtsAnchor = false;
  m_lastTablesTicks = 0;
  m_tablesEmitted = false;
}

uint64_t MpegTSMuxer::presentationTicks(const struct timeval& presentationTime) {
  uint64_t microseconds = (static_cast<uint64_t>(presentationTime.tv_sec) * 1000000ull) + presentationTime.tv_usec;
  if (!m_hasPtsAnchor) {
    m_ptsAnchorMicroseconds = microseconds;
    m_hasPtsAnchor = true;
  }
  // Anchor to zero at the first frame to keep PTS/PCR values small, then convert
  // microseconds to the 90 kHz TS timestamp clock: ticks = us * 90000 / 1e6 = us * 9 / 100.
  uint64_t delta = microseconds - m_ptsAnchorMicroseconds;
  return (delta * 9ull) / 100ull;
}

/*static*/ bool MpegTSMuxer::accessUnitIsKeyframe(const uint8_t* accessUnit, size_t length) {
  // Scan for an IDR slice (NAL type 5) or SPS (type 7). The encoder emits SPS/PPS
  // inline only at IDR frames, so either marks a random-access point. We search
  // for the 3-byte start-code prefix, which also matches inside 4-byte start codes.
  for (size_t i = 0; i + 4 <= length; ++i) {
    if (accessUnit[i] == 0x00 && accessUnit[i + 1] == 0x00 && accessUnit[i + 2] == 0x01) {
      uint8_t nalType = accessUnit[i + 3] & 0x1F;
      if (nalType == 5 || nalType == 7)
        return true;
      i += 2; // skip past this start code
    }
  }
  return false;
}

/*static*/ void MpegTSMuxer::writePTSField(uint8_t* out, uint64_t pts90k) {
  // 5-byte PTS field with the '0010' prefix (PTS only) and the three marker bits.
  out[0] = static_cast<uint8_t>(0x21 | (((pts90k >> 30) & 0x07) << 1));
  out[1] = static_cast<uint8_t>((pts90k >> 22) & 0xFF);
  out[2] = static_cast<uint8_t>(0x01 | (((pts90k >> 15) & 0x7F) << 1));
  out[3] = static_cast<uint8_t>((pts90k >> 7) & 0xFF);
  out[4] = static_cast<uint8_t>(0x01 | ((pts90k & 0x7F) << 1));
}

/*static*/ void MpegTSMuxer::writePCRField(uint8_t* out, uint64_t pcrBase90k) {
  // 6-byte PCR: 33-bit base (90 kHz), 6 reserved bits (all 1), 9-bit extension
  // (27 MHz remainder). We carry no sub-90kHz precision, so the extension is 0.
  constexpr uint32_t extension = 0;
  out[0] = static_cast<uint8_t>((pcrBase90k >> 25) & 0xFF);
  out[1] = static_cast<uint8_t>((pcrBase90k >> 17) & 0xFF);
  out[2] = static_cast<uint8_t>((pcrBase90k >> 9) & 0xFF);
  out[3] = static_cast<uint8_t>((pcrBase90k >> 1) & 0xFF);
  out[4] = static_cast<uint8_t>(((pcrBase90k & 0x01) << 7) | 0x7E | ((extension >> 8) & 0x01));
  out[5] = static_cast<uint8_t>(extension & 0xFF);
}

/*static*/ uint32_t MpegTSMuxer::crc32Mpeg(const uint8_t* data, size_t length) {
  // MPEG-2 systems CRC-32: poly 0x04C11DB7, init 0xFFFFFFFF, MSB-first, no final XOR.
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc ^= static_cast<uint32_t>(data[i]) << 24;
    for (int bit = 0; bit < 8; ++bit) {
      if (crc & 0x80000000u)
        crc = (crc << 1) ^ 0x04C11DB7u;
      else
        crc <<= 1;
    }
  }
  return crc;
}

void MpegTSMuxer::appendPSISection(uint16_t pid, uint8_t& continuityCounter, const uint8_t* section, size_t sectionLength) {
  // A PSI table fits in a single TS packet here. Payload-only packet: PUSI set,
  // a pointer_field of 0, the section, then 0xFF stuffing to fill 188 bytes.
  uint8_t packet[kTSPacketSize];
  packet[0] = 0x47;
  packet[1] = static_cast<uint8_t>(0x40 | ((pid >> 8) & 0x1F)); // PUSI = 1
  packet[2] = static_cast<uint8_t>(pid & 0xFF);
  packet[3] = static_cast<uint8_t>(0x10 | (continuityCounter & 0x0F)); // payload only
  packet[4] = 0x00; // pointer_field

  memcpy(packet + 5, section, sectionLength);
  size_t used = 5 + sectionLength;
  memset(packet + used, 0xFF, kTSPacketSize - used);

  continuityCounter = (continuityCounter + 1) & 0x0F;
  m_out.insert(m_out.end(), packet, packet + kTSPacketSize);
}

void MpegTSMuxer::appendTables() {
  // PAT: one program (program 1) whose PMT lives on kPidPMT.
  uint8_t pat[16];
  pat[0] = 0x00; // table_id
  pat[1] = 0xB0; // syntax indicator + section_length high nibble (0)
  pat[2] = 0x0D; // section_length = 13
  pat[3] = 0x00; // transport_stream_id high
  pat[4] = 0x01; // transport_stream_id low
  pat[5] = 0xC1; // reserved + version 0 + current_next = 1
  pat[6] = 0x00; // section_number
  pat[7] = 0x00; // last_section_number
  pat[8] = static_cast<uint8_t>((kProgramNumber >> 8) & 0xFF);
  pat[9] = static_cast<uint8_t>(kProgramNumber & 0xFF);
  pat[10] = static_cast<uint8_t>(0xE0 | ((kPidPMT >> 8) & 0x1F)); // reserved + PMT PID high
  pat[11] = static_cast<uint8_t>(kPidPMT & 0xFF); // PMT PID low
  uint32_t patCrc = crc32Mpeg(pat, 12);
  pat[12] = static_cast<uint8_t>((patCrc >> 24) & 0xFF);
  pat[13] = static_cast<uint8_t>((patCrc >> 16) & 0xFF);
  pat[14] = static_cast<uint8_t>((patCrc >> 8) & 0xFF);
  pat[15] = static_cast<uint8_t>(patCrc & 0xFF);
  appendPSISection(kPidPAT, m_patContinuity, pat, sizeof(pat));

  // PMT: program 1 is a single H.264 elementary stream; PCR shares the video PID.
  uint8_t pmt[21];
  pmt[0] = 0x02; // table_id
  pmt[1] = 0xB0; // syntax indicator + section_length high nibble (0)
  pmt[2] = 0x12; // section_length = 18
  pmt[3] = static_cast<uint8_t>((kProgramNumber >> 8) & 0xFF);
  pmt[4] = static_cast<uint8_t>(kProgramNumber & 0xFF);
  pmt[5] = 0xC1; // reserved + version 0 + current_next = 1
  pmt[6] = 0x00; // section_number
  pmt[7] = 0x00; // last_section_number
  pmt[8] = static_cast<uint8_t>(0xE0 | ((kPidVideo >> 8) & 0x1F)); // reserved + PCR PID high
  pmt[9] = static_cast<uint8_t>(kPidVideo & 0xFF); // PCR PID low
  pmt[10] = 0xF0; // reserved + program_info_length high (0)
  pmt[11] = 0x00; // program_info_length low (0)
  pmt[12] = kStreamTypeH264; // stream_type
  pmt[13] = static_cast<uint8_t>(0xE0 | ((kPidVideo >> 8) & 0x1F)); // reserved + elementary PID high
  pmt[14] = static_cast<uint8_t>(kPidVideo & 0xFF); // elementary PID low
  pmt[15] = 0xF0; // reserved + ES_info_length high (0)
  pmt[16] = 0x00; // ES_info_length low (0)
  uint32_t pmtCrc = crc32Mpeg(pmt, 17);
  pmt[17] = static_cast<uint8_t>((pmtCrc >> 24) & 0xFF);
  pmt[18] = static_cast<uint8_t>((pmtCrc >> 16) & 0xFF);
  pmt[19] = static_cast<uint8_t>((pmtCrc >> 8) & 0xFF);
  pmt[20] = static_cast<uint8_t>(pmtCrc & 0xFF);
  appendPSISection(kPidPMT, m_pmtContinuity, pmt, sizeof(pmt));
}

void MpegTSMuxer::writeAccessUnit(const uint8_t* accessUnit, size_t length, const struct timeval& presentationTime) {
  if (!accessUnit || !length)
    return;

  // PCR is the frame's real timestamp; PTS leads it by the buffering cushion.
  const uint64_t pcr90k = presentationTicks(presentationTime);
  const uint64_t pts90k = pcr90k + kPresentationLeadTicks;
  const bool keyframe = accessUnitIsKeyframe(accessUnit, length);

  m_out.clear();

  // Emit the PAT/PMT on the first frame, on any keyframe, and periodically by
  // time thereafter. The first/keyframe cases get the tables to the client
  // immediately; the time-based resend keeps the stream self-describing when
  // keyframes are rare (periodic intra-refresh emits only the one initial IDR).
  if (!m_tablesEmitted || keyframe || (pcr90k - m_lastTablesTicks) >= kTablesResendIntervalTicks) {
    appendTables();
    m_lastTablesTicks = pcr90k;
    m_tablesEmitted = true;
  }

  // Build the PES payload: header + AUD + the Annex-B access unit.
  uint8_t pesHeader[14];
  pesHeader[0] = 0x00;
  pesHeader[1] = 0x00;
  pesHeader[2] = 0x01; // packet_start_code_prefix
  pesHeader[3] = kStreamIdVideo; // stream_id
  pesHeader[4] = 0x00; // PES_packet_length = 0 (unbounded; permitted for video)
  pesHeader[5] = 0x00;
  pesHeader[6] = 0x84; // '10' marker, data_alignment_indicator = 1
  pesHeader[7] = 0x80; // PTS_DTS_flags = '10' (PTS only)
  pesHeader[8] = 0x05; // PES_header_data_length
  writePTSField(pesHeader + 9, pts90k);

  const size_t payloadTotal = sizeof(pesHeader) + sizeof(kAccessUnitDelimiter) + length;

  // Packetize the PES payload across 188-byte TS packets. The first packet of the
  // access unit sets PUSI and carries an adaptation field with the PCR (and the
  // random-access flag on keyframes). The final packet is padded to 188 bytes
  // with adaptation-field stuffing. We pull payload from three logical segments
  // (PES header, AUD, access unit) without concatenating them into one buffer.
  size_t position = 0;
  bool firstPacket = true;
  while (position < payloadTotal) {
    uint8_t packet[kTSPacketSize];
    const size_t remaining = payloadTotal - position;

    packet[0] = 0x47;
    uint8_t byte1 = (kPidVideo >> 8) & 0x1F;
    if (firstPacket)
      byte1 |= 0x40; // PUSI
    packet[1] = byte1;
    packet[2] = static_cast<uint8_t>(kPidVideo & 0xFF);

    size_t payloadOffset; // index in packet[] where payload bytes begin
    size_t payloadLength; // number of payload bytes this packet carries

    if (firstPacket) {
      // Adaptation field carrying flags + PCR (8 bytes after the header), plus
      // stuffing if this lone packet does not fill up with payload.
      constexpr size_t kPcrAdaptationContent = 7; // flags(1) + PCR(6)
      constexpr size_t kMaxPayloadWithPcr = kTSPacketSize - 4 - 1 - kPcrAdaptationContent; // 176
      size_t stuffing = (remaining < kMaxPayloadWithPcr) ? (kMaxPayloadWithPcr - remaining) : 0;
      size_t adaptationLength = kPcrAdaptationContent + stuffing;

      packet[3] = static_cast<uint8_t>(0x30 | (m_videoContinuity & 0x0F)); // adaptation + payload
      packet[4] = static_cast<uint8_t>(adaptationLength);
      packet[5] = keyframe ? 0x50 : 0x10; // PCR_flag (+ random_access_indicator on keyframes)
      writePCRField(packet + 6, pcr90k);
      if (stuffing)
        memset(packet + 12, 0xFF, stuffing);

      payloadOffset = 4 + 1 + adaptationLength;
      payloadLength = (remaining < kMaxPayloadWithPcr) ? remaining : kMaxPayloadWithPcr;
    } else if (remaining >= kTSPacketSize - 4) {
      // Full payload-only packet.
      packet[3] = static_cast<uint8_t>(0x10 | (m_videoContinuity & 0x0F));
      payloadOffset = 4;
      payloadLength = kTSPacketSize - 4; // 184
    } else {
      // Final packet: adaptation field used purely to stuff the tail to 188 bytes.
      size_t adaptationLength = (kTSPacketSize - 4 - 1) - remaining; // = 183 - remaining
      packet[3] = static_cast<uint8_t>(0x30 | (m_videoContinuity & 0x0F));
      packet[4] = static_cast<uint8_t>(adaptationLength);
      if (adaptationLength >= 1) {
        packet[5] = 0x00; // adaptation flags: none set
        if (adaptationLength >= 2)
          memset(packet + 6, 0xFF, adaptationLength - 1);
      }
      payloadOffset = 4 + 1 + adaptationLength;
      payloadLength = remaining;
    }

    // Copy payloadLength bytes starting at logical offset `position`, spanning the
    // PES header, the AUD, and the access unit as needed.
    size_t copied = 0;
    while (copied < payloadLength) {
      size_t logical = position + copied;
      const uint8_t* segment;
      size_t segmentBase;
      size_t segmentLength;
      if (logical < sizeof(pesHeader)) {
        segment = pesHeader;
        segmentBase = 0;
        segmentLength = sizeof(pesHeader);
      } else if (logical < sizeof(pesHeader) + sizeof(kAccessUnitDelimiter)) {
        segment = kAccessUnitDelimiter;
        segmentBase = sizeof(pesHeader);
        segmentLength = sizeof(kAccessUnitDelimiter);
      } else {
        segment = accessUnit;
        segmentBase = sizeof(pesHeader) + sizeof(kAccessUnitDelimiter);
        segmentLength = length;
      }
      size_t within = logical - segmentBase;
      size_t chunk = segmentLength - within;
      if (chunk > payloadLength - copied)
        chunk = payloadLength - copied;
      memcpy(packet + payloadOffset + copied, segment + within, chunk);
      copied += chunk;
    }

    position += payloadLength;
    m_videoContinuity = (m_videoContinuity + 1) & 0x0F;
    firstPacket = false;
    m_out.insert(m_out.end(), packet, packet + kTSPacketSize);
  }

  m_sink(m_out.data(), m_out.size());
}
