#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace CameraImuCalib {

// Enumerates the PGM frames CalibrationWriter wrote into one cameraN/ directory. Filenames
// are zero-padded 16-digit nanosecond timestamps on the same zero-based clock as imu.csv.
// Frames are sorted by timestamp; gaps from frame decimation (writeInterval) or dropped
// frames are preserved -- consumers derive each pair's dt from the actual timestamps rather
// than assuming a fixed frame period.
class FrameSequence {
public:
  // Enumerate and sort *.pgm in cameraDir; reads the first image to capture the frame size.
  // Returns false if the directory is unreadable or contains no parseable frames.
  bool load(const std::string& cameraDir);

  size_t frameCount() const { return m_frames.size(); }
  double timestampSeconds(size_t i) const { return m_frames[i].timestampNs * 1e-9; }
  uint64_t timestampNs(size_t i) const { return m_frames[i].timestampNs; }
  uint32_t width() const { return m_width; }
  uint32_t height() const { return m_height; }

  // Load frame i as an 8-bit greyscale cv::Mat. Returns false on read failure.
  bool loadGreyscale(size_t i, cv::Mat& outImage) const;

private:
  struct Frame {
    uint64_t timestampNs = 0;
    std::string path;
  };
  std::vector<Frame> m_frames;
  uint32_t m_width = 0, m_height = 0;
};

} // namespace CameraImuCalib
