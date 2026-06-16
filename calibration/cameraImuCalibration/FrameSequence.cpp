#include "FrameSequence.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>

namespace CameraImuCalib {

bool FrameSequence::load(const std::string& cameraDir) {
  m_frames.clear();

  DIR* dir = opendir(cameraDir.c_str());
  if (!dir) {
    fprintf(stderr, "FrameSequence::load: cannot open directory '%s': %s\n",
      cameraDir.c_str(), strerror(errno));
    return false;
  }

  const char* kExt = ".pgm";
  const size_t kExtLen = 4;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    const size_t len = strlen(entry->d_name);
    if (len <= kExtLen || strcmp(entry->d_name + len - kExtLen, kExt) != 0)
      continue;
    // Filename is a zero-padded nanosecond timestamp; parse the numeric prefix.
    char* end = nullptr;
    const uint64_t ts = strtoull(entry->d_name, &end, 10);
    if (end == entry->d_name)
      continue; // not a numeric name
    Frame f;
    f.timestampNs = ts;
    f.path = cameraDir + "/" + entry->d_name;
    m_frames.push_back(std::move(f));
  }
  closedir(dir);

  if (m_frames.empty()) {
    fprintf(stderr, "FrameSequence::load: no *.pgm frames in '%s'\n", cameraDir.c_str());
    return false;
  }

  std::sort(m_frames.begin(), m_frames.end(),
    [](const Frame& a, const Frame& b) { return a.timestampNs < b.timestampNs; });

  // Read the first frame to capture the image size.
  cv::Mat first;
  if (!loadGreyscale(0, first)) {
    fprintf(stderr, "FrameSequence::load: cannot read first frame '%s'\n", m_frames[0].path.c_str());
    return false;
  }
  m_width = static_cast<uint32_t>(first.cols);
  m_height = static_cast<uint32_t>(first.rows);

  const double span = (m_frames.back().timestampNs - m_frames.front().timestampNs) * 1e-9;
  printf("FrameSequence: %zu frames, %ux%u, spanning %.3f s (%.1f fps mean)\n",
    m_frames.size(), m_width, m_height, span,
    span > 0.0 ? static_cast<double>(m_frames.size() - 1) / span : 0.0);
  return true;
}

bool FrameSequence::loadGreyscale(size_t i, cv::Mat& outImage) const {
  if (i >= m_frames.size())
    return false;
  outImage = cv::imread(m_frames[i].path, cv::IMREAD_GRAYSCALE);
  return !outImage.empty();
}

} // namespace CameraImuCalib
