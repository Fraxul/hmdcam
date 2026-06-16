#include "ImuTrace.h"
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace CameraImuCalib {

bool ImuTrace::load(const std::string& path) {
  m_samples.clear();
  m_times.clear();

  FILE* f = fopen(path.c_str(), "r");
  if (!f) {
    fprintf(stderr, "ImuTrace::load: cannot open '%s': %s\n", path.c_str(), strerror(errno));
    return false;
  }

  char line[512];
  // First line is the CSV header -- skip it.
  if (!fgets(line, sizeof(line), f)) {
    fprintf(stderr, "ImuTrace::load: '%s' is empty\n", path.c_str());
    fclose(f);
    return false;
  }

  size_t lineNumber = 1;
  size_t nonMonotonic = 0;
  uint64_t lastTimestampNs = 0;
  bool haveLast = false;

  while (fgets(line, sizeof(line), f)) {
    ++lineNumber;

    uint64_t timestampNs = 0;
    double gx, gy, gz, ax, ay, az;
    // timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z  (trailing fields ignored)
    if (sscanf(line, "%lu,%lf,%lf,%lf,%lf,%lf,%lf",
          &timestampNs, &gx, &gy, &gz, &ax, &ay, &az) != 7) {
      // Tolerate blank trailing lines silently; warn on anything else.
      if (line[0] != '\0' && line[0] != '\n' && line[0] != '\r')
        fprintf(stderr, "ImuTrace::load: %s:%zu: malformed row, skipping\n",
          path.c_str(), lineNumber);
      continue;
    }

    // Reject duplicate or backward timestamps with a warning
    if (haveLast && timestampNs <= lastTimestampNs) {
      ++nonMonotonic;
      continue;
    }
    lastTimestampNs = timestampNs;
    haveLast = true;

    ImuSample s;
    s.time = static_cast<double>(timestampNs) * 1e-9; // ns -> s
    s.gyroDeg = Eigen::Vector3d(gx, gy, gz);
    s.accelG = Eigen::Vector3d(ax, ay, az);
    m_samples.push_back(s);
    m_times.push_back(s.time);
  }
  fclose(f);

  if (nonMonotonic)
    fprintf(stderr, "ImuTrace::load: WARNING: dropped %zu non-monotonic samples from '%s'\n",
      nonMonotonic, path.c_str());

  if (m_samples.size() < 2) {
    fprintf(stderr, "ImuTrace::load: '%s' yielded %zu usable samples (need >= 2)\n",
      path.c_str(), m_samples.size());
    return false;
  }

  printf("ImuTrace: loaded %zu samples spanning %.3f s (%.1f Hz mean)\n",
    m_samples.size(), lastTime() - firstTime(),
    static_cast<double>(m_samples.size() - 1) / (lastTime() - firstTime()));
  return true;
}

bool ImuTrace::findStaticBias(double maxRateDegPerSec, double minDurationSec,
  Eigen::Vector3d& outBiasDeg) const {
  // Longest contiguous run of below-threshold |gyro|. Track the best run's [begin, end).
  size_t bestBegin = 0, bestEnd = 0; // best run is [bestBegin, bestEnd)
  size_t runBegin = 0;
  for (size_t i = 0; i <= m_samples.size(); ++i) {
    const bool quiet = (i < m_samples.size()) && (m_samples[i].gyroDeg.norm() <= maxRateDegPerSec);
    if (!quiet) {
      // Run [runBegin, i) just ended; keep it if it is the longest by duration.
      if (i > runBegin) {
        const double bestDur = (bestEnd > bestBegin)
          ? (m_samples[bestEnd - 1].time - m_samples[bestBegin].time)
          : -1.0;
        const double thisDur = m_samples[i - 1].time - m_samples[runBegin].time;
        if (thisDur > bestDur) {
          bestBegin = runBegin;
          bestEnd = i;
        }
      }
      runBegin = i + 1;
    }
  }

  if (bestEnd <= bestBegin)
    return false;
  const double dur = m_samples[bestEnd - 1].time - m_samples[bestBegin].time;
  if (dur < minDurationSec)
    return false;

  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  for (size_t i = bestBegin; i < bestEnd; ++i)
    sum += m_samples[i].gyroDeg;
  outBiasDeg = sum / static_cast<double>(bestEnd - bestBegin);
  printf("ImuTrace: static segment %.2f s (%zu samples), bias init = (%.4f, %.4f, %.4f) deg/s\n",
    dur, bestEnd - bestBegin, outBiasDeg.x(), outBiasDeg.y(), outBiasDeg.z());
  return true;
}

} // namespace CameraImuCalib
