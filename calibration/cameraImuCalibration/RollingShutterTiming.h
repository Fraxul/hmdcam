#pragma once
#include <cstdint>

// =====================================================================================
// Rolling-shutter timing model
//
// Per-row exposure-start time as a function of readout-row index r in [0, linesPerFrame):
//   t_row(i, r) = t_frame(i) + r * lineDelaySeconds
//
// `t_frame(i)` is the start-of-frame (row-0 readout) timestamp, as recorded by
// CalibrationWriter (the PGM filename, derived from oldestSensorTimestamp()).
//
// v1 aligns on the frame-center reference time:
//   t_center(i) = t_frame(i) + ((visibleHeight - 1) / 2) * lineDelaySeconds
//
// t_center is FLIP-INVARIANT -- the temporal center of the readout is the same regardless
// of readout direction -- so the v1 estimator's result does not depend on `flipReadout`.
// The flag is ingested anyway: the optional row-timed RS-aware mode needs it, and
// rowExposureTime() below is the hook that mode will use.
// =====================================================================================

namespace CameraImuCalib {

struct RollingShutterTiming {
  // Fixed line delay. For this rig: 1250 lines/frame * 90 fps -> 1 / 112500 s.
  double lineDelaySeconds = 1.0 / 112500.0;

  // Number of visible image rows (e.g. 1080).
  uint32_t visibleHeight = 1080;

  // readoutBottomToTop from calibration.yml: false = top-to-bottom (row 0 is the top
  // image row), true = bottom-to-top (readout row 0 is the bottom image row). There is
  // no separate visible-row-0 readout offset on this rig, so the first visible row is
  // always readout row 0 -- any constant offset that would introduce is absorbed into
  // delta regardless. Used only by the RS-aware mode; inert for v1.
  bool flipReadout = false;

  // Frame-center reference time for frame i (seconds), given its start-of-frame time.
  double frameCenterTime(double frameStartTime) const {
    return frameStartTime + (static_cast<double>(visibleHeight - 1) * 0.5) * lineDelaySeconds;
  }

  // Pair-midpoint camera time: the time assigned to the relative rotation
  // measured between two consecutive frames' centers.
  double pairMidpointTime(double frameStartTimeA, double frameStartTimeB) const {
    return 0.5 * (frameCenterTime(frameStartTimeA) + frameCenterTime(frameStartTimeB));
  }

  // Per-row exposure-start time for image row y_img in a frame whose start-of-frame time
  // is frameStartTime. Maps image row -> readout-row index honoring the flip,
  // then applies the line delay.
  double rowExposureTime(double frameStartTime, uint32_t imageRow) const {
    const uint32_t r = flipReadout ? (visibleHeight - 1 - imageRow) : imageRow;
    return frameStartTime + static_cast<double>(r) * lineDelaySeconds;
  }
};

} // namespace CameraImuCalib
