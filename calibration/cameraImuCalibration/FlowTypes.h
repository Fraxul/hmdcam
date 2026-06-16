#pragma once
#include <Eigen/Core>
#include <vector>

namespace CameraImuCalib {

// One dense-flow correspondence for a frame pair, in pixel coordinates of the native
// (distorted) image. pixel0 is the sample location in frame i; pixel1 = pixel0 + flow is
// where it lands in frame i+1. Stage 1 undistorts both endpoints to normalized camera
// coordinates before fitting. `confidence` in [0, 1] is the OFA cost-derived weight (1.0
// when a source has no confidence signal).
struct FlowSample {
  Eigen::Vector2d pixel0;
  Eigen::Vector2d pixel1;
  double confidence = 1.0;
};

// Dense flow for one consecutive frame pair i -> i+1, with the two frames' start-of-frame
// (row-0 readout) timestamps on the camera clock. Samples are already strided/subsampled
// by the flow source to the grid the fit consumes.
struct FramePairFlow {
  double frameStartTimeA = 0.0; // t_frame(i),   seconds
  double frameStartTimeB = 0.0; // t_frame(i+1), seconds
  std::vector<FlowSample> samples;
};

} // namespace CameraImuCalib
