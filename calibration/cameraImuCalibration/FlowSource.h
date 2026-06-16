#pragma once
#include "FlowTypes.h"
#include <cstddef>

namespace CameraImuCalib {

// Swappable source of dense optical flow per consecutive frame pair. The estimator
// consumes only this interface, so the on-disk/hardware format can change (synthetic
// projection, OFA over recorded PGMs, or a future flow-file loader) without touching the
// estimator. Implementations are responsible for striding the dense field down to the grid
// the fit consumes and for populating per-sample confidence when available.
class FlowSource {
public:
  virtual ~FlowSource() = default;

  // Number of consecutive frame pairs available (frameCount - 1 for a contiguous capture).
  virtual size_t pairCount() const = 0;

  // Fill `outPair` with the flow for pair index `pairIndex`. Returns false if the pair is
  // unavailable (e.g. a frame failed to load); the estimator skips false pairs.
  virtual bool pair(size_t pairIndex, FramePairFlow& outPair) = 0;
};

} // namespace CameraImuCalib
