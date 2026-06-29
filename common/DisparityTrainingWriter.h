#pragma once
#include "common/AsyncGpuDumpRing.h"
#include <cstdint>
#include <cuda.h>
#include <glm/gtc/quaternion.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

class DepthMapGenerator;
class CameraSystem;

// Captures stereo training samples for neural disparity refinement / smoothing. Per stereo view,
// one sample bundles:
//   - rectified left + right RGB views   (algoInput resolution, written as binary PPM / P6)
//   - the raw disparity surface (uint16) (internal resolution, written as binary PGM / P5, maxval
//     65535, big-endian per the Netpbm spec)
//   - the raw cost surface (uint8)       (internal resolution, written as binary PGM / P5)
// plus a per-sample CSV row recording the camera's inter-frame rotation (the metadata a temporal
// refinement model needs, analogous to CalibrationWriter's imu.csv).
//
// Owned by DepthMapGenerator. The pixel surfaces are streamed to disk with minimal CPU overhead via
// an AsyncGpuDumpRing: a packed [leftRGB | rightRGB | disparity | cost] host slot is filled by DtoH
// copies and written by a worker thread. The metadata CSV row is tiny and written synchronously on
// the calling (main) thread, so worker threads never contend on the CSV fd.
//
// Backends drive the writer in two phases to keep color aligned with the disparity it produced:
//   - At submit time, render the rectified RGB for the frame being submitted to the matcher.
//   - When that frame's disparity/cost return, acquireSlot(), copy all four surfaces in, and
//     submit() -- so each sample pairs RGB with the disparity computed from those same images.
class DisparityTrainingWriter {
public:
  DisparityTrainingWriter(DepthMapGenerator*);
  ~DisparityTrainingWriter();

  DisparityTrainingWriter(const DisparityTrainingWriter&) = delete;
  DisparityTrainingWriter& operator=(const DisparityTrainingWriter&) = delete;

  bool isActive() const { return m_active; }

  // Creates a timestamped output directory tree (one subdir per view, each with a metadata.csv) and
  // begins capturing, or drains in-flight writes and closes the outputs. Resets the frame index and
  // counters on activation.
  void setActive(bool);

  // Advance the shared per-frame index used for filenames.
  // Call once per generator frame, before backends submit data.
  // May deactivate the writer if the maximum frame count limit has been reached.
  void beginFrame();
  uint64_t currentFrameIndex() const { return m_frameIndex; }

  // Acquire a host slot for one sample, or nullptr if inactive or the ring is exhausted (the latter
  // counts as a dropped sample). On success the caller fills it via copyColor/copyDisparity/copyCost
  // and then calls submit().
  AsyncGpuDumpRing::Slot* acquireSlot();

  // DtoH copies into the slot's packed layout. All issued on `stream`; the caller must run these
  // before submit() records the slot's completion event on the same stream.
  void copyColor(AsyncGpuDumpRing::Slot*, const cv::cuda::GpuMat& leftRGB, const cv::cuda::GpuMat& rightRGB, CUstream stream);
  void copyDisparity(AsyncGpuDumpRing::Slot*, const cv::cuda::GpuMat& disparityU16, CUstream stream);
  void copyCost(AsyncGpuDumpRing::Slot*, const cv::cuda::GpuMat& costU8, CUstream stream);

  // Records the slot's copy-completion event on `stream`, writes the metadata CSV row for this
  // sample (synchronously), and dispatches the async image write. Consumes the slot.
  void submit(AsyncGpuDumpRing::Slot*, size_t viewIndex, uint64_t frameIndex, uint64_t timestampNs, const glm::quat& interframeRotation, CUstream stream);

  void renderIMGUI();

  uint32_t algoInputWidth() const { return m_algoInputWidth; }
  uint32_t algoInputHeight() const { return m_algoInputHeight; }
  uint32_t internalWidth() const { return m_internalWidth; }
  uint32_t internalHeight() const { return m_internalHeight; }
  size_t viewCount() const { return m_viewOutput.size(); }

private:
  DepthMapGenerator* m_depthMapGenerator;
  CameraSystem* m_cameraSystem;

  DepthMapGenerator* depthMapGenerator() const { return m_depthMapGenerator; }
  CameraSystem* cameraSystem() const { return m_cameraSystem; }

  void closeOutputDescriptors();
  // Worker-thread serialization of one filled slot. Byte-swaps the disparity plane in place (to
  // big-endian) and writes the four image files under viewDirFd.
  void writeSample(AsyncGpuDumpRing::Slot*, int viewDirFd, uint64_t frameIndex);

  uint32_t m_algoInputWidth, m_algoInputHeight;
  uint32_t m_internalWidth, m_internalHeight;

  // Packed slot byte layout: [leftRGB][rightRGB][disparity][cost], each region tightly packed.
  size_t m_rgbBytes; // per eye (algoInput W*H*3)
  size_t m_disparityBytes; // internal W*H*2
  size_t m_costBytes; // internal W*H*1
  size_t m_leftRGBOffset, m_rightRGBOffset, m_disparityOffset, m_costOffset;

  bool m_active = false;
  uint64_t m_frameIndex = 0;
  uint64_t m_frameLimit = 0; // Maximum frames to write before auto-deactivating; 0 for infinite.
  uint32_t m_writtenSamples = 0;
  uint32_t m_droppedSamples = 0;

  int m_rootDirFd = -1;
  struct ViewOutput {
    int dirFd = -1; // per-view output directory
    int metadataFd = -1; // per-view metadata.csv
  };
  std::vector<ViewOutput> m_viewOutput;

  AsyncGpuDumpRing* m_dumpRing = nullptr;
};
