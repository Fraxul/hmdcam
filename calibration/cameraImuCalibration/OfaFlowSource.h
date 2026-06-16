#pragma once
#include "FlowSource.h"
#include "FrameSequence.h"
#include <cstdint>
#include <vector>

#include <cuda.h>
#include "nvmedia_iofa.h"

class NvSciCudaInteropBuffer;
class NvSciCudaInteropSync;

namespace CameraImuCalib {

// Dense optical-flow source backed by the Jetson Orin Optical Flow Accelerator (OFA),
// running pyramid optical flow (PYDOF) over consecutive frames of a recorded PGM sequence.
// Reuses the in-tree NvSci/CUDA interop (common/tegra) and RHICUDA context. Tegra-only.
//
// pair(i) runs OFA on frames (i, i+1), reads back the full-resolution level-0 flow surface
// (Signed_R16G16, Q10.5 fixed point -> /32 px), strides it to the fit grid, and emits one
// FlowSample per strided pixel. dt is derived from the frames' actual timestamps, so frame
// decimation / drops are handled naturally.
class OfaFlowSource : public FlowSource {
public:
  // downsampleShift reduces the resolution OFA runs at: processing size = input >> shift
  // (0=full, 1=half, 2=quarter, 3=eighth). OFA cost scales with processing resolution, so
  // this is the main speed lever. (OFA PYDOF does not support output gridSize > 1x1 -- that
  // is a stereo/SGM-only feature -- so input downsampling is how we trade resolution for
  // speed.) `stride` further subsamples the flow output. Flow is mapped back to full-resolution
  // pixel coordinates before the fit, so the fixed intrinsics still apply.
  OfaFlowSource(const FrameSequence& frames, int stride, int downsampleShift);
  ~OfaFlowSource() override;

  // Allocate the OFA engine and surface pyramids for the sequence's frame size. Returns
  // false on hardware/setup failure.
  bool initialize();

  size_t pairCount() const override {
    return m_frames.frameCount() > 0 ? m_frames.frameCount() - 1 : 0;
  }
  bool pair(size_t pairIndex, FramePairFlow& outPair) override;

private:
  NvSciCudaInteropBuffer* makeRegisteredSurface(uint32_t width, uint32_t height,
    int colorFmt, bool setColorStd);
  bool runOfa(const cv::Mat& image0, const cv::Mat& image1);

  const FrameSequence& m_frames;
  int m_stride;
  int m_downsampleShift; // 0..3; OFA processing dims = input >> shift
  static constexpr double kFlowScale = 1.0 / 32.0; // grid 1x1 -> Q10.5 (divide by 32)
  uint32_t m_width = 0, m_height = 0; // full-resolution frame dims
  uint32_t m_procW = 0, m_procH = 0; // OFA processing dims = full >> downsampleShift
  int m_numLevels = 0;

  NvMediaIofa* m_iofa = nullptr;
  CUstream m_stream = nullptr;

  std::vector<uint32_t> m_levelW, m_levelH;
  std::vector<NvSciCudaInteropBuffer*> m_inBuf, m_refBuf, m_outBuf, m_costBuf;
  std::vector<cv::Mat> m_inMat, m_refMat; // per-level scratch for uploads

  NvSciCudaInteropSync* m_preSync = nullptr;
  NvSciCudaInteropSync* m_eofSync = nullptr;

  cv::Mat m_flowRaw; // level-0 readback, CV_16SC2
  cv::Mat m_costRaw; // level-0 cost readback, CV_8UC1
  bool m_initialized = false;
};

} // namespace CameraImuCalib
