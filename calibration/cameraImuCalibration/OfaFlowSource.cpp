#include "OfaFlowSource.h"
#include "OfaFlowUtil.h"
#include "common/tegra/NvSciCudaInterop.h"
#include "common/tegra/NvSciUtil.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICUDA.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdio>

namespace CameraImuCalib {

OfaFlowSource::OfaFlowSource(const FrameSequence& frames, int stride, int downsampleShift) :
  m_frames(frames),
  m_stride(std::max(1, stride)),
  m_downsampleShift(std::max(0, std::min(3, downsampleShift))) {}

OfaFlowSource::~OfaFlowSource() {
  auto freeBuf = [&](NvSciCudaInteropBuffer*& b) {
    if (b) {
      NvMediaIOFAUnregisterNvSciBufObj(m_iofa, b->m_nvSciBuf);
      delete b;
      b = nullptr;
    }
  };
  if (m_iofa) {
    if (m_preSync) NvMediaIOFAUnregisterNvSciSyncObj(m_iofa, m_preSync->m_nvSciSync);
    if (m_eofSync) NvMediaIOFAUnregisterNvSciSyncObj(m_iofa, m_eofSync->m_nvSciSync);
  }
  for (auto& b : m_inBuf) freeBuf(b);
  for (auto& b : m_refBuf) freeBuf(b);
  for (auto& b : m_outBuf) freeBuf(b);
  for (auto& b : m_costBuf) freeBuf(b);
  delete m_preSync;
  delete m_eofSync;
  if (m_iofa) NvMediaIOFADestroy(m_iofa);
}

NvSciCudaInteropBuffer* OfaFlowSource::makeRegisteredSurface(uint32_t width, uint32_t height,
  int colorFmt, bool setColorStd) {
  NvSciBufAttrList attrList = nullptr;
  NVSCI_CHECK(NvSciBufAttrListCreate(gBufModule(), &attrList));
  NVMEDIA_CHECK(NvMediaIOFAFillNvSciBufAttrList(attrList));
  populateImageBufAttrList(attrList, width, height,
    static_cast<NvSciBufAttrValColorFmt>(colorFmt), setColorStd);
  NvSciBufAttrList reconciled = ReconcileNvSciBufAttrLists(attrList);
  NvSciCudaInteropBuffer* buf = new NvSciCudaInteropBuffer(reconciled);
  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(m_iofa, buf->m_nvSciBuf));
  return buf;
}

bool OfaFlowSource::initialize() {
  if (m_frames.frameCount() < 2) {
    fprintf(stderr, "OfaFlowSource: need >= 2 frames\n");
    return false;
  }
  m_width = m_frames.width();
  m_height = m_frames.height();

  RHICUDA::initRHICUDA();
  CUDA_CHECK(cuStreamCreate(&m_stream, CU_STREAM_NON_BLOCKING));

  NvMediaVersion version;
  memset(&version, 0, sizeof(version));
  NVMEDIA_CHECK(NvMediaIOFAGetVersion(&version));
  printf("OfaFlowSource: IOFA version %u.%u.%u\n", version.major, version.minor, version.patch);

  m_iofa = NvMediaIOFACreate();
  if (!m_iofa) {
    fprintf(stderr, "OfaFlowSource: NvMediaIOFACreate failed\n");
    return false;
  }

  NvMediaIofaCapability caps;
  memset(&caps, 0, sizeof(caps));
  NVMEDIA_CHECK(NvMediaIOFAGetCapability(m_iofa, NVMEDIA_IOFA_MODE_PYDOF, &caps));
  if (m_width > caps.maxWidth || m_height > caps.maxHeight ||
    m_width < caps.minWidth || m_height < caps.minHeight) {
    fprintf(stderr, "OfaFlowSource: %ux%u outside HW range %ux%u..%ux%u\n",
      m_width, m_height, caps.minWidth, caps.minHeight, caps.maxWidth, caps.maxHeight);
    return false;
  }

  // OFA processes at the (optionally downsampled) resolution; flow is mapped back to full
  // resolution in pair(). Level 0 is the processing resolution, each coarser level halves.
  m_procW = std::max<uint32_t>(1, m_width >> m_downsampleShift);
  m_procH = std::max<uint32_t>(1, m_height >> m_downsampleShift);
  const uint32_t minCoarse = std::max<uint32_t>(caps.minWidth, caps.minHeight);
  int maxLevels = 1;
  while (maxLevels < static_cast<int>(NVMEDIA_IOFA_MAX_PYD_LEVEL)) {
    const uint32_t w = m_procW >> maxLevels, h = m_procH >> maxLevels;
    if (std::min(w, h) < minCoarse) break;
    ++maxLevels;
  }
  m_numLevels = std::max(1, std::min(maxLevels, static_cast<int>(NVMEDIA_IOFA_MAX_PYD_LEVEL)));
  m_levelW.resize(m_numLevels);
  m_levelH.resize(m_numLevels);
  for (int l = 0; l < m_numLevels; ++l) {
    m_levelW[l] = std::max<uint32_t>(1, m_procW >> l);
    m_levelH[l] = std::max<uint32_t>(1, m_procH >> l);
  }
  printf("OfaFlowSource: full %ux%u, OFA processing %ux%u (downsample 1/%d), %d pyramid levels\n",
    m_width, m_height, m_procW, m_procH, 1 << m_downsampleShift, m_numLevels);

  NvMediaIofaInitParams params;
  memset(&params, 0, sizeof(params));
  params.ofaMode = NVMEDIA_IOFA_MODE_PYDOF;
  params.ofaPydLevel = static_cast<uint8_t>(m_numLevels);
  // Grid 1x1: PYDOF does not support output gridding. Output size equals input per level.
  for (int l = 0; l < m_numLevels; ++l) {
    params.width[l] = static_cast<uint16_t>(m_levelW[l]);
    params.height[l] = static_cast<uint16_t>(m_levelH[l]);
    params.gridSize[l] = NVMEDIA_IOFA_GRIDSIZE_1X1;
    params.outWidth[l] = static_cast<uint16_t>(m_levelW[l]);
    params.outHeight[l] = static_cast<uint16_t>(m_levelH[l]);
  }
  params.pydMode = NVMEDIA_IOFA_PYD_FRAME_MODE;
  params.preset = NVMEDIA_IOFA_PRESET_HQ;
  NVMEDIA_CHECK(NvMediaIOFAInit(m_iofa, &params, /*maxInputBuffering=*/ 4));

  m_preSync = new NvSciCudaInteropSync(NvSciCudaInteropSync::kSyncCudaSignalerToNvSciWaiter,
    m_iofa, /*allowCpuWaiter=*/ false);
  m_eofSync = new NvSciCudaInteropSync(NvSciCudaInteropSync::kSyncNvSciSignalerToCudaWaiter,
    m_iofa, /*allowCpuWaiter=*/ false);
  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciSyncObj(m_iofa, NVMEDIA_PRESYNCOBJ, m_preSync->m_nvSciSync));
  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciSyncObj(m_iofa, NVMEDIA_EOFSYNCOBJ, m_eofSync->m_nvSciSync));

  m_inBuf.resize(m_numLevels);
  m_refBuf.resize(m_numLevels);
  m_outBuf.resize(m_numLevels);
  m_costBuf.resize(m_numLevels);
  m_inMat.resize(m_numLevels);
  m_refMat.resize(m_numLevels);
  for (int l = 0; l < m_numLevels; ++l) {
    m_inBuf[l] = makeRegisteredSurface(m_levelW[l], m_levelH[l], NvSciColor_Y8, true);
    m_refBuf[l] = makeRegisteredSurface(m_levelW[l], m_levelH[l], NvSciColor_Y8, true);
    m_outBuf[l] = makeRegisteredSurface(m_levelW[l], m_levelH[l], NvSciColor_Signed_R16G16, false);
    m_costBuf[l] = makeRegisteredSurface(m_levelW[l], m_levelH[l], NvSciColor_A8, false);
  }

  m_flowRaw.create(cv::Size(m_procW, m_procH), CV_16SC2);
  m_costRaw.create(cv::Size(m_procW, m_procH), CV_8UC1);
  m_initialized = true;
  return true;
}

bool OfaFlowSource::runOfa(const cv::Mat& image0, const cv::Mat& image1) {
  // Build the per-level input/reference pyramids from the full-resolution frames. Level 0 is
  // the OFA processing resolution (full >> downsampleShift); each coarser level halves.
  for (int l = 0; l < m_numLevels; ++l) {
    const cv::Size levelSize(m_levelW[l], m_levelH[l]);
    if (image0.size() == levelSize) {
      m_inMat[l] = image0;
      m_refMat[l] = image1;
    } else {
      cv::resize(image0, m_inMat[l], levelSize, 0, 0, cv::INTER_AREA);
      cv::resize(image1, m_refMat[l], levelSize, 0, 0, cv::INTER_AREA);
    }
    copyCvMatToSurface(m_inMat[l], m_inBuf[l]->m_cuArray, m_stream);
    copyCvMatToSurface(m_refMat[l], m_refBuf[l]->m_cuArray, m_stream);
  }

  // CUDA -> OFA handoff fence.
  m_preSync->signalCudaToNvSci(m_stream);
  NVMEDIA_CHECK(NvMediaIOFAInsertPreNvSciSyncFence(m_iofa, &m_preSync->m_nvSciSyncFence));
  NvSciSyncFenceClear(&m_preSync->m_nvSciSyncFence);
  NVMEDIA_CHECK(NvMediaIOFASetNvSciSyncObjforEOF(m_iofa, m_eofSync->m_nvSciSync));

  NvMediaIofaBufArray surfArray;
  memset(&surfArray, 0, sizeof(surfArray));
  for (int l = 0; l < m_numLevels; ++l) {
    surfArray.inputSurface[l] = m_inBuf[l]->m_nvSciBuf;
    surfArray.refSurface[l] = m_refBuf[l]->m_nvSciBuf;
    surfArray.outSurface[l] = m_outBuf[l]->m_nvSciBuf;
    surfArray.costSurface[l] = m_costBuf[l]->m_nvSciBuf;
  }
  NvMediaIofaProcessParams processParams;
  memset(&processParams, 0, sizeof(processParams));
  NVMEDIA_CHECK(NvMediaIOFAProcessFrame(m_iofa, &surfArray, &processParams,
    /*pEpiInfo=*/ nullptr, /*pROIParams=*/ nullptr));

  // OFA -> CUDA fence, then read back the level-0 flow and cost surfaces.
  NVMEDIA_CHECK(NvMediaIOFAGetEOFNvSciSyncFence(m_iofa, m_eofSync->m_nvSciSync,
    &m_eofSync->m_nvSciSyncFence));
  m_eofSync->waitNvSciToCuda(m_stream);
  copySurfaceToCvMat(m_outBuf[0]->m_cuArray, m_flowRaw, m_stream);
  copySurfaceToCvMat(m_costBuf[0]->m_cuArray, m_costRaw, m_stream);
  CUDA_CHECK(cuStreamSynchronize(m_stream));
  return true;
}

bool OfaFlowSource::pair(size_t pairIndex, FramePairFlow& outPair) {
  if (!m_initialized || pairIndex + 1 >= m_frames.frameCount())
    return false;

  cv::Mat image0, image1;
  if (!m_frames.loadGreyscale(pairIndex, image0) ||
    !m_frames.loadGreyscale(pairIndex + 1, image1)) {
    fprintf(stderr, "OfaFlowSource: failed to load frame pair %zu\n", pairIndex);
    return false;
  }
  if (!runOfa(image0, image1))
    return false;

  outPair.frameStartTimeA = m_frames.timestampSeconds(pairIndex);
  outPair.frameStartTimeB = m_frames.timestampSeconds(pairIndex + 1);
  outPair.samples.clear();
  outPair.samples.reserve((m_procW / m_stride + 1) * (m_procH / m_stride + 1));

  // Flow is computed at the processing resolution. Map each sample (and its displacement)
  // back to full-resolution pixels by the downsample factor so the fixed intrinsics apply.
  // `stride` further subsamples the flow field.
  const double factor = static_cast<double>(1 << m_downsampleShift);
  const double half = 0.5 * factor; // sample the center of the block this proc-pixel covers
  for (uint32_t py = static_cast<uint32_t>(m_stride) / 2; py < m_procH; py += m_stride) {
    const cv::Vec2s* flowRow = m_flowRaw.ptr<cv::Vec2s>(py);
    const double v = py * factor + half;
    for (uint32_t px = static_cast<uint32_t>(m_stride) / 2; px < m_procW; px += m_stride) {
      const cv::Vec2s raw = flowRow[px];
      const double fx = raw[0] * kFlowScale * factor; // proc-pixel displacement -> full-res
      const double fy = raw[1] * kFlowScale * factor;
      const double u = px * factor + half;
      FlowSample s;
      s.pixel0 = Eigen::Vector2d(u, v);
      s.pixel1 = Eigen::Vector2d(u + fx, v + fy);
      // OFA cost surface read back but its polarity is not yet wired into a confidence
      // prior; Stage 1's robust IRLS handles outliers. Left at 1.0 for v1.
      s.confidence = 1.0;
      outPair.samples.push_back(s);
    }
  }
  return true;
}

} // namespace CameraImuCalib
