#include "common/tegra/DepthMapGeneratorOFA.h"
#include "imgui.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/tegra/NvSciUtil.h"
#include "common/glmCvInterop.h"
#include "common/remapArray.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/RHISurface.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda.h>
#include <nppi.h>

#include "nvmedia_core.h"
#include "nvmedia_iofa.h"
#include "nvscibuf.h"

#include <string>

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const uint32_t kOFAMaxDisparity = 128; // must be {128, 256}
extern CUdevice cudaDevice;

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

void setSinglePlaneImageAttrs(NvSciBufAttrList inAttrList, uint32_t width, uint32_t height, NvSciBufAttrValColorFmt fmt, /*optional*/ NvSciBufAttrValColorStd std = (NvSciBufAttrValColorStd) -1) {
  NvSciBufType bufType = NvSciBufType_Image;
  uint32_t planeCount = 1;
  uint32_t planeWidths[] = {width};
  uint32_t planeHeights[] = {height};

  NvSciBufAttrValColorFmt planecolorfmts[] = {fmt};
  NvSciBufAttrValColorStd planecolorstds[] = {std};

  NvSciBufAttrKeyValuePair imgBufAttrs[] = {
    {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
    {NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount)},
    {NvSciBufImageAttrKey_PlaneColorFormat, planecolorfmts, sizeof(planecolorfmts)},
    {NvSciBufImageAttrKey_PlaneWidth, planeWidths, sizeof(planeWidths)},
    {NvSciBufImageAttrKey_PlaneHeight, planeHeights, sizeof(planeHeights)},
  };

  NVSCI_CHECK(NvSciBufAttrListSetAttrs(inAttrList, imgBufAttrs, sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));

  if (std != ((NvSciBufAttrValColorStd) -1)) {
    NvSciBufAttrKeyValuePair colorStdAttr[] = {
      {NvSciBufImageAttrKey_PlaneColorStd, planecolorstds, sizeof(planecolorstds)},
    };
    NVSCI_CHECK(NvSciBufAttrListSetAttrs(inAttrList, colorStdAttr, sizeof(colorStdAttr) / sizeof(NvSciBufAttrKeyValuePair)));
  }
}

NvSciBufAttrList finishAndReconcileBufAttrList(NvSciBufAttrList inAttrList) {
  // Add common attributes
  NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
  NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
  NvSciBufAttrValImageScanType planescantype[] = {NvSciBufScan_ProgressiveType};

  CUuuid devUUID;
  CUDA_CHECK(cuDeviceGetUuid(&devUUID, cudaDevice));

  NvSciBufAttrKeyValuePair imgBufAttrs[] = {
    {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
    {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
    {NvSciBufImageAttrKey_ScanType, planescantype, sizeof(planescantype)},

    // CUDA device UUID attribute is required to map the NvSciBuf into CUDA
    {NvSciBufGeneralAttrKey_GpuId, &devUUID, sizeof(devUUID)},
  };


  NVSCI_CHECK(NvSciBufAttrListSetAttrs(inAttrList, imgBufAttrs, sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));

  // Add IOFA interop attributes
  NVMEDIA_CHECK(NvMediaIOFAFillNvSciBufAttrList(inAttrList));

  NvSciBufAttrList reconciledAttrList = nullptr;
  NvSciBufAttrList conflictList = nullptr;
  NVSCI_CHECK(NvSciBufAttrListReconcile(&inAttrList, 1, &reconciledAttrList, &conflictList));

  assert(reconciledAttrList != nullptr);
  bool isReconciled = false;
  NVSCI_CHECK(NvSciBufAttrListIsReconciled(reconciledAttrList, &isReconciled));
  assert(isReconciled);

  NvSciBufAttrListFree(inAttrList);
  return reconciledAttrList;
}


DepthMapGeneratorOFA::DepthMapGeneratorOFA() : DepthMapGenerator(kDepthBackendOFA) {
  m_algoDownsampleX = 4;
  m_algoDownsampleY = 4;
  m_maxDisparity = kOFAMaxDisparity;
  // NvMediaIOFA docs say:
  // Returned [disparity] values are in Q10.5 format, i.e., signed fixed point with 5 fractional bits.
  // Divide it by 32.0f to convert it to floating point.
  m_disparityPrescale = (1.0f / 32.0f);

  CUDA_CHECK(cuEventCreate(&m_masterFrameStartEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&m_ofaHandoffCompleteEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&m_masterFrameFinishedEvent, CU_EVENT_DEFAULT));

  // Set up a good initial state for the frame timing events
  CUDA_CHECK_NONFATAL(cuEventRecord(m_masterFrameStartEvent, (CUstream) m_globalStream.cudaPtr()));
  CUDA_CHECK_NONFATAL(cuEventRecord(m_ofaHandoffCompleteEvent, (CUstream) m_globalStream.cudaPtr()));
  CUDA_CHECK_NONFATAL(cuEventRecord(m_masterFrameFinishedEvent, (CUstream) m_globalStream.cudaPtr()));


  // Open OFA hardware
  // Version check
  {
    NvMediaVersion iofaVersion;
    memset(&iofaVersion, 0, sizeof(iofaVersion));
    NVMEDIA_CHECK(NvMediaIOFAGetVersion(&iofaVersion));
    printf("IOFA version: %u.%u.%u\n", iofaVersion.major, iofaVersion.minor, iofaVersion.patch);
    if ( (iofaVersion.major != NVMEDIA_IOFA_VERSION_MAJOR)
      || (iofaVersion.minor != NVMEDIA_IOFA_VERSION_MINOR)
      || (iofaVersion.patch != NVMEDIA_IOFA_VERSION_PATCH)) {

      printf("!!!WARNING: NvMedia IOFA Header version mismatch -- expected %u.%u.%u\n",
        NVMEDIA_IOFA_VERSION_MAJOR,
        NVMEDIA_IOFA_VERSION_MINOR,
        NVMEDIA_IOFA_VERSION_PATCH);
    }
  }

  m_iofa = NvMediaIOFACreate();
  if (!m_iofa) {
    die("NvMediaIOFACreate() returned NULL!\n");
  }

  {
    NvMediaIofaCapability caps;
    memset(&caps, 0, sizeof(caps));
    NVMEDIA_CHECK(NvMediaIOFAGetCapability(m_iofa, NVMEDIA_IOFA_MODE_STEREO, &caps));
    printf("NvMedia IOFA HW capabilities: Size range %ux%u - %ux%u\n",
      caps.minWidth, caps.minHeight, caps.maxWidth, caps.maxHeight);
  }
}

void DepthMapGeneratorOFA::internalPostInitWithCameraSystem() {
  // Finish initializing the hardware now that we know the dimensions

  // Hardware settings
  NvMediaIofaInitParams iofaParams;
  memset(&iofaParams, 0, sizeof(iofaParams));
  iofaParams.ofaMode = NVMEDIA_IOFA_MODE_STEREO;
  iofaParams.ofaPydLevel = 0;
  iofaParams.width[0] = internalWidth();
  iofaParams.height[0] = internalHeight();

  // Using a 1x1 grid, input and output sizes are identical.
  // For other cases:
  // outWidth  = (width  + (1 << gridSize) - 1)) >> gridSize
  // outHeight = (height + (1 << gridSize) - 1)) >> gridSize
  iofaParams.gridSize[0] = NVMEDIA_IOFA_GRIDSIZE_1X1;
  iofaParams.outWidth[0] = iofaParams.width[0];
  iofaParams.outHeight[0] = iofaParams.height[0];

  iofaParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_128;
  iofaParams.pydMode = NVMEDIA_IOFA_PYD_FRAME_MODE;
  iofaParams.preset = NVMEDIA_IOFA_PRESET_HQ; // high-quality mode

  NVMEDIA_CHECK(NvMediaIOFAInit(m_iofa, &iofaParams, /*maxInputBuffering=*/ 4));

  // Processing parameters
  memset(&m_iofaProcessParams, 0, sizeof(m_iofaProcessParams));
  // Structure pointer is required, but all processing params set to zero is fine


  // Build attribute lists for buffer types

  // Input
  {
    NvSciBufAttrList attrList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListCreate(gBufModule(), &attrList));
    setSinglePlaneImageAttrs(attrList, internalWidth(), internalHeight(), NvSciColor_Y8, NvSciColorStd_REC709_ER);
    m_inputBufferAttrList = finishAndReconcileBufAttrList(attrList);
  }

  // Disparity
  {
    NvSciBufAttrList attrList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListCreate(gBufModule(), &attrList));
    setSinglePlaneImageAttrs(attrList, internalWidth(), internalHeight(), NvSciColor_Signed_A16);
    m_disparityBufferAttrList = finishAndReconcileBufAttrList(attrList);
  }

  // Cost
  {
    NvSciBufAttrList attrList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListCreate(gBufModule(), &attrList));
    setSinglePlaneImageAttrs(attrList, internalWidth(), internalHeight(), NvSciColor_A8);
    m_costBufferAttrList = finishAndReconcileBufAttrList(attrList);
  }
}

DepthMapGeneratorOFA::~DepthMapGeneratorOFA() {
  cuEventDestroy(m_masterFrameStartEvent);
  cuEventDestroy(m_ofaHandoffCompleteEvent);
  cuEventDestroy(m_masterFrameFinishedEvent);
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)
void DepthMapGeneratorOFA::internalLoadSettings(cv::FileStorage& fs) {
  cv::FileNode ofa = fs["ofa"];
  if (ofa.isMap()) {
    // cv::read(ofa["confidenceThreshold"], m_params.confidenceThreshold, m_params.confidenceThreshold);
    // cv::read(ofa["quality"], m_params.quality, m_params.quality);
  }
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void DepthMapGeneratorOFA::internalSaveSettings(cv::FileStorage& fs) {
  fs.startWriteStruct(cv::String("ofa"), cv::FileNode::MAP, cv::String());
    // fs.write("confidenceThreshold", m_params.confidenceThreshold);
    // fs.write("quality", m_params.quality);
  fs.endWriteStruct();
}
#undef writeNode


void DepthMapGeneratorOFA::internalUpdateViewData() {
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
    auto vd = viewDataAtIndex(viewIdx);

    vd->releaseResources();

    if (!vd->m_isStereoView)
      continue;

    // Need IOFA pointer for future resource destruction
    vd->m_iofa = m_iofa;

    vd->updateDisparityTexture(internalWidth(), internalHeight(), kSurfaceFormat_R16i);

    // Build a half-res undistortRectifyMap to save some processing time
    unsigned int downsampleFactor = 2;

    PER_EYE {
      CameraSystem::Camera& cam = m_cameraSystem->cameraAtIndex(v.cameraIndices[eyeIdx]);
      vd->m_undistortRectifyMap_gpu[eyeIdx] = remapArray_initUndistortRectifyMap(cam.intrinsicMatrix, cam.distCoeffs, v.stereoRectification[eyeIdx], v.stereoProjection[eyeIdx], cv::Size(inputWidth(), inputHeight()), downsampleFactor);
    }

    // Output from remapArray
    PER_EYE vd->m_rectifiedMat[eyeIdx].create(cv::Size(internalWidth(), internalHeight()), CV_8U);

    // OFA input buffers
    PER_EYE {
      vd->m_ofaInputBuffer[eyeIdx] = new NvSciCudaInteropBuffer(m_inputBufferAttrList);
      NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(m_iofa, vd->m_ofaInputBuffer[eyeIdx]->m_nvSciBuf));
    }

    // OFA outputs
    vd->m_ofaOutputDisparityBuffer = new NvSciCudaInteropBuffer(m_disparityBufferAttrList);
    NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(m_iofa, vd->m_ofaOutputDisparityBuffer->m_nvSciBuf));

    vd->m_ofaOutputCostBuffer = new NvSciCudaInteropBuffer(m_costBufferAttrList);
    NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(m_iofa, vd->m_ofaOutputCostBuffer->m_nvSciBuf));

    // OFA syncs

    vd->m_ofaPreSync = new NvSciCudaInteropSync(NvSciCudaInteropSync::kSyncCudaSignalerToNvSciWaiter, m_iofa);
    NVMEDIA_CHECK(NvMediaIOFARegisterNvSciSyncObj(m_iofa, NVMEDIA_PRESYNCOBJ, vd->m_ofaPreSync->m_nvSciSync));
    
    vd->m_ofaEofSync = new NvSciCudaInteropSync(NvSciCudaInteropSync::kSyncNvSciSignalerToCudaWaiter, m_iofa);
    NVMEDIA_CHECK(NvMediaIOFARegisterNvSciSyncObj(m_iofa, NVMEDIA_EOFSYNCOBJ, vd->m_ofaEofSync->m_nvSciSync));

    // OFA surface array setup
    memset(&vd->m_ofaSurfArray, 0, sizeof(vd->m_ofaSurfArray));
    vd->m_ofaSurfArray.inputSurface[0] = vd->m_ofaInputBuffer[0]->m_nvSciBuf; // left surface
    vd->m_ofaSurfArray.refSurface[0] = vd->m_ofaInputBuffer[1]->m_nvSciBuf; // right surface
    vd->m_ofaSurfArray.outSurface[0] = vd->m_ofaOutputDisparityBuffer->m_nvSciBuf;
    vd->m_ofaSurfArray.costSurface[0] = vd->m_ofaOutputCostBuffer->m_nvSciBuf;
  }
}

void copyGpuMatToNvSciBuf(cv::cuda::GpuMat& inGpuMat, NvSciCudaInteropBuffer* buf, CUstream stream) {
  // Sanity checks
  assert(inGpuMat.cols && inGpuMat.rows);
  assert(buf->m_width && buf->m_height);
  assert(inGpuMat.cols == buf->m_width);
  assert(inGpuMat.rows == buf->m_height);

  size_t copyWidth = buf->m_width;
  size_t copyHeight = buf->m_height;

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.srcDevice = (CUdeviceptr) inGpuMat.cudaPtr();
  copyDescriptor.srcPitch = inGpuMat.step;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = buf->m_cuArray;

  //assert((bpp/8) == inGpuMat.elemSize()); // sanity check

  copyDescriptor.WidthInBytes = copyWidth * inGpuMat.elemSize();
  copyDescriptor.Height = copyHeight;
  CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
}

void copyNvSciBufToGpuMat(NvSciCudaInteropBuffer* buf, cv::cuda::GpuMat& outGpuMat, CUstream stream) {
  // Sanity checks
  assert(outGpuMat.cols && outGpuMat.rows);
  assert(buf->m_width && buf->m_height);
  assert(outGpuMat.cols == buf->m_width);
  assert(outGpuMat.rows == buf->m_height);

  size_t copyWidth = buf->m_width;
  size_t copyHeight = buf->m_height;


  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.srcArray = buf->m_cuArray;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.dstDevice = (CUdeviceptr) outGpuMat.cudaPtr();
  copyDescriptor.dstPitch = outGpuMat.step;

  //assert((bpp/8) == outGpuMat.elemSize()); // sanity check

  copyDescriptor.WidthInBytes = copyWidth * outGpuMat.elemSize();
  copyDescriptor.Height = copyHeight;
  CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
}

void copyNvSciBufToSurface(NvSciCudaInteropBuffer* buf, RHISurface::ptr surface, CUstream stream) {

  size_t copyWidth = std::min<size_t>(surface->width(), buf->m_width);
  size_t copyHeight = std::min<size_t>(surface->height(), buf->m_height);

  CUarray pSurfaceMip0Array;
  CUDA_CHECK(cuGraphicsResourceSetMapFlags(surface->cuGraphicsResource(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
  CUDA_CHECK(cuGraphicsMapResources(1, &surface->cuGraphicsResource(), stream));
  CUDA_CHECK(cuGraphicsSubResourceGetMappedArray(&pSurfaceMip0Array, surface->cuGraphicsResource(), /*arrayIndex=*/ 0, /*mipLevel=*/ 0));

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.srcArray = buf->m_cuArray;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = pSurfaceMip0Array;

  //assert((bpp/8) == outGpuMat.elemSize()); // sanity check

  size_t bytesPerPixel = rhiSurfaceFormatSize(surface->format());
  copyDescriptor.WidthInBytes = copyWidth * bytesPerPixel;
  copyDescriptor.Height = copyHeight;
  CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));

  CUDA_CHECK(cuGraphicsUnmapResources(1, &surface->cuGraphicsResource(), stream));
}


void DepthMapGeneratorOFA::internalProcessFrame() {

  // Copy the results from the previous frame to GL
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView || vd->anyCameraStreamFailed())
      continue;

    copyNvSciBufToGpuMat(vd->m_ofaOutputDisparityBuffer, vd->m_disparityGpuMat, (CUstream) m_globalStream.cudaPtr());

    if (m_populateDebugTextures) {
      if (!vd->m_leftGray)
        vd->m_leftGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      if (!vd->m_rightGray)
        vd->m_rightGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      copyNvSciBufToSurface(vd->m_ofaInputBuffer[0], vd->m_leftGray, (CUstream) m_globalStream.cudaPtr());
      copyNvSciBufToSurface(vd->m_ofaInputBuffer[1], vd->m_rightGray, (CUstream) m_globalStream.cudaPtr());
    }

    if (debugDisparityCPUAccessEnabled()) {
      PER_EYE vd->m_rectifiedMat[eyeIdx].download(vd->m_debugCPUDisparityInput[eyeIdx], m_globalStream);
    }
  }

  internalFinalizeDisparityTexture();

  if (m_enableProfiling) {
    // Collect profiling data from previous frame
    cuEventElapsedTime(&m_preOfaFrameTimeMs, m_masterFrameStartEvent, m_ofaHandoffCompleteEvent);
    cuEventElapsedTime(&m_ofaFrameTimeMs, m_ofaHandoffCompleteEvent, m_masterFrameFinishedEvent);
  }

  // Begin new frame
  CUDA_CHECK(cuEventRecord(m_masterFrameStartEvent, (CUstream) m_globalStream.cudaPtr()));

  // First pass: do preprocessing, hand off to NvSci, and submit OFA tasks per view
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView || vd->anyCameraStreamFailed())
      continue;

    // Remap for distortion correction
    cv::Size inputSize = cv::Size(inputWidth(), inputHeight());
    PER_EYE remapArray(m_cameraSystem->cameraProvider()->cudaLumaTexObject(m_cameraSystem->viewAtIndex(viewIdx).cameraIndices[eyeIdx]), inputSize, vd->m_undistortRectifyMap_gpu[eyeIdx], vd->m_rectifiedMat[eyeIdx], (CUstream) m_globalStream.cudaPtr(), /*downsampleFactor=*/ 2);

    // Populate NvSci input buffer
    // TODO: This really should be merged in with the remap above -- write straight to the CUarray, skip a copy.
    PER_EYE copyGpuMatToNvSciBuf(vd->m_rectifiedMat[eyeIdx], vd->m_ofaInputBuffer[eyeIdx], (CUstream) m_globalStream.cudaPtr());

    // Signal preprocess semaphore for OFA handoff
    vd->m_ofaPreSync->signalCudaToNvSci((CUstream) m_globalStream.cudaPtr());

    // Tell OFA to wait on the pre fence for this frame
    NVMEDIA_CHECK(NvMediaIOFAInsertPreNvSciSyncFence(m_iofa, &vd->m_ofaPreSync->m_nvSciSyncFence));
    // EOF sync object needs to be provided before frame submission
    NVMEDIA_CHECK(NvMediaIOFASetNvSciSyncObjforEOF(m_iofa, vd->m_ofaEofSync->m_nvSciSync));

    // OFA processing
    NVMEDIA_CHECK(NvMediaIOFAProcessFrame(m_iofa, &vd->m_ofaSurfArray, &m_iofaProcessParams, /*pEpiInfo=*/ nullptr, /*pROIParams=*/ nullptr));

    // Get EOF fence so CUDA can wait on it later
    NVMEDIA_CHECK(NvMediaIOFAGetEOFNvSciSyncFence(m_iofa, vd->m_ofaEofSync->m_nvSciSync, &vd->m_ofaEofSync->m_nvSciSyncFence));
  } // view loop

  CUDA_CHECK(cuEventRecord(m_ofaHandoffCompleteEvent, (CUstream) m_globalStream.cudaPtr()));

  // Second pass: wait on OFA processing to finish
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView || vd->anyCameraStreamFailed())
      continue;

    vd->m_ofaEofSync->waitNvSciToCuda((CUstream) m_globalStream.cudaPtr());
  }

  // Frame finished event occurs after the CUDA stream has waited on all OFA processing
  CUDA_CHECK(cuEventRecord(m_masterFrameFinishedEvent, (CUstream) m_globalStream.cudaPtr()));
}

void DepthMapGeneratorOFA::internalRenderIMGUI() {
  // TODO: expose SGM parameters

  // ImGui::SliderInt("Confidence Threshold", &m_params.confidenceThreshold, 0, 65535);
  // ImGui::SliderInt("Quality", &m_params.quality, 1, 255);
}

void DepthMapGeneratorOFA::internalRenderIMGUIPerformanceGraphs() {
  if (!m_enableProfiling)
    return;

  // TODO: graphs
  ImGui::Text("Setup %.1fms OFA %.3fms", m_preOfaFrameTimeMs, m_ofaFrameTimeMs);
}

