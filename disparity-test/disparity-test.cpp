#define NV_IS_SAFETY 0

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <unistd.h>
#include <algorithm>
#include <vector>

#include <boost/core/noncopyable.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

#include "nvscibuf.h"
#include "nvmedia_core.h"
#include "nvmedia_iofa.h"

#include <cuda.h>
#include "common/disparityOcclusionMask.h"
#include "common/fgsFilter.h"
#include "common/tegra/ofaCostToConfidence.h"
#include "common/tegra/NvSciUtil.h"
#include "common/tegra/NvSciCudaInterop.h"
#include "common/Timing.h"
#include "readPFM.h"
#include "rhi/cuda/CudaUtil.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

#define die(msg, ...)                         \
  do {                                        \
    fprintf(stderr, msg "\n", ##__VA_ARGS__); \
    abort();                                  \
  } while (0)
#define CHECK_PTR(x)                                                               \
  if (!(x)) {                                                                      \
    fprintf(stderr, "%s:%d: %s failed (returned NULL)\n", __FILE__, __LINE__, #x); \
    abort();                                                                       \
  }

// ------------------------ globals --------------------------

// CUDA
CUdevice cudaDevice;
CUcontext cudaContext;

// NVSCI
NvSciSyncModule syncModule;
NvSciBufModule bufModule;

// NVMEDIA IOFA
NvMediaIofa* iofa = nullptr;

void populateInputImageBufAttrList(NvSciBufAttrList& attrList, uint32_t width, uint32_t height) {
  NvSciBufType bufType = NvSciBufType_Image;
  NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
  NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

  uint32_t planeCount = 1;
  uint32_t planeWidths[] = {width};
  uint32_t planeHeights[] = {height};

  bool cpuAccessFlag = false;

  NvSciBufAttrValColorFmt planecolorfmts[] = {NvSciColor_Y8};
  NvSciBufAttrValColorStd planecolorstds[] = {NvSciColorStd_REC709_ER};
  NvSciBufAttrValImageScanType planescantype[] = {NvSciBufScan_ProgressiveType};

  CUuuid devUUID;
  CUDA_CHECK(cuDeviceGetUuid(&devUUID, cudaDevice));

  NvSciBufAttrKeyValuePair imgBufAttrs[] = {
    {         NvSciBufGeneralAttrKey_Types,       &bufType,        sizeof(bufType)},
    {      NvSciBufImageAttrKey_PlaneCount,    &planeCount,     sizeof(planeCount)},
    {          NvSciBufImageAttrKey_Layout,        &layout,         sizeof(layout)},

    {NvSciBufImageAttrKey_PlaneColorFormat, planecolorfmts, sizeof(planecolorfmts)},
    {   NvSciBufImageAttrKey_PlaneColorStd, planecolorstds, sizeof(planecolorstds)},
    {      NvSciBufImageAttrKey_PlaneWidth,    planeWidths,    sizeof(planeWidths)},
    {     NvSciBufImageAttrKey_PlaneHeight,   planeHeights,   sizeof(planeHeights)},
    { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag,  sizeof(cpuAccessFlag)},
    {  NvSciBufGeneralAttrKey_RequiredPerm,          &perm,           sizeof(perm)},
    {        NvSciBufImageAttrKey_ScanType,  planescantype,  sizeof(planescantype)},

 // Required for CUDA interop
    {         NvSciBufGeneralAttrKey_GpuId,       &devUUID,        sizeof(devUUID)},
  };

  NVSCI_CHECK(NvSciBufAttrListSetAttrs(attrList, imgBufAttrs, sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
}


void populateOutputImageBufAttrList(NvSciBufAttrList& attrList, uint32_t width, uint32_t height, bool forCostMap) {
  NvSciBufType bufType = NvSciBufType_Image;
  NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
  NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

  uint32_t planeCount = 1;
  uint32_t planeWidths[] = {width};
  uint32_t planeHeights[] = {height};

  bool cpuAccessFlag = false;

  NvSciBufAttrValColorFmt planecolorfmts[] = {forCostMap ? NvSciColor_A8 : NvSciColor_Signed_A16};
  NvSciBufAttrValImageScanType planescantype[] = {NvSciBufScan_ProgressiveType};

  CUuuid devUUID;
  CUDA_CHECK(cuDeviceGetUuid(&devUUID, cudaDevice));

  NvSciBufAttrKeyValuePair imgBufAttrs[] = {
    {         NvSciBufGeneralAttrKey_Types,       &bufType,        sizeof(bufType)},
    {      NvSciBufImageAttrKey_PlaneCount,    &planeCount,     sizeof(planeCount)},
    {          NvSciBufImageAttrKey_Layout,        &layout,         sizeof(layout)},

    {NvSciBufImageAttrKey_PlaneColorFormat, planecolorfmts, sizeof(planecolorfmts)},
    {      NvSciBufImageAttrKey_PlaneWidth,    planeWidths,    sizeof(planeWidths)},
    {     NvSciBufImageAttrKey_PlaneHeight,   planeHeights,   sizeof(planeHeights)},
    { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag,  sizeof(cpuAccessFlag)},
    {  NvSciBufGeneralAttrKey_RequiredPerm,          &perm,           sizeof(perm)},
    {        NvSciBufImageAttrKey_ScanType,  planescantype,  sizeof(planescantype)},

 // Required for CUDA interop
    {         NvSciBufGeneralAttrKey_GpuId,       &devUUID,        sizeof(devUUID)},
  };

  NVSCI_CHECK(NvSciBufAttrListSetAttrs(attrList, imgBufAttrs, sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));
}

void copyCvMatToSurface(const cv::Mat& mat, CUarray arr, CUstream stream) {
  // TODO: This doesn't handle any layout differences between the mat and the CUarray.
  size_t copyWidth = mat.cols;
  size_t copyHeight = mat.rows;

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.srcHost = mat.ptr();
  copyDescriptor.srcPitch = mat.step;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = arr;

  copyDescriptor.WidthInBytes = copyWidth * mat.elemSize();
  copyDescriptor.Height = copyHeight;
  if (stream) {
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  } else {
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
  }
}

void copySurfaceToCvMat(CUarray arr, cv::Mat& mat, CUstream stream) {
  // TODO: This doesn't handle any layout differences between the mat and the CUarray.
  size_t copyWidth = mat.cols;
  size_t copyHeight = mat.rows;

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.srcArray = arr;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.dstHost = mat.ptr();
  copyDescriptor.dstPitch = mat.step;

  copyDescriptor.WidthInBytes = copyWidth * mat.elemSize();
  copyDescriptor.Height = copyHeight;
  if (stream) {
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  } else {
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
  }
}

void copySurfaceToGpuMat(CUarray arr, cv::cuda::GpuMat& mat, CUstream stream) {
  // TODO: This doesn't handle any layout differences between the mat and the CUarray.
  size_t copyWidth = mat.cols;
  size_t copyHeight = mat.rows;

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.srcArray = arr;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.dstDevice = (CUdeviceptr) mat.ptr();
  copyDescriptor.dstPitch = mat.step;

  copyDescriptor.WidthInBytes = copyWidth * mat.elemSize();
  copyDescriptor.Height = copyHeight;
  if (stream) {
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  } else {
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
  }
}

cv::Mat colormap(cv::Mat inMat, float scale, int colormap = cv::COLORMAP_JET) {
  cv::Mat tmp;
  inMat.convertTo(tmp, CV_8UC1, scale, 0);
  cv::applyColorMap(tmp, tmp, colormap);
  return tmp;
}

int main(int argc, char* argv[]) {

  if (argc < 3) {
    printf("usage: %s leftImage rightImage {quality|performance} {1|2|4|8}\n", argv[0]);
    return 0;
  }

  std::string leftImageFilename = argv[1];
  std::string rightImageFilename = argv[2];
  std::string refDisparityFilename;

  NvMediaIofaPreset iofaPreset = NVMEDIA_IOFA_PRESET_HQ;
  unsigned int gridSizeShift = 0; // 1x1
  uint8_t lowCostThreshold = 4;
  uint8_t highCostThreshold = 48;
  float costCurve = 2.0f;
  bool useOcclusionMask = false;
  uint32_t occlusionMaskSearchWindow = 0; // 0 = auto (max-disparity-in-pixels)
  float occlusionMaskHysteresis = 2.0f;
  uint8_t occlusionMaskConfidenceCeiling = 192;
  bool occlusionMaskSmear = true;

  bool useFGSFilter = false;
  bool useCudaFGSFilter = false;
  float fgsLambda = 8000.0f;
  float fgsSigma = 1.0f;


  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--ref-disparity")) {
      if (i == (argc - 1)) {
        printf("--ref-disparity: requires argument\n");
        return 1;
      }
      refDisparityFilename = argv[++i];
    } else if (!strcmp(argv[i], "--grid-shift")) {
      if (i == (argc - 1)) {
        printf("--grid-shift: requires argument\n");
        return 1;
      }
      gridSizeShift = atoi(argv[++i]);
      if (gridSizeShift > 3) {
        printf("Grid shift %u out of range; valid range is 0-3\n", gridSizeShift);
        return 1;
      }
    } else if (!strcmp(argv[i], "--quality")) {
      iofaPreset = NVMEDIA_IOFA_PRESET_HQ;
    } else if (!strcmp(argv[i], "--performance")) {
      iofaPreset = NVMEDIA_IOFA_PRESET_HP;
    } else if (!strcmp(argv[i], "--low-cost-threshold")) {
      if (i == (argc - 1)) {
        printf("--low-cost-threshold: requires argument\n");
        return 1;
      }
      lowCostThreshold = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--high-cost-threshold")) {
      if (i == (argc - 1)) {
        printf("--high-cost-threshold: requires argument\n");
        return 1;
      }
      highCostThreshold = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--cost-curve")) {
      if (i == (argc - 1)) {
        printf("--cost-curve: requires argument\n");
        return 1;
      }
      costCurve = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--use-occlusion-mask")) {
      useOcclusionMask = true;
    } else if (!strcmp(argv[i], "--occlusion-window")) {
      if (i == (argc - 1)) {
        printf("--occlusion-window: requires argument\n");
        return 1;
      }
      occlusionMaskSearchWindow = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--occlusion-hysteresis")) {
      if (i == (argc - 1)) {
        printf("--occlusion-hysteresis: requires argument\n");
        return 1;
      }
      occlusionMaskHysteresis = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--occlusion-confidence-ceiling")) {
      if (i == (argc - 1)) {
        printf("--occlusion-confidence-ceiling: requires argument\n");
        return 1;
      }
      int v = atoi(argv[++i]);
      if (v < 0 || v > 255) {
        printf("--occlusion-confidence-ceiling: argument out of range (0-255)\n");
        return 1;
      }
      occlusionMaskConfidenceCeiling = static_cast<uint8_t>(v);
    } else if (!strcmp(argv[i], "--no-occlusion-smear")) {
      occlusionMaskSmear = false;
    } else if (!strcmp(argv[i], "--use-fgs-filter")) {
      useFGSFilter = true;
    } else if (!strcmp(argv[i], "--cuda-fgs-filter")) {
      useFGSFilter = true;
      useCudaFGSFilter = true;
    } else if (!strcmp(argv[i], "--fgs-lambda")) {
      if (i == (argc - 1)) {
        printf("--fgs-lambda: requires argument\n");
        return 1;
      }
      fgsLambda = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--fgs-sigma")) {
      if (i == (argc - 1)) {
        printf("--fgs-sigma: requires argument\n");
        return 1;
      }
      fgsSigma = atof(argv[++i]);
    } else {
      printf("Unrecognized argument %s\n", argv[i]);
      return 1;
    }
  }

  // Need an RGB view of the left image for guiding the FGS filter.
  cv::Mat cvImageLeftRGB = cv::imread(leftImageFilename.c_str());
  if (cvImageLeftRGB.empty())
    die("Can't open left image %s\n", leftImageFilename.c_str());
  cv::Mat cvImageLeft;
  cv::cvtColor(/*src=*/ cvImageLeftRGB, /*dst=*/ cvImageLeft, cv::COLOR_BGR2GRAY);

  cv::Mat cvImageRight = cv::imread(rightImageFilename.c_str(), cv::IMREAD_GRAYSCALE);
  if (cvImageRight.empty())
    die("Can't open right image %s\n", rightImageFilename.c_str());

  cv::Mat cvRefDisparity;
  if (!refDisparityFilename.empty()) {
    if (!readPFM(refDisparityFilename.c_str(), cvRefDisparity))
      die("Can't open ref disparity image %s\n", refDisparityFilename.c_str());
  }

  printf("lowCostThreshold=%u highCostThreshold=%u\n",
    lowCostThreshold, highCostThreshold);

  assert(cvImageLeft.cols == cvImageRight.cols && cvImageLeft.rows == cvImageRight.rows);
  uint32_t imageWidth = cvImageLeft.cols;
  uint32_t imageHeight = cvImageLeft.rows;
  // Output size computation as a function of input size and grid size:
  // outWidth  = (width  + (1 << gridSizeShift) - 1)) >> gridSizeShift
  // outHeight = (height + (1 << gridSizeShift) - 1)) >> gridSizeShift
  // For 1x1 grid (gridSizeShift == 0), input and output sizes are identical.
  uint32_t outputWidth = (imageWidth + (1 << gridSizeShift) - 1) >> gridSizeShift;
  uint32_t outputHeight = (imageHeight + (1 << gridSizeShift) - 1) >> gridSizeShift;

  printf("Input: %ux%u || Output: %ux%u\n", imageWidth, imageHeight, outputWidth, outputHeight);

  cv::Mat cvDisparity, cvDisparityPreFilter;
  cvDisparity.create(cv::Size(outputWidth, outputHeight), CV_16U);
  cvDisparityPreFilter.create(cv::Size(outputWidth, outputHeight), CV_16U);
  cv::Mat cvConfidence;
  cvConfidence.create(cv::Size(outputWidth, outputHeight), CV_8U);

  // CUDA init
  {
    cuInit(0);

    cuDeviceGet(&cudaDevice, 0);
    cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice);
    cuCtxSetCurrent(cudaContext);
  }

  // CUDA stream
  CUstream hStream;
  CUDA_CHECK(cuStreamCreate(&hStream, CU_STREAM_NON_BLOCKING));

  // CUDA events
  CUevent startEvent, copyInFinishedEvent, ofaFinishedEvent, filterStartEvent, filterFinishedEvent;
  CUDA_CHECK(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&copyInFinishedEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&ofaFinishedEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&filterStartEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&filterFinishedEvent, CU_EVENT_DEFAULT));


  NvMediaVersion iofaVersion;
  memset(&iofaVersion, 0, sizeof(iofaVersion));
  NVMEDIA_CHECK(NvMediaIOFAGetVersion(&iofaVersion));
  // printf("IOFA version: %u.%u.%u\n", iofaVersion.major, iofaVersion.minor, iofaVersion.patch);
  if ((iofaVersion.major != NVMEDIA_IOFA_VERSION_MAJOR) || (iofaVersion.minor != NVMEDIA_IOFA_VERSION_MINOR) || (iofaVersion.patch != NVMEDIA_IOFA_VERSION_PATCH)) {

    printf("WARNING: Header version mismatch -- expected %u.%u.%u\n",
      NVMEDIA_IOFA_VERSION_MAJOR,
      NVMEDIA_IOFA_VERSION_MINOR,
      NVMEDIA_IOFA_VERSION_PATCH);
  }

  NVSCI_CHECK(NvSciSyncModuleOpen(&syncModule));
  NVSCI_CHECK(NvSciBufModuleOpen(&bufModule));

  CHECK_PTR(iofa = NvMediaIOFACreate());

  NvMediaIofaCapability caps;
  memset(&caps, 0, sizeof(caps));
  NVMEDIA_CHECK(NvMediaIOFAGetCapability(iofa, NVMEDIA_IOFA_MODE_STEREO, &caps));
#if 0
  printf("HW capabilities: Size range %ux%u - %ux%u\n",
    caps.minWidth, caps.minHeight, caps.maxWidth, caps.maxHeight);
#endif


  NvMediaIofaInitParams iofaParams;
  memset(&iofaParams, 0, sizeof(iofaParams));
  iofaParams.ofaMode = NVMEDIA_IOFA_MODE_STEREO;
  iofaParams.ofaPydLevel = 0;
  iofaParams.width[0] = imageWidth;
  iofaParams.height[0] = imageHeight;

  iofaParams.gridSize[0] = (NvMediaIofaGridSize) gridSizeShift;
  iofaParams.outWidth[0] = outputWidth;
  iofaParams.outHeight[0] = outputHeight;

  iofaParams.dispRange = NVMEDIA_IOFA_DISPARITY_RANGE_128;
  iofaParams.pydMode = NVMEDIA_IOFA_PYD_FRAME_MODE;
  iofaParams.preset = iofaPreset; // default is HQ

#if 0
  // Struct layout sanity checking
#define off(x) printf("offsetof(%s) = 0x%zx\n", #x, offsetof(NvMediaIofaInitParams, x));
  off(ofaMode);
  off(ofaPydLevel);
  off(width);
  off(height);
  off(gridSize);
  off(outWidth);
  off(outHeight);
  off(dispRange);
  off(pydMode);
  off(vprMode);
  off(preset);
  off(epiSearchRange);
#undef off
#endif

  NVMEDIA_CHECK(NvMediaIOFAInit(iofa, &iofaParams, /*maxInputBuffering=*/ 4));

  // Create and register sync objects between NvSci and CUDA
  NvSciCudaInteropSync preSync(NvSciCudaInteropSync::kSyncCudaSignalerToNvSciWaiter, iofa);
  NvSciCudaInteropSync eofSync(NvSciCudaInteropSync::kSyncNvSciSignalerToCudaWaiter, iofa);

  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciSyncObj(iofa, NVMEDIA_PRESYNCOBJ, preSync.m_nvSciSync));
  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciSyncObj(iofa, NVMEDIA_EOFSYNCOBJ, eofSync.m_nvSciSync));


  // Input buffer attribute list

  NvSciBufAttrList reconciledInputImageAttrList = nullptr;
  {
    NvSciBufAttrList inputImageAttrList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListCreate(bufModule, &inputImageAttrList));

    NVMEDIA_CHECK(NvMediaIOFAFillNvSciBufAttrList(inputImageAttrList));
    populateInputImageBufAttrList(inputImageAttrList, imageWidth, imageHeight);

    // Reconcile the list and free the input temporary.
    reconciledInputImageAttrList = ReconcileNvSciBufAttrLists(inputImageAttrList);
  }

  // Input buffers

  NvSciCudaInteropBuffer* leftInputBuffer = new NvSciCudaInteropBuffer(reconciledInputImageAttrList);
  NvSciCudaInteropBuffer* rightInputBuffer = new NvSciCudaInteropBuffer(reconciledInputImageAttrList);

  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(iofa, leftInputBuffer->m_nvSciBuf));
  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(iofa, rightInputBuffer->m_nvSciBuf));

  // Output buffer attribute list

  NvSciBufAttrList reconciledOutputImageAttrList = nullptr;
  {
    NvSciBufAttrList outputImageAttrList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListCreate(bufModule, &outputImageAttrList));

    NVMEDIA_CHECK(NvMediaIOFAFillNvSciBufAttrList(outputImageAttrList));
    populateOutputImageBufAttrList(outputImageAttrList, outputWidth, outputHeight, /*forCostMap=*/ false);

    // Reconcile attr list and free the input temporary.
    reconciledOutputImageAttrList = ReconcileNvSciBufAttrLists(outputImageAttrList);
  }

  NvSciCudaInteropBuffer* outputBuffer = new NvSciCudaInteropBuffer(reconciledOutputImageAttrList);
  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(iofa, outputBuffer->m_nvSciBuf));

  // Cost buffer
  NvSciBufAttrList reconciledCostImageAttrList = nullptr;
  {
    NvSciBufAttrList costImageAttrList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListCreate(bufModule, &costImageAttrList));

    NVMEDIA_CHECK(NvMediaIOFAFillNvSciBufAttrList(costImageAttrList));
    populateOutputImageBufAttrList(costImageAttrList, outputWidth, outputHeight, /*forCostMap=*/ true);

    reconciledCostImageAttrList = ReconcileNvSciBufAttrLists(costImageAttrList);
  }

  NvSciCudaInteropBuffer* costBuffer = new NvSciCudaInteropBuffer(reconciledCostImageAttrList);
  NVMEDIA_CHECK(NvMediaIOFARegisterNvSciBufObj(iofa, costBuffer->m_nvSciBuf));

  constexpr int kIterations = 1; // For longer timing tests

  cv::cuda::GpuMat confidence;
  confidence.create(/*rows=*/ outputHeight, /*cols=*/ outputWidth, CV_8U);

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    // Start CUDA processing
    CUDA_CHECK(cuEventRecord(startEvent, hStream));
    // Copy input images to CUDA arrays
    copyCvMatToSurface(cvImageLeft, leftInputBuffer->m_cuArray, hStream);
    copyCvMatToSurface(cvImageRight, rightInputBuffer->m_cuArray, hStream);

    CUDA_CHECK(cuEventRecord(copyInFinishedEvent, hStream));

    // Signal preprocess semaphore
    preSync.signalCudaToNvSci(hStream);

    // Tell OFA to wait on the pre fence for this frame
    NVMEDIA_CHECK(NvMediaIOFAInsertPreNvSciSyncFence(iofa, &preSync.m_nvSciSyncFence));
    // Must clear fence after using it and before the next frame processing cycle
    NvSciSyncFenceClear(&preSync.m_nvSciSyncFence);
    // EOF sync object needs to be provided before frame submission
    NVMEDIA_CHECK(NvMediaIOFASetNvSciSyncObjforEOF(iofa, eofSync.m_nvSciSync));

    // OFA processing
    {
      NvMediaIofaBufArray surfArray;
      memset(&surfArray, 0, sizeof(surfArray));
      surfArray.inputSurface[0] = leftInputBuffer->m_nvSciBuf;
      surfArray.refSurface[0] = rightInputBuffer->m_nvSciBuf;
      surfArray.outSurface[0] = outputBuffer->m_nvSciBuf;
      surfArray.costSurface[0] = costBuffer->m_nvSciBuf;

      NvMediaIofaProcessParams processParams;
      memset(&processParams, 0, sizeof(processParams));
      // Structure pointer is required, but all processing params set to zero is fine

      NVMEDIA_CHECK(NvMediaIOFAProcessFrame(iofa, &surfArray, &processParams, /*pEpiInfo=*/ nullptr, /*pROIParams=*/ nullptr));
    }


    // Get EOF fence and hand it back to CUDA
    NVMEDIA_CHECK(NvMediaIOFAGetEOFNvSciSyncFence(iofa, eofSync.m_nvSciSync, &eofSync.m_nvSciSyncFence));
    eofSync.waitNvSciToCuda(hStream);

    CUDA_CHECK(cuEventRecord(ofaFinishedEvent, hStream));

    // Process cost into confidence
    ofaCostToConfidence(costBuffer->m_cuTex, confidence, lowCostThreshold, highCostThreshold, costCurve, hStream);

    // Copy output surface back to CPU
    copySurfaceToCvMat(outputBuffer->m_cuArray, cvDisparityPreFilter, hStream);

    // Wait for processing to finish on the CPU side
    cuStreamSynchronize(hStream);

    confidence.download(cvConfidence);

    float ms[2];

    cuEventElapsedTime(&ms[0], startEvent, copyInFinishedEvent);
    cuEventElapsedTime(&ms[1], copyInFinishedEvent, ofaFinishedEvent);

    printf("event timings: copyIn %.3fms OFA %.3fms\n", ms[0], ms[1]);
  }

  cv::Mat cvDisparityPostMask;

  if (useOcclusionMask) {
    cv::cuda::GpuMat disparity;
    disparity.create(/*rows=*/ outputHeight, /*cols=*/ outputWidth, CV_16U);

    copySurfaceToGpuMat(outputBuffer->m_cuArray, disparity, hStream);

    // disparityPrescale = 1 / (1 << subpixelFractionalBits). The test
    // uses 5 subpixel bits (matches maxValidDisparityRaw = 128 * 32).
    const float disparityPrescale = 1.0f / 32.0f;
    uint32_t searchWindow = occlusionMaskSearchWindow;
    if (searchWindow == 0) {
      searchWindow = 128 + 1; // max disparity in pixels
    }
    disparityOcclusionMask(
      disparity, confidence,
      /*maxValidDisparityRaw=*/ 128 * 32,
      disparityPrescale,
      searchWindow,
      occlusionMaskHysteresis,
      occlusionMaskConfidenceCeiling,
      /*smearLeftScanPixels=*/ occlusionMaskSmear ? searchWindow : 0,
      hStream);

    cuStreamSynchronize(hStream);
    // Download modified mats
    disparity.download(cvDisparityPostMask);
    confidence.download(cvConfidence);
    cvDisparityPostMask.copyTo(cvDisparity);
  } else {
    cvDisparityPostMask = cvDisparity;
  }


  if (useFGSFilter) {

    // Apply FGS filter. This is the core of the OpenCV ximgproc WLS filter;
    // it expects to receive left and right disparity maps and makes a confidence map from them.
    // We already have a confidence map, so we skip straight to invoking the FastGlobalSmootherFilter, which is
    // what it does internally.

    cv::Mat confFloat, dispFloat;
    cvDisparity.convertTo(dispFloat, CV_32FC1, 1.0 / static_cast<double>(32 << gridSizeShift), 0);
    cvConfidence.convertTo(confFloat, CV_32FC1);

    cv::Mat disp_mul_conf;

#define EPS 1e-43f

    if (useCudaFGSFilter) {
      // FGS dims must match the guide (cvImageLeft) -- only meaningful at gridSizeShift = 0.
      assert(cvImageLeft.cols == dispFloat.cols && cvImageLeft.rows == dispFloat.rows);

      // Texture object over the (already-uploaded) left luma surface, point-sampled in
      // unnormalized coordinates. cudaReadModeNormalizedFloat (the default for 8-bit
      // surfaces with flags=0) returns [0, 1] floats, so we scale fgsSigma by 1/255 to
      // match the CPU's 8-bit-difference sigma convention.
      CUtexObject guideTex;
      {
        CUDA_RESOURCE_DESC resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
        resDesc.res.array.hArray = leftInputBuffer->m_cuArray;

        CUDA_TEXTURE_DESC texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
        texDesc.flags = 0;
        CUDA_CHECK(cuTexObjectCreate(&guideTex, &resDesc, &texDesc, /*resViewDesc=*/ nullptr));
      }

      cv::cuda::GpuMat dispGpu, confGpu, dispMulConfGpu, confFilteredGpu, dispFilteredGpu, finalDispGpu, fusedPair;
      dispGpu.upload(dispFloat);
      confGpu.upload(confFloat);
      cv::cuda::multiply(confGpu, dispGpu, dispMulConfGpu);
      cuStreamSynchronize(hStream);

      static FGSFilterState fgsState;
      uint64_t startTime = currentTimeNs();
      const int partitionIters = getenv("FGS_ITERS") ? atoi(getenv("FGS_ITERS")) : 3;
      const bool useFloat2 = !getenv("FGS_SINGLE_CHANNEL");
      if (useFloat2) {
        // Pack (disp*conf, conf) into a CV_32FC2 in one kernel pass.
        fgsPackDispConfMul(dispGpu, confGpu, fusedPair, hStream);
        // Fused two-channel filter: one Thomas factorization, both data lanes.
        fgsFilter(fgsState, guideTex, fusedPair, fusedPair, fgsLambda, fgsSigma / 255.0f, /*lambda_attenuation=*/ 0.25f, /*num_iter=*/ partitionIters, hStream);
        // Recover the filtered disparity: pair.x / (pair.y + EPS).
        fgsUnpackDivideEps(fusedPair, finalDispGpu, EPS, hStream);
      } else {
        fgsFilter(fgsState, guideTex, dispMulConfGpu, dispFilteredGpu, fgsLambda, fgsSigma / 255.0f, /*lambda_attenuation=*/ 0.25f, /*num_iter=*/ partitionIters, hStream);
        fgsFilter(fgsState, guideTex, confGpu, confFilteredGpu, fgsLambda, fgsSigma / 255.0f, /*lambda_attenuation=*/ 0.25f, /*num_iter=*/ partitionIters, hStream);
        fgsDivideEpsPair(dispFilteredGpu, confFilteredGpu, finalDispGpu, EPS, hStream);
      }
      cuStreamSynchronize(hStream);
      uint64_t endTime = currentTimeNs();
      printf("FGS filter time: %.3f ms\n", deltaTimeMs(startTime, endTime));

      finalDispGpu.download(confFloat);
      CUDA_CHECK(cuTexObjectDestroy(guideTex));

      // Run the CPU path on the same inputs and report max abs diff for validation.
      {
        cv::Mat refDispFloat, refConfFloat;
        cvDisparity.convertTo(refDispFloat, CV_32FC1, 1.0 / static_cast<double>(32 << gridSizeShift), 0);
        cvConfidence.convertTo(refConfFloat, CV_32FC1);
        cv::Mat ref_disp_mul_conf = refConfFloat.mul(refDispFloat);
        cv::Mat ref_conf_filtered;
        cv::Ptr<cv::ximgproc::FastGlobalSmootherFilter> fgs = cv::ximgproc::createFastGlobalSmootherFilter(/*src (guide)=*/ cvImageLeft, fgsLambda, fgsSigma, /*lambda_attenuation=*/ 0.25, /*num_iter=*/ partitionIters);
        fgs->filter(ref_disp_mul_conf, ref_disp_mul_conf);
        fgs->filter(refConfFloat, ref_conf_filtered);
        cv::Mat ref_result = ref_disp_mul_conf.mul(1 / (ref_conf_filtered + EPS));

        cv::Mat absdiff;
        cv::absdiff(confFloat, ref_result, absdiff);
        double maxDiff, meanDiff;
        cv::Point maxLoc;
        cv::minMaxLoc(absdiff, /*minVal=*/ nullptr, &maxDiff, /*minLoc=*/ nullptr, &maxLoc);
        meanDiff = cv::mean(absdiff)[0];
        double refMin, refMax;
        cv::minMaxLoc(ref_result, &refMin, &refMax);
        printf("CUDA-vs-CPU FGS diff: max=%.6f mean=%.6f at (col=%d row=%d) -- ours=%f ref=%f (reference value range [%.3f, %.3f])\n",
          maxDiff, meanDiff, maxLoc.x, maxLoc.y,
          confFloat.at<float>(maxLoc.y, maxLoc.x), ref_result.at<float>(maxLoc.y, maxLoc.x),
          refMin, refMax);
        // Show diff stats by percentile.
        std::vector<float> diffs;
        diffs.reserve(absdiff.total());
        for (int yy = 0; yy < absdiff.rows; ++yy) {
          const float* row = absdiff.ptr<float>(yy);
          for (int xx = 0; xx < absdiff.cols; ++xx) diffs.push_back(row[xx]);
        }
        std::sort(diffs.begin(), diffs.end());
        auto pct = [&](double p) { return diffs[std::min<size_t>(diffs.size() - 1, static_cast<size_t>(p * diffs.size()))]; };
        printf("CUDA-vs-CPU diff percentiles: 50%%=%.6f 90%%=%.6f 99%%=%.6f 99.9%%=%.6f 99.99%%=%.6f\n",
          pct(0.50), pct(0.90), pct(0.99), pct(0.999), pct(0.9999));
      }
    } else {
      // CPU baseline FGS filter
      uint64_t startTime = currentTimeNs();
      disp_mul_conf = confFloat.mul(dispFloat);
      cv::Mat conf_filtered;
      cv::Ptr<cv::ximgproc::FastGlobalSmootherFilter> fgs = cv::ximgproc::createFastGlobalSmootherFilter(/*src (guide)=*/ cvImageLeft, fgsLambda, fgsSigma, /*lambda_attenuation=*/ 0.25, /*num_iter=*/ 3);
      fgs->filter(disp_mul_conf, disp_mul_conf);
      fgs->filter(confFloat, conf_filtered);
      confFloat = disp_mul_conf.mul(1 / (conf_filtered + EPS));
      uint64_t endTime = currentTimeNs();
      printf("FGS filter time: %.3f ms\n", deltaTimeMs(startTime, endTime));
    }

    confFloat.convertTo(cvDisparity, CV_16U, static_cast<double>(32 << gridSizeShift));
  }


  const int kMaxDisparity = 128;
  if (!cvRefDisparity.empty()) {
    // Reference disparity exists, so we will generate a comparison image.

    // Scale our disparity to float from fixed-point range, accounting for grid-size shift.
    cv::Mat floatDisparity;
    cvDisparity.convertTo(floatDisparity, CV_32FC1, 1.0 / static_cast<double>(32 << gridSizeShift), 0);

    cv::Mat floatDisparityPreFilter;
    cvDisparityPreFilter.convertTo(floatDisparityPreFilter, CV_32FC1, 1.0 / static_cast<double>(32 << gridSizeShift), 0);

    cv::Mat cvRefDisparityScaled;
    if (cvRefDisparity.cols != cvDisparity.cols || cvRefDisparity.rows != cvDisparity.rows) {
      cv::resize(cvRefDisparity, cvRefDisparityScaled, cvRefDisparity.size());
    } else {
      cvRefDisparityScaled = cvRefDisparity;
    }

    // Compute abs-diff between ours and reference/ground-truth disparity
    cv::Mat deltaDisparity = cv::abs(floatDisparity - cvRefDisparity);

    // Abs-diff between filtered and unfiltered disparity, so we can see what the filter is doing.
    cv::Mat filterDeltaDisparity = cv::abs(floatDisparity - floatDisparityPreFilter);

    // Compute some stats
    uint32_t bad0_5 = 0, bad1 = 0, bad2 = 0, bad4 = 0;
    float avgError = 0.0f;
    float rmsError = 0.0f;
    for (uint32_t row = 0; row < deltaDisparity.rows; ++row) {
      const float* rowPtr = deltaDisparity.ptr<const float>(row);
      for (uint32_t col = 0; col < deltaDisparity.cols; ++col) {
        float f = rowPtr[col];
        if (f >= 0.5f) bad0_5++;
        if (f >= 1.0f) bad1++;
        if (f >= 2.0f) bad2++;
        if (f >= 4.0f) bad4++;
        avgError += f;
        if (f > (1.0f / 1024.0f)) // prevent NaN
          rmsError += sqrtf(f);
      }
    }
    uint32_t pixelCount = deltaDisparity.cols * deltaDisparity.rows;

    avgError /= static_cast<float>(pixelCount);
    rmsError /= static_cast<float>(pixelCount);
    rmsError = rmsError * rmsError;

    printf("Error stats: bad0.5 %.3f%% || bad1 %.3f%% || bad2 %.3f%% || bad4 %.3f%% || avgError %.3f || rmsError %.3f\n",
      (static_cast<float>(bad0_5) / static_cast<float>(pixelCount)) * 100.0f,
      (static_cast<float>(bad1) / static_cast<float>(pixelCount)) * 100.0f,
      (static_cast<float>(bad2) / static_cast<float>(pixelCount)) * 100.0f,
      (static_cast<float>(bad4) / static_cast<float>(pixelCount)) * 100.0f,
      avgError, rmsError);

    // Colormap everything for visualization
    cv::Mat deltaDisparityColor = colormap(deltaDisparity, 255.0f / kMaxDisparity);
    cv::Mat refDisparityColor = colormap(cvRefDisparity, 255.0f / kMaxDisparity);
    cv::Mat cvDisparityColor = colormap(floatDisparity, 255.0f / kMaxDisparity);

    cv::Mat cvDisparityPreFilterColor = colormap(floatDisparityPreFilter, 255.0f / kMaxDisparity);
    cv::Mat filterDeltaDisparityColor = colormap(filterDeltaDisparity, 255.0f / kMaxDisparity);

    // Confidence just gets to be greyscale
    cv::cvtColor(/*src=*/ cvConfidence, /*dst=*/ cvConfidence, cv::COLOR_GRAY2BGR);

    // Luma surface
    // (TODO: need to downsample this with gridSizeShift != 0)
    cv::Mat luma;
    cv::cvtColor(/*src=*/ cvImageLeft, /*dst=*/ luma, cv::COLOR_GRAY2BGR);

    // placeholder
    cv::Mat empty = cv::Mat::zeros(cvDisparityColor.size(), CV_8UC3);

    cv::Mat postMaskColor = empty;
    if (useOcclusionMask) {
      cv::Mat fDispPostMask;
      cvDisparityPostMask.convertTo(fDispPostMask, CV_32FC1, 1.0 / static_cast<double>(32 << gridSizeShift), 0);
      postMaskColor = colormap(fDispPostMask, 255.0f / kMaxDisparity);
    }

    cv::Mat col0, col1, col2;
    // Stack left column views: unfiltered disparity, confidence, luma
    cv::vconcat(std::array<cv::Mat, 3>{cvDisparityPreFilterColor, cvConfidence, luma}, col0);

    // Stack middle column views: filtered disparity, delta vs. unfiltered, (post-mask disparity or empty)
    cv::vconcat(std::array<cv::Mat, 3>{cvDisparityColor, filterDeltaDisparityColor, postMaskColor}, col1);

    // Stack right column views: ref disparity, delta vs. ref, empty
    cv::vconcat(std::array<cv::Mat, 3>{refDisparityColor, deltaDisparityColor, empty}, col2);

    // Stack the columns together
    cv::Mat outSurface;
    cv::hconcat(std::array<cv::Mat, 3>{col0, col1, col2}, outSurface);
    imwrite("disparity.png", outSurface);

  } else {
    // Scale result and write it to disk. Disparities are in Q10.5 format,
    // so to map it to float, it gets divided by 32. Then the resulting disparity range,
    // from 0 to kMaxDisparity gets mapped to 0-255 for proper output.
    cvDisparity.convertTo(cvDisparity, CV_8UC1, 255.0 / (32 * kMaxDisparity), 0);
    cvDisparityPreFilter.convertTo(cvDisparityPreFilter, CV_8UC1, 255.0 / (32 * kMaxDisparity), 0);

    // Apply JET colormap to turn the disparities into color, reddish hues
    // represent objects closer to the camera, blueish are farther away.
    cv::Mat cvDisparityColor, cvDisparityPreFilterColor;
    applyColorMap(cvDisparity, cvDisparityColor, cv::COLORMAP_JET);
    applyColorMap(cvDisparityPreFilter, cvDisparityPreFilterColor, cv::COLORMAP_JET);

    // Confidence just gets to be greyscale
    cv::cvtColor(/*src=*/ cvConfidence, /*dst=*/ cvConfidence, cv::COLOR_GRAY2BGR);


    // Stack the views together: disparity (original), disparity (filtered), confidence
    cv::Mat outSurface;
    cv::vconcat(std::array<cv::Mat, 3>{cvDisparityPreFilterColor, cvDisparityColor, cvConfidence}.data(), 3, outSurface);
    imwrite("disparity.png", outSurface);
  }

  // Cleanup
  NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciSyncObj(iofa, preSync.m_nvSciSync));
  NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciSyncObj(iofa, eofSync.m_nvSciSync));
  NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciBufObj(iofa, leftInputBuffer->m_nvSciBuf));
  NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciBufObj(iofa, rightInputBuffer->m_nvSciBuf));
  NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciBufObj(iofa, outputBuffer->m_nvSciBuf));
  NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciBufObj(iofa, costBuffer->m_nvSciBuf));

  NVMEDIA_CHECK(NvMediaIOFADestroy(iofa));
  iofa = nullptr;

  return 0;
}
