#include "OfaFlowUtil.h"
#include "common/tegra/NvSciUtil.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICUDA.h"
#include <cstring>

namespace CameraImuCalib {

void populateImageBufAttrList(NvSciBufAttrList& attrList, uint32_t width, uint32_t height,
  NvSciBufAttrValColorFmt colorFmt, bool setColorStd) {
  NvSciBufType bufType = NvSciBufType_Image;
  NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
  NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

  uint32_t planeCount = 1;
  uint32_t planeWidths[] = {width};
  uint32_t planeHeights[] = {height};
  bool cpuAccessFlag = false;

  NvSciBufAttrValColorFmt planecolorfmts[] = {colorFmt};
  NvSciBufAttrValColorStd planecolorstds[] = {NvSciColorStd_REC709_ER};
  NvSciBufAttrValImageScanType planescantype[] = {NvSciBufScan_ProgressiveType};

  CUuuid devUUID;
  CUDA_CHECK(cuDeviceGetUuid(&devUUID, RHICUDA::cudaDevice));

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
    {         NvSciBufGeneralAttrKey_GpuId,       &devUUID,        sizeof(devUUID)},
    {   NvSciBufImageAttrKey_PlaneColorStd, planecolorstds, sizeof(planecolorstds)}, // optional, last
  };
  size_t attrCount = sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair);
  if (!setColorStd)
    attrCount -= 1; // drop the trailing PlaneColorStd entry

  NVSCI_CHECK(NvSciBufAttrListSetAttrs(attrList, imgBufAttrs, attrCount));
}

void copyCvMatToSurface(const cv::Mat& mat, CUarray arr, CUstream stream) {
  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.srcHost = mat.ptr();
  copyDescriptor.srcPitch = mat.step;
  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = arr;
  copyDescriptor.WidthInBytes = static_cast<size_t>(mat.cols) * mat.elemSize();
  copyDescriptor.Height = mat.rows;
  if (stream)
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  else
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
}

void copySurfaceToCvMat(CUarray arr, cv::Mat& mat, CUstream stream) {
  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.srcArray = arr;
  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.dstHost = mat.ptr();
  copyDescriptor.dstPitch = mat.step;
  copyDescriptor.WidthInBytes = static_cast<size_t>(mat.cols) * mat.elemSize();
  copyDescriptor.Height = mat.rows;
  if (stream)
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  else
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
}

} // namespace CameraImuCalib
