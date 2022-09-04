#ifdef HAVE_VPI2
#include "common/DepthMapGeneratorVPI.h"
#include "imgui.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/VPIUtil.h"
#include "common/glmCvInterop.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda.h>
#include <nppi.h>
#include <epoxy/gl.h> // epoxy_is_desktop_gl
#include <vpi/VPI.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>
#include <vpi/algo/Rescale.h>
#include <vpi/algo/StereoDisparity.h>

#include <string>

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const uint32_t kVPIMaxDisparity = 128; // must be {64, 128, 256}

DepthMapGeneratorVPI::DepthMapGeneratorVPI() : DepthMapGenerator(kDepthBackendVPI) {
  m_algoDownsampleX = 4;
  m_algoDownsampleY = 4;
  m_maxDisparity = kVPIMaxDisparity;
  // VPI 2.0 docs say:
  // Returned [disparity] values are in Q10.5 format, i.e., signed fixed point with 5 fractional bits.
  // Divide it by 32.0f to convert it to floating point.
  m_disparityPrescale = (1.0f / 32.0f);

  vpiInitStereoDisparityEstimatorParams(&m_params);
}

DepthMapGeneratorVPI::~DepthMapGeneratorVPI() {
  vpiStreamDestroy(m_masterStream);
  cuStreamDestroy(m_masterCUStream);
  vpiEventDestroy(m_masterFrameStartEvent);
  vpiEventDestroy(m_masterFrameFinishedEvent);
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)
void DepthMapGeneratorVPI::internalLoadSettings(cv::FileStorage& fs) {
  cv::FileNode vpi = fs["vpi"];
  if (vpi.isMap()) {
    cv::read(vpi["confidenceThreshold"], m_params.confidenceThreshold, m_params.confidenceThreshold);
    cv::read(vpi["quality"], m_params.quality, m_params.quality);
    // maxDisparity and windowSize are not used on the CUDA backend.

    //readNode(vpi, algorithm);
  }
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void DepthMapGeneratorVPI::internalSaveSettings(cv::FileStorage& fs) {
  fs.startWriteStruct(cv::String("vpi"), cv::FileNode::MAP, cv::String());
    fs.write("confidenceThreshold", m_params.confidenceThreshold);
    fs.write("quality", m_params.quality);
    // maxDisparity and windowSize are not used on the CUDA backend.

    //writeNode(fs, algorithm);
  fs.endWriteStruct();
}
#undef writeNode


// input format should be CV_32F
VPIPayload cvRemapToVPIRemapPayload(cv::Mat m1, cv::Mat m2) {
  assert(m1.type() == CV_32F && m2.type() == CV_32F);
  assert(m1.cols == m2.cols && m1.rows == m2.rows);

  VPIWarpMap warpMap;
  memset(&warpMap, 0, sizeof(warpMap));

  warpMap.grid.numHorizRegions  = 1;
  warpMap.grid.numVertRegions   = 1;
  warpMap.grid.horizInterval[0] = 1;
  warpMap.grid.vertInterval[0]  = 1;
  warpMap.grid.regionWidth[0]   = m1.cols;
  warpMap.grid.regionHeight[0]  = m1.rows;

  VPI_CHECK(vpiWarpMapAllocData(&warpMap));
  VPI_CHECK(vpiWarpMapGenerateIdentity(&warpMap));

  for (size_t pY = 0; pY < warpMap.numVertPoints; ++pY) {
    for (size_t pX = 0; pX < warpMap.numHorizPoints; ++pX) {
      VPIKeypointF32& kp = reinterpret_cast<VPIKeypointF32*>(reinterpret_cast<uint8_t*>(warpMap.keypoints) + (pY * warpMap.pitchBytes))[pX];
      // Keypoints can be generated that go off the active region for alignment reasons -- for example,
      // asking for a 1920x1080 warp map will generate a 1920x1088 keypoint grid.

      unsigned int inX = std::min<unsigned int>(kp.x, m1.cols - 1);
      unsigned int inY = std::min<unsigned int>(kp.y, m1.rows - 1);
      assert(inX < m1.cols && inY < m1.rows);

      kp.x = m1.at<float>(inY, inX);
      kp.y = m2.at<float>(inY, inX);
    }
  }

  VPIPayload payload = NULL;
  VPI_CHECK(vpiCreateRemap(VPI_BACKEND_CUDA, &warpMap, &payload));

  vpiWarpMapFreeData(&warpMap);
  return payload;
}

void DepthMapGeneratorVPI::internalUpdateViewData() {

  if (!m_masterStream) {
    // First time init of shared objects
    CUDA_CHECK(cuStreamCreate(&m_masterCUStream, CU_STREAM_NON_BLOCKING));
    VPI_CHECK(vpiStreamCreateWrapperCUDA(m_masterCUStream, /*flags=*/ 0, &m_masterStream));
    VPI_CHECK(vpiEventCreate(0, &m_masterFrameStartEvent));
    VPI_CHECK(vpiEventCreate(0, &m_masterFrameFinishedEvent));

    // Setup initial event state
    vpiEventRecord(m_masterFrameStartEvent, m_masterStream);
    vpiEventRecord(m_masterFrameFinishedEvent, m_masterStream);
  }

  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
    auto vd = viewDataAtIndex(viewIdx);

    vd->releaseVPIResources();

    if (!vd->m_isStereoView)
      continue;

    vd->updateDisparityTexture(internalWidth(), internalHeight(), kSurfaceFormat_R16i);

    {
      cv::Size imageSize = cv::Size(inputWidth(), inputHeight());
      cv::Mat m1, m2;
      PER_EYE {
        // TODO validate CameraSystem:::updateViewStereoDistortionParameters against the distortion map initialization code here
        CameraSystem::Camera& cam = m_cameraSystem->cameraAtIndex(v.cameraIndices[eyeIdx]);
        cv::initUndistortRectifyMap(cam.intrinsicMatrix, cam.distCoeffs, v.stereoRectification[eyeIdx], v.stereoProjection[eyeIdx], imageSize, CV_32F, m1, m2);
        vd->m_remapPayload[eyeIdx] = cvRemapToVPIRemapPayload(m1, m2);
      }
    }

    // TODO we should only enable BACKEND_CPU if disparity debug access is requested
    uint64_t imageFlags = VPI_BACKEND_CUDA | VPI_BACKEND_CPU | VPI_REQUIRE_BACKENDS;

    PER_EYE VPI_CHECK(vpiImageCreate(inputWidth(),    inputHeight(),    VPI_IMAGE_FORMAT_Y8_ER, imageFlags, &vd->m_grey[eyeIdx]));
    PER_EYE VPI_CHECK(vpiImageCreate(inputWidth(),    inputHeight(),    VPI_IMAGE_FORMAT_Y8_ER, imageFlags, &vd->m_rectifiedGrey[eyeIdx]));
    PER_EYE VPI_CHECK(vpiImageCreate(internalWidth(), internalHeight(), VPI_IMAGE_FORMAT_Y8_ER, imageFlags, &vd->m_resized[eyeIdx]));

    VPI_CHECK(vpiImageCreate(internalWidth(), internalHeight(), VPI_IMAGE_FORMAT_S16,    imageFlags, &vd->m_disparity));
    VPI_CHECK(vpiImageCreate(internalWidth(), internalHeight(), VPI_IMAGE_FORMAT_U16,    imageFlags, &vd->m_confidence));

    VPIStereoDisparityEstimatorCreationParams creationParams;
    vpiInitStereoDisparityEstimatorCreationParams(&creationParams);
    creationParams.maxDisparity = kVPIMaxDisparity;

    if (vd->m_isVerticalStereo) {
      PER_EYE VPI_CHECK(vpiImageCreate(internalWidth(), internalHeight(), VPI_IMAGE_FORMAT_Y8_ER, imageFlags, &vd->m_resizedTransposed[eyeIdx]));

      VPI_CHECK(vpiImageCreate(internalHeight(), internalWidth(), VPI_IMAGE_FORMAT_S16, imageFlags, &vd->m_disparityTransposed));
      VPI_CHECK(vpiImageCreate(internalHeight(), internalWidth(), VPI_IMAGE_FORMAT_U16, imageFlags, &vd->m_confidenceTransposed));
      VPI_CHECK(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, internalHeight(), internalWidth(), VPI_IMAGE_FORMAT_Y8_ER, &creationParams, &vd->m_disparityEstimator));
    } else {
      VPI_CHECK(vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, internalWidth(), internalHeight(), VPI_IMAGE_FORMAT_Y8_ER, &creationParams, &vd->m_disparityEstimator));
    }
  }
}

// Only works with single-plane images; not a lot of error checking.
void copyVPIImageToSurface(VPIImage img, RHISurface::ptr surface, CUstream stream) {

  VPIImageData imgData;
  VPI_CHECK(vpiImageLockData(img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData));

  size_t copyWidth = std::min<size_t>(surface->width(), imgData.buffer.pitch.planes[0].width);
  size_t copyHeight = std::min<size_t>(surface->height(), imgData.buffer.pitch.planes[0].height);

  CUarray pSurfaceMip0Array;
  CUDA_CHECK(cuGraphicsResourceSetMapFlags(surface->cuGraphicsResource(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
  CUDA_CHECK(cuGraphicsMapResources(1, &surface->cuGraphicsResource(), stream));
  CUDA_CHECK(cuGraphicsSubResourceGetMappedArray(&pSurfaceMip0Array, surface->cuGraphicsResource(), /*arrayIndex=*/ 0, /*mipLevel=*/ 0));

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.srcDevice = (CUdeviceptr) imgData.buffer.pitch.planes[0].data;
  copyDescriptor.srcPitch = imgData.buffer.pitch.planes[0].pitchBytes;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = pSurfaceMip0Array;

  int bpp = vpiPixelTypeGetBitsPerPixel(imgData.buffer.pitch.planes[0].pixelType);
  assert((bpp/8) == rhiSurfaceFormatSize(surface->format())); // sanity check

  copyDescriptor.WidthInBytes = copyWidth * rhiSurfaceFormatSize(surface->format());
  copyDescriptor.Height = copyHeight;
  if (stream) {
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  } else {
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
  }

  CUDA_CHECK(cuGraphicsUnmapResources(1, &surface->cuGraphicsResource(), stream));
  VPI_CHECK(vpiImageUnlock(img));
}

// Copies to a tightly packed CPU array. Only works with single-plane images.
void copyVPIImageToCPU(VPIImage img, void* outArray, size_t bytesPerPixel) {

  VPIImageData imgData;
  VPI_CHECK(vpiImageLockData(img, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));

  VPIImagePlanePitchLinear& plane = imgData.buffer.pitch.planes[0];

  int bpp = vpiPixelTypeGetBitsPerPixel(plane.pixelType);
  assert((bpp/8) == bytesPerPixel); // sanity check

  for (size_t y = 0; y < plane.height; ++y) {
    void* outRow = reinterpret_cast<uint8_t*>(outArray) + (y * plane.width * bytesPerPixel);
    memcpy(outRow, reinterpret_cast<uint8_t*>(plane.data) + (y * plane.pitchBytes), plane.width * bytesPerPixel);
  }

  VPI_CHECK(vpiImageUnlock(img));
}


void DepthMapGeneratorVPI::internalProcessFrame() {
  // Ensure previous processing is finished
  VPI_CHECK(vpiEventSync(m_masterFrameFinishedEvent));
  // Copy the results from the previous frame to GL
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    copyVPIImageToSurface(vd->m_disparity, vd->m_disparityTexture, m_masterCUStream);

    if (m_debugDisparityCPUAccessEnabled) {
      vd->ensureDebugCPUAccessEnabled(/*bytesPerPixel=*/ 2);
      copyVPIImageToCPU(vd->m_disparity, vd->m_debugCPUDisparity, vd->m_debugCPUDisparityBytesPerPixel);
    }

    if (m_populateDebugTextures) {
      if (!vd->m_leftGray)
        vd->m_leftGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      if (!vd->m_rightGray)
        vd->m_rightGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      copyVPIImageToSurface(vd->m_resized[0], vd->m_leftGray, m_masterCUStream);
      copyVPIImageToSurface(vd->m_resized[1], vd->m_rightGray, m_masterCUStream);
    }
  }

  internalGenerateDisparityMips();

  if (m_enableProfiling) {
    // Collect profiling data from previous frame
    vpiEventElapsedTimeMillis(m_masterFrameStartEvent, m_masterFrameFinishedEvent, &m_frameTimeMs);
    for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
      auto vd = viewDataAtIndex(viewIdx);
      if (!vd->m_isStereoView)
        continue;

      vpiEventElapsedTimeMillis(vd->m_frameStartedEvent, vd->m_convertFinishedEvent, &vd->m_convertTimeMs);
      vpiEventElapsedTimeMillis(vd->m_convertFinishedEvent, vd->m_remapFinishedEvent, &vd->m_remapTimeMs);
      vpiEventElapsedTimeMillis(vd->m_remapFinishedEvent, vd->m_rescaleFinishedEvent, &vd->m_rescaleTimeMs);
      vpiEventElapsedTimeMillis(vd->m_rescaleFinishedEvent, vd->m_frameFinishedEvent, &vd->m_stereoTimeMs);
    }
  }

  VPI_CHECK(vpiEventRecord(m_masterFrameStartEvent, m_masterStream));
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    VPI_CHECK(vpiEventRecord(vd->m_frameStartedEvent, vd->m_stream));
    // Setup phase

    VPIConvertImageFormatParams convParams;
    vpiInitConvertImageFormatParams(&convParams);

    // Convert to greyscale
    PER_EYE VPI_CHECK(vpiSubmitConvertImageFormat(vd->m_stream, VPI_BACKEND_CUDA, m_cameraSystem->cameraProvider()->vpiImage(m_cameraSystem->viewAtIndex(viewIdx).cameraIndices[eyeIdx]), vd->m_grey[eyeIdx], &convParams));
    VPI_CHECK(vpiEventRecord(vd->m_convertFinishedEvent, vd->m_stream));

    // Remap
    PER_EYE VPI_CHECK(vpiSubmitRemap(vd->m_stream, VPI_BACKEND_CUDA, vd->m_remapPayload[eyeIdx], vd->m_grey[eyeIdx], vd->m_rectifiedGrey[eyeIdx], VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));
    VPI_CHECK(vpiEventRecord(vd->m_remapFinishedEvent, vd->m_stream));

    // Rescale
    PER_EYE VPI_CHECK(vpiSubmitRescale(vd->m_stream, VPI_BACKEND_CUDA, vd->m_rectifiedGrey[eyeIdx], vd->m_resized[eyeIdx], VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0));

    if (vd->m_isVerticalStereo) {
      // Ensure all CUDA work has been enqueued
      vpiStreamFlush(vd->m_stream);
      PER_EYE {
        VPIImageData input, output;

        VPI_CHECK(vpiImageLockData(vd->m_resized[eyeIdx], VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &input));
        VPI_CHECK(vpiImageLockData(vd->m_resizedTransposed[eyeIdx], VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &output));

        NppiSize sz; // input image size
        sz.width  = input.buffer.pitch.planes[0].width;
        sz.height = input.buffer.pitch.planes[0].height;

        nppiTranspose_16u_C1R_Ctx((const Npp16u*) input.buffer.pitch.planes[0].data, input.buffer.pitch.planes[0].pitchBytes, (Npp16u*) output.buffer.pitch.planes[0].data, output.buffer.pitch.planes[0].pitchBytes, sz, vd->m_nppStreamContext);
        VPI_CHECK(vpiImageUnlock(vd->m_resized[eyeIdx]));
        VPI_CHECK(vpiImageUnlock(vd->m_resizedTransposed[eyeIdx]));
      }
    }

    // Stereo algo phase
    VPI_CHECK(vpiEventRecord(vd->m_rescaleFinishedEvent, vd->m_stream));

    VPIImage* stereoInput = vd->m_isVerticalStereo ? vd->m_resizedTransposed : vd->m_resized;
    VPIImage outDisparity = vd->m_isVerticalStereo ? vd->m_disparityTransposed : vd->m_disparity;
    VPIImage outConfidence = vd->m_isVerticalStereo ? vd->m_confidenceTransposed : vd->m_confidence;

    VPI_CHECK(vpiSubmitStereoDisparityEstimator(vd->m_stream, VPI_BACKEND_CUDA, vd->m_disparityEstimator, stereoInput[0], stereoInput[1], outDisparity, outConfidence, &m_params));

    if (vd->m_isVerticalStereo) {
      // Transpose disparity and confidence images back
      vpiStreamFlush(vd->m_stream);
      VPIImageData dispTr, confTr; // inputs
      VPIImageData disp, conf; // outputs

      VPI_CHECK(vpiImageLockData(vd->m_disparityTransposed, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &dispTr));
      VPI_CHECK(vpiImageLockData(vd->m_confidenceTransposed, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &confTr));
      VPI_CHECK(vpiImageLockData(vd->m_disparity, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &disp));
      VPI_CHECK(vpiImageLockData(vd->m_confidence, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &conf));

      NppiSize dispTrSz, confTrSz; // input image size
      dispTrSz.width  = dispTr.buffer.pitch.planes[0].width;
      dispTrSz.height = dispTr.buffer.pitch.planes[0].height;
      confTrSz.width  = confTr.buffer.pitch.planes[0].width;
      confTrSz.height = confTr.buffer.pitch.planes[0].height;

      nppiTranspose_16u_C1R_Ctx((const Npp16u*) dispTr.buffer.pitch.planes[0].data, dispTr.buffer.pitch.planes[0].pitchBytes, (Npp16u*) disp.buffer.pitch.planes[0].data, disp.buffer.pitch.planes[0].pitchBytes, dispTrSz, vd->m_nppStreamContext);
      nppiTranspose_16u_C1R_Ctx((const Npp16u*) confTr.buffer.pitch.planes[0].data, confTr.buffer.pitch.planes[0].pitchBytes, (Npp16u*) conf.buffer.pitch.planes[0].data, conf.buffer.pitch.planes[0].pitchBytes, confTrSz, vd->m_nppStreamContext);
      VPI_CHECK(vpiImageUnlock(vd->m_disparityTransposed));
      VPI_CHECK(vpiImageUnlock(vd->m_confidenceTransposed));
      VPI_CHECK(vpiImageUnlock(vd->m_disparity));
      VPI_CHECK(vpiImageUnlock(vd->m_confidence));
    }

    // Finish and join to master stream
    VPI_CHECK(vpiEventRecord(vd->m_frameFinishedEvent, vd->m_stream));
    VPI_CHECK(vpiStreamWaitEvent(m_masterStream, vd->m_frameFinishedEvent));
  }
  VPI_CHECK(vpiEventRecord(m_masterFrameFinishedEvent, m_masterStream));

}

void DepthMapGeneratorVPI::internalRenderIMGUI() {
  ImGui::SliderInt("Confidence Threshold", &m_params.confidenceThreshold, 0, 65535);
  ImGui::SliderInt("Quality", &m_params.quality, 1, 255); // TODO range?
}

void DepthMapGeneratorVPI::internalRenderIMGUIPerformanceGraphs() {
  if (!m_enableProfiling)
    return;

  // TODO: graphs
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    ImGui::Text("View [%zu]: Convert %.3f Remap %.3fms Rescale %.3fms Stereo %.3fms", viewIdx, vd->m_convertTimeMs, vd->m_remapTimeMs, vd->m_rescaleTimeMs, vd->m_stereoTimeMs);
  }
  ImGui::Text("Total: %.3fms", m_frameTimeMs);
}

#endif // HAVE_VPI2
