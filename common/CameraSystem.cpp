#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/glmCvInterop.h"
#include "common/FxThreading.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICUDA.h"
#include "common/UndistortRectifyKernel.h"
#include "imgui.h"
#include <iostream>
#include <set>
#include <assert.h>
#include <cuda.h>
#include <epoxy/gl.h> // epoxy_is_desktop_gl

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 14, CV_64F);

CameraSystem::CameraSystem(ICameraProvider* cam) :
  calibrationFilename("calibration.yml"),
  m_cameraProvider(cam),
  m_calibrationDataRevision(0) {

  m_cameras.resize(cameraProvider()->streamCount());

  // Compute stereo distortion map size.
  // The stereoDistortionMap will be generated at half-resolution relative to the stream size.
  // This introduces a tiny amount of distortion error (typical <0.1px) in exchange for
  // cutting the per-frame distortion map generation time to 1/4.
  m_distortionMapWidth = cameraProvider()->streamWidth() / 2;
  m_distortionMapHeight = cameraProvider()->streamHeight() / 2;
}


void CameraSystem::processFrame() {
  // Regenerate the stereoDistortionMap for each stereo view by refilling the
  // per-row R_y * iR buffer on the CPU and launching the kernel. R_y is identity
  // for now -- the IMU integration will populate non-trivial per-row rotations
  // in a later change. See SYNCHRONIZATION ASSUMPTION in UndistortRectifyKernel.h.

  for (size_t viewIdx = 0; viewIdx < m_views.size(); ++viewIdx) {
    View& v = m_views[viewIdx];
    if (!v.isStereo) continue;
    if (v.rsParamsHost == nullptr) continue; // No calibration yet.

    for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
      float* eyeBase = v.rsPerRowIRHost + (eyeIdx * m_distortionMapHeight * 9);
      const cv::Mat& iR = v.rsIRBase[eyeIdx];
      for (int y = 0; y < m_distortionMapHeight; ++y) {
        // Identity R_y placeholder: every row gets the same iR. When IMU
        // integration lands, replace this with R_y * iR per row.
        for (int r = 0; r < 3; ++r) {
          for (int col = 0; col < 3; ++col) {
            eyeBase[y * 9 + r * 3 + col] = static_cast<float>(iR.at<double>(r, col));
          }
        }
      }
    }

    for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
      const UndistortRectifyParams* paramsDev = reinterpret_cast<const UndistortRectifyParams*>(v.rsParamsDevice) + eyeIdx;
      const float* perRowDev = reinterpret_cast<const float*>(v.rsPerRowIRDevice) + (eyeIdx * m_distortionMapHeight * 9);
      launchUndistortRectifyKernel(
        paramsDev, perRowDev,
        v.stereoDistortionCudaSurface[eyeIdx],
        m_distortionMapWidth, m_distortionMapHeight,
        RHICUDA::defaultAsyncStream);
    }
  }
}

bool CameraSystem::loadCalibrationData() {

  cv::FileStorage fs(calibrationFilename.c_str(), cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("Unable to open calibration data file\n");
    return false;
  }

  return loadCalibrationData(fs);
}

bool CameraSystem::loadCalibrationData(cv::FileStorage& fs) {
  try {
    cv::FileNode camerasFn = fs["cameras"];
    if (camerasFn.isSeq()) {
      for (size_t cameraIdx = 0; cameraIdx < m_cameras.size(); ++cameraIdx) {
        cv::FileNode cfn = camerasFn[cameraIdx];
        if (cfn.isMap()) {
          Camera& c = m_cameras[cameraIdx];

          cfn["intrinsicMatrix"] >> c.intrinsicMatrix;
          cfn["distortionCoeffs"] >> c.distCoeffs;
        }
      }
    }

    cv::FileNode viewsFn = fs["views"];
    if (viewsFn.isSeq()) {
      m_views.resize(viewsFn.size());

      for (size_t viewIdx = 0; viewIdx < viewsFn.size(); ++viewIdx) {
        cv::FileNode vfn = viewsFn[viewIdx];
        if (vfn.isMap()) {
          View& v = m_views[viewIdx];

          vfn["isStereo"] >> v.isStereo;

          if (v.isStereo) {
            cv::Mat tmpMat;

            vfn["isPanorama"] >> v.isPanorama;
            vfn["leftCameraIndex"] >> v.cameraIndices[0];
            vfn["rightCameraIndex"] >> v.cameraIndices[1];
            vfn["stereoTranslationInitialGuess"] >> tmpMat;
            v.stereoTranslationInitialGuess = glmVec3FromCV(tmpMat);
            vfn["stereoRotation"] >> v.stereoRotation;
            vfn["stereoTranslation"] >> v.stereoTranslation;
            //vfn["essentialMatrix"] >> v.essentialMatrix;
            //vfn["fundamentalMatrix"] >> v.fundamentalMatrix;

            vfn["viewTranslation"] >> tmpMat;
            v.viewTranslation = glmVec3FromCV(tmpMat);
            vfn["viewRotation"] >> tmpMat;
            v.viewRotation = glmVec3FromCV(tmpMat);
          } else {
            vfn["cameraIndex"] >> v.cameraIndices[0];
            v.cameraIndices[1] = v.cameraIndices[0];
          }
        }
      }
    }

  } catch (const std::exception& ex) {
    printf("Unable to load calibration data: %s\n", ex.what());
    return false;
  }

  for (size_t cameraIdx = 0; cameraIdx < m_cameras.size(); ++cameraIdx) {
    Camera& c = cameraAtIndex(cameraIdx);
    if (c.hasIntrinsicCalibration()) {
      updateCameraIntrinsicDistortionParameters(cameraIdx);
      printf("CameraSystem::loadCalibrationData: Camera %zu: Loaded intrinsic calibration\n", cameraIdx);
    } else {
      printf("CameraSystem::loadCalibrationData: Camera %zu: Intrinsic calibration required\n", cameraIdx);
    }
  }

  for (size_t viewIdx = 0; viewIdx < m_views.size(); ++viewIdx) {
    View& v = viewAtIndex(viewIdx);
    if (v.isStereo) {
      if (v.hasStereoCalibration()) {
        updateViewStereoDistortionParameters(viewIdx);
        printf("CameraSystem::loadCalibrationData: View %zu: Loaded stereo calibration\n", viewIdx);
      } else {
        printf("CameraSystem::loadCalibrationData: View %zu: Stereo calibration required\n", viewIdx);
      }
    }
  }

  return true;
}

void CameraSystem::saveCalibrationData() {
  {
    // Try to save a backup of the previous calibration
    char backupFn[256];
    time_t t = time(NULL);
    strftime(backupFn, sizeof(backupFn), "calibration-backup-%Y%m%d-%H%M%S.yml", localtime(&t));
    if (rename(calibrationFilename.c_str(), backupFn) == 0) {
      printf("Saved backup of previous calibration to %s\n", backupFn);
    } else {
      printf("Couldn't save backup of previous calibration (to %s): %s\n", backupFn, strerror(errno));
    }
  }

  cv::FileStorage fs(calibrationFilename.c_str(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  saveCalibrationData(fs);
}

void CameraSystem::saveCalibrationData(cv::FileStorage& fs) {


  fs.startWriteStruct("cameras", cv::FileNode::SEQ, cv::String());
  for (size_t cameraIdx = 0; cameraIdx < m_cameras.size(); ++cameraIdx) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());
    Camera& c = cameraAtIndex(cameraIdx);
    if (c.hasIntrinsicCalibration()) {
      fs.write("intrinsicMatrix", c.intrinsicMatrix);
      fs.write("distortionCoeffs", c.distCoeffs);
    }
    fs.endWriteStruct();
  }
  fs.endWriteStruct();

  fs.startWriteStruct("views", cv::FileNode::SEQ, cv::String());
  for (size_t viewIdx = 0; viewIdx < m_views.size(); ++viewIdx) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());
    View& v = viewAtIndex(viewIdx);
    fs.write("isStereo", v.isStereo);
    if (v.isStereo) {
      fs.write("isPanorama", v.isPanorama);
      fs.write("leftCameraIndex", (int) v.cameraIndices[0]);
      fs.write("rightCameraIndex", (int) v.cameraIndices[1]);
      fs.write("stereoTranslationInitialGuess", cv::Mat(cvVec3FromGlm(v.stereoTranslationInitialGuess)));
      if (v.hasStereoCalibration()) {
        fs.write("stereoRotation", v.stereoRotation);
        fs.write("stereoTranslation", v.stereoTranslation);
        //fs.write("essentialMatrix", v.essentialMatrix);
        //fs.write("fundamentalMatrix", v.fundamentalMatrix);
      }
      fs.write("viewTranslation", cv::Mat(cvVec3FromGlm(v.viewTranslation)));
      fs.write("viewRotation", cv::Mat(cvVec3FromGlm(v.viewRotation)));
    } else {
      fs.write("cameraIndex", (int) v.cameraIndices[0]);
    }

    fs.endWriteStruct();
  }
}

glm::mat4 CameraSystem::viewWorldTransform(size_t viewIdx) const {
  assert(viewIdx < views());

  glm::mat4 vt = viewAtIndex(viewIdx).viewLocalTransform();

  // View 0's tranform is stored relative to the world (user input)
  // Other view tranforms are calibrated, so they're stored relative to view 0 to maintain calibration when adjusting the view offset.

  if (viewIdx == 0)
    return vt;
  else
    return viewAtIndex(0).viewLocalTransform() * vt;
}

void CameraSystem::updateCameraIntrinsicDistortionParameters(size_t cameraIdx) {
  cv::Size imageSize = cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight());
  float alpha = 0.25; // scaling factor. 0 = no invalid pixels in output (no black borders), 1 = use all input pixels
  cv::Mat map1, map2;
  Camera& c = cameraAtIndex(cameraIdx);

  // Intrinic-only remap -- no rectification transform, using optimized matrix
  c.optimizedMatrix = cv::getOptimalNewCameraMatrix(c.intrinsicMatrix, c.distCoeffs, imageSize, alpha, cv::Size(), NULL, /*centerPrincipalPoint=*/ true);
  cv::initUndistortRectifyMap(c.intrinsicMatrix, c.distCoeffs, cv::noArray(), c.optimizedMatrix, imageSize, CV_32F, map1, map2);

  // Compute FOV
  {
    double focalLength, aspectRatio; // not valid without a real aperture size, which we don't bother providing
    cv::Point2d principalPoint;

    cv::calibrationMatrixValues(c.optimizedMatrix, cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), 0.0, 0.0, c.fovX, c.fovY, focalLength, principalPoint, aspectRatio);
  }

  c.intrinsicDistortionMap = generateGPUDistortionMap(map1, map2, imageSize);

  m_calibrationDataRevision += 1;
}

void CameraSystem::updateViewStereoDistortionParameters(size_t viewIdx) {
  cv::Size imageSize = cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight());
  cv::Mat map1, map2;
  View& v = viewAtIndex(viewIdx);
  Camera& leftC = cameraAtIndex(v.cameraIndices[0]);
  Camera& rightC = cameraAtIndex(v.cameraIndices[1]);

  // Compute rectification/projection transforms from the stereo calibration data
  float alpha = -1.0f; //0.25;

  cv::stereoRectify(
    leftC.intrinsicMatrix, leftC.distCoeffs,
    rightC.intrinsicMatrix, rightC.distCoeffs,
    cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()),
    v.stereoRotation, v.stereoTranslation,
    v.stereoRectification[0], v.stereoRectification[1],
    v.stereoProjection[0], v.stereoProjection[1],
    v.stereoDisparityToDepth,
    /*flags=*/ cv::CALIB_ZERO_DISPARITY, alpha, cv::Size(),
    &v.stereoValidROI[0], &v.stereoValidROI[1]);

  // Camera info dump
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    printf("\n ===== View %zu | %s Camera (%u) ===== \n", viewIdx, eyeIdx == 0 ? "Left" : "Right", v.cameraIndices[eyeIdx]);

    std::cout << "* Stereo Rectification matrix:" << std::endl
              << v.stereoRectification[eyeIdx] << std::endl;
    std::cout << "* Stereo Projection matrix:" << std::endl
              << v.stereoProjection[eyeIdx] << std::endl;
    std::cout << "* Stereo Valid ROI:" << std::endl
              << v.stereoValidROI[eyeIdx] << std::endl;


    const double apertureSize = 6.35; // mm, 1/4" sensor. TODO parameterize
    double fovX, fovY, focalLength, aspectRatio;
    cv::Point2d principalPoint;

    Camera& c = cameraAtIndex(v.cameraIndices[eyeIdx]);
    cv::calibrationMatrixValues(c.intrinsicMatrix, cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Intrinsic matrix: FOV %.1f x %.1f deg, approx focal length %.2fmm\n", fovX, fovY, focalLength);
    cv::calibrationMatrixValues(c.optimizedMatrix, cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Optimized matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
    cv::calibrationMatrixValues(cv::Mat(v.stereoProjection[eyeIdx], cv::Rect(0, 0, 3, 3)), cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Stereo projection matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
  }
  printf("\n ==================== \n");
  std::cout << "View " << viewIdx << " Stereo translation:" << std::endl
            << v.stereoTranslation << std::endl;
  std::cout << "View " << viewIdx << " Stereo rotation matrix:" << std::endl
            << v.stereoRotation << std::endl;

  {
    // Compute FOV. Should be the same for stereoProjection[0] and [1], so we only keep a single value.
    double focalLength, aspectRatio;
    cv::Point2d principalPoint;

    cv::calibrationMatrixValues(cv::Mat(v.stereoProjection[0], cv::Rect(0, 0, 3, 3)), cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), 0.0, 0.0, v.fovX, v.fovY, focalLength, principalPoint, aspectRatio);
  }

  // Check the valid image regions for a failed stereo calibration. A bad calibration will usually result in a valid ROI for one or both views with a 0-pixel dimension.
  /*
  if (needStereoCalibration) {
    if (stereoValidROI[0].area() == 0 || stereoValidROI[1].area() == 0) {
      printf("Stereo calibration failed: one or both of the valid image regions has zero area.\n");
      goto retryStereoCalibration;
    }
  }
*/

  // Allocate the interop surface for each eye's distortion map. These start
  // empty -- the kernel fill below populates them.
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    v.stereoDistortionMap[eyeIdx] = rhi()->newInteropSurface(
      m_distortionMapWidth, m_distortionMapHeight,
      RHISurfaceDescriptor(kSurfaceFormat_RG16));

    // Allocate CUDA surface object
    CUDA_SAFE_FREE_SURFACE_OBJECT(v.stereoDistortionCudaSurface[eyeIdx]);
    CUDA_SAFE_FREE_TEXTURE_OBJECT(v.stereoDistortionCudaTexture[eyeIdx]);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = v.stereoDistortionMap[eyeIdx]->cudaArray();
    CUDA_CHECK(cudaCreateSurfaceObject(&v.stereoDistortionCudaSurface[eyeIdx], &resDesc));

    // CUDA texture object
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; // Linear filtering
    texDesc.readMode = cudaReadModeNormalizedFloat; // Read as normalized float -- surface datatype is unorm16x2
    texDesc.normalizedCoords = 1; // Use normalized coordinates during read, since we're doing bilinear filtering to decouple the distortion map size from the camera image size.
    CUDA_CHECK(cudaCreateTextureObject(&v.stereoDistortionCudaTexture[eyeIdx], &resDesc, &texDesc, /*resViewDesc=*/ nullptr));
  }

  // Allocate the pinned per-View rolling-shutter buffers if this is the first
  // time this View has been configured. (updateViewStereoDistortionParameters
  // can be called multiple times across recalibration; we don't re-alloc since
  // image dimensions don't change.)
  if (v.rsParamsHost == nullptr) {
    CUdeviceptr paramsDev = 0;
    CUDA_CHECK(cuMemHostAlloc(reinterpret_cast<void**>(&v.rsParamsHost),
      2 * sizeof(UndistortRectifyParams),
      CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostGetDevicePointer(&paramsDev, v.rsParamsHost, 0));
    v.rsParamsDevice = static_cast<unsigned long long>(paramsDev);

    const size_t perRowBytes = 2 * static_cast<size_t>(m_distortionMapHeight) * 9 * sizeof(float);
    CUdeviceptr perRowDev = 0;
    CUDA_CHECK(cuMemHostAlloc(reinterpret_cast<void**>(&v.rsPerRowIRHost),
      perRowBytes,
      CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostGetDevicePointer(&perRowDev, v.rsPerRowIRHost, 0));
    v.rsPerRowIRDevice = static_cast<unsigned long long>(perRowDev);
  }

  // Populate static per-eye params and stash iR for per-frame refold.
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    Camera& c = cameraAtIndex(v.cameraIndices[eyeIdx]);
    UndistortRectifyParams& p = v.rsParamsHost[eyeIdx];

    for (int r = 0; r < 3; ++r) {
      for (int col = 0; col < 3; ++col) {
        p.cameraMatrix[r * 3 + col] = static_cast<float>(c.intrinsicMatrix.at<double>(r, col));
      }
    }
    for (int i = 0; i < 5; ++i) {
      p.distCoeffs[i] = static_cast<float>(c.distCoeffs.at<double>(i, 0));
    }
    p.distortionMapWidth = m_distortionMapWidth;
    p.distortionMapHeight = m_distortionMapHeight;
    p.streamWidth = imageSize.width;
    p.streamHeight = imageSize.height;

    // For a half-resolution distortion map, we scale 2x to convert distortion map pixels to stream pixels.
    p.distortionMapToStreamScale[0] = 2.0f;
    p.distortionMapToStreamScale[1] = 2.0f;
    // We also apply a 1-pixel bias to make each half-res distortion map sample land in the center of the 2x2 pixel neighborhood from its corresponding native-res map
    p.distortionMapToStreamBias[0] = 1.0f;
    p.distortionMapToStreamBias[1] = 1.0f;

    // cv::initUndistortRectifyMap takes a 3x4 newProjection but uses only its upper-left 3x3.
    cv::Mat newCam3x3 = v.stereoProjection[eyeIdx](cv::Rect(0, 0, 3, 3));
    v.rsIRBase[eyeIdx] = (newCam3x3 * v.stereoRectification[eyeIdx]).inv();
  }

  // Initial fill: identity per-row R (i.e., per-row entries == iRBase). Gives
  // stereoDistortionMap[] valid contents before the first processFrame.
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    float* eyeBase = v.rsPerRowIRHost + (eyeIdx * m_distortionMapHeight * 9);
    for (int y = 0; y < m_distortionMapHeight; ++y) {
      for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) {
          eyeBase[y * 9 + r * 3 + col] = static_cast<float>(v.rsIRBase[eyeIdx].at<double>(r, col));
        }
      }
    }
  }
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    const UndistortRectifyParams* paramsDev = reinterpret_cast<const UndistortRectifyParams*>(v.rsParamsDevice) + eyeIdx;
    const float* perRowDev = reinterpret_cast<const float*>(v.rsPerRowIRDevice) + (eyeIdx * m_distortionMapHeight * 9);
    launchUndistortRectifyKernel(
      paramsDev, perRowDev,
      v.stereoDistortionCudaSurface[eyeIdx],
      m_distortionMapWidth, m_distortionMapHeight,
      RHICUDA::defaultAsyncStream);
  }

  // The initial fill will be CUDA-accessible on the async stream, but still requires `syncCUDAToRHI()` to be called before use in rendering.

  m_calibrationDataRevision += 1;
}

RHISurface::ptr CameraSystem::generateGPUDistortionMap(cv::Mat map1, cv::Mat map2, cv::Size sourceImageSize) {
  assert(map1.rows == map2.rows && map1.cols == map2.cols);
  size_t width = map1.cols;
  size_t height = map1.rows;

  // map1 and map2 should contain absolute x and y coords for sampling the input image, in pixel scale (map1 is 0-1280, map2 is 0-720, etc) -- this is the output format of cv::initUndistortRectifyMap
  // Combine the maps into a buffer we can upload. Remap the absolute pixel coordinates to UV (0...1) range to save work in the pixel shader.
  uint16_t* distortionMapTmp = new uint16_t[width * height * 2];

  // Apply half-texel bias to line up the coordinate systems. This makes shader-based remapping line up exactly with cv::remap.
  float texelBiasX = 0.5f;
  float texelBiasY = 0.5f;
  for (int y = 0; y < height; ++y) {
    uint16_t* rowP = distortionMapTmp + (y * width * 2);
    for (int x = 0; x < width; ++x) {
      // .at(row, col) -- Y rows, X columns.
      *(rowP++) = static_cast<uint16_t>(glm::clamp((map1.at<float>(y, x) + texelBiasX) / static_cast<float>(sourceImageSize.width), 0.0f, 1.0f) * 65535.0f);
      *(rowP++) = static_cast<uint16_t>(glm::clamp((map2.at<float>(y, x) + texelBiasY) / static_cast<float>(sourceImageSize.height), 0.0f, 1.0f) * 65535.0f);
    }
  }

  RHISurface::ptr distortionMap = rhi()->newInteropSurface(width, height, RHISurfaceDescriptor(kSurfaceFormat_RG16));

  // CUDA 2D host-to-array memcpy into distortionMap
  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.srcHost = distortionMapTmp;
  copyDescriptor.srcPitch = sizeof(uint16_t) * width * 2;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = (CUarray) distortionMap->cudaArray();

  copyDescriptor.WidthInBytes = sizeof(uint16_t) * width * 2;
  copyDescriptor.Height = height;

  CUDA_CHECK(cuMemcpy2D(&copyDescriptor));

  delete[] distortionMapTmp;
  return distortionMap;
}

size_t CameraSystem::createMonoView(size_t cameraIndex) {
  assert(cameraIndex < m_cameras.size());

  m_views.resize(m_views.size() + 1);
  size_t newViewIndex = m_views.size() - 1;

  View& v = m_views[newViewIndex];

  v.isStereo = false;
  v.cameraIndices[0] = cameraIndex;
  v.cameraIndices[1] = cameraIndex;

  return newViewIndex;
}

size_t CameraSystem::createStereoView(size_t leftCameraIndex, size_t rightCameraIndex) {
  assert(leftCameraIndex < m_cameras.size());
  assert(rightCameraIndex < m_cameras.size());

  m_views.resize(m_views.size() + 1);
  size_t newViewIndex = m_views.size() - 1;

  View& v = m_views[newViewIndex];

  v.isStereo = true;
  v.cameraIndices[0] = leftCameraIndex;
  v.cameraIndices[1] = rightCameraIndex;

  return newViewIndex;
}
