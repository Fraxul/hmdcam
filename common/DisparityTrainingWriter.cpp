#include "common/DisparityTrainingWriter.h"
#include "common/CameraSystem.h"
#include "common/DepthMapGenerator.h"
#include "common/PosixFileUtil.h"
#include "imgui/imgui.h"
#include "rhi/cuda/CudaUtil.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

DisparityTrainingWriter::DisparityTrainingWriter(DepthMapGenerator* _depthMapGenerator) :
  m_depthMapGenerator(_depthMapGenerator),
  m_cameraSystem(_depthMapGenerator->cameraSystem()) {

  m_algoInputWidth = m_depthMapGenerator->algoInputWidth();
  m_algoInputHeight = m_depthMapGenerator->algoInputHeight();
  m_internalWidth = m_depthMapGenerator->internalWidth();
  m_internalHeight = m_depthMapGenerator->internalHeight();
  m_rgbBytes = static_cast<size_t>(m_algoInputWidth) * m_algoInputHeight * 3;
  m_disparityBytes = static_cast<size_t>(m_internalWidth) * m_internalHeight * 2;
  m_costBytes = static_cast<size_t>(m_internalWidth) * m_internalHeight;
  m_leftRGBOffset = 0;
  m_rightRGBOffset = m_rgbBytes;
  m_disparityOffset = 2 * m_rgbBytes;
  m_costOffset = ((2 * m_rgbBytes) + m_disparityBytes);
  m_viewOutput.resize(cameraSystem()->views());
  m_dumpRing = new AsyncGpuDumpRing(/*maxInFlight=*/ m_viewOutput.size() * 4, (2 * m_rgbBytes) + m_disparityBytes + m_costBytes);
}

DisparityTrainingWriter::~DisparityTrainingWriter() {
  if (m_active) {
    m_active = false;
    m_dumpRing->drainBlocking();
    closeOutputDescriptors();
  }
  delete m_dumpRing;
}

void DisparityTrainingWriter::setActive(bool active) {
  if (m_active == active)
    return;

  if (active) {
    char dirName[256];
    std::time_t now = std::time(nullptr);
    std::strftime(dirName, sizeof(dirName), "training-data-%Y%m%d-%H%M%S", std::localtime(&now));

    if (mkdir(dirName, 0777) != 0) {
      printf("DisparityTrainingWriter::setActive(): Can't create directory \"%s\": %s\n", dirName, strerror(errno));
      return;
    }

    m_rootDirFd = open(dirName, O_DIRECTORY | O_PATH);
    if (m_rootDirFd < 0) {
      printf("DisparityTrainingWriter::setActive(): Can't open directory \"%s\": %s\n", dirName, strerror(errno));
      return;
    }

    for (size_t viewIdx = 0; viewIdx < m_viewOutput.size(); ++viewIdx) {
      snprintf(dirName, sizeof(dirName), "view%zu", viewIdx);
      if (mkdirat(m_rootDirFd, dirName, 0777) != 0) {
        printf("DisparityTrainingWriter::setActive(): Can't create directory \"%s\": %s\n", dirName, strerror(errno));
        closeOutputDescriptors();
        return;
      }

      m_viewOutput[viewIdx].dirFd = openat(m_rootDirFd, dirName, O_DIRECTORY | O_PATH);
      if (m_viewOutput[viewIdx].dirFd < 0) {
        printf("DisparityTrainingWriter::setActive(): Can't open directory \"%s\": %s\n", dirName, strerror(errno));
        closeOutputDescriptors();
        return;
      }

      m_viewOutput[viewIdx].metadataFd = openat(m_viewOutput[viewIdx].dirFd, "metadata.csv", O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (m_viewOutput[viewIdx].metadataFd < 0) {
        printf("DisparityTrainingWriter::setActive(): Can't open metadata.csv: %s\n", strerror(errno));
        closeOutputDescriptors();
        return;
      }

      // frame_index keys the per-sample image files.
      // the quaternion is the camera's inter-frame rotation (current frame relative to the previous one), in (w, x, y, z) order.
      const char* csvHeader = "frame_index,timestamp,qw,qx,qy,qz,leftExposureTimeNs,rightExposureTimeNs,leftISO,rightISO,leftDigitalGain,rightDigitalGain,leftAnalogGain,rightAnalogGain,leftSceneLux,rightSceneLux,leftAwbCct,rightAwbCct,\n";
      write(m_viewOutput[viewIdx].metadataFd, csvHeader, strlen(csvHeader));

      int viewDataFd = openat(m_viewOutput[viewIdx].dirFd, "viewData.yml", O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (viewDataFd < 0) {
        printf("DisparityTrainingWriter::setActive(): Can't open viewData.yml: %s\n", strerror(errno));
        closeOutputDescriptors();
        return;
      }

      // Serialize per-view calibration data
      {
        cv::FileStorage fs(cv::String(), cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_YAML);
        const CameraSystem::View& v = cameraSystem()->viewAtIndex(viewIdx);
        const CameraSystem::Camera& leftC = cameraSystem()->cameraAtIndex(v.cameraIndices[0]);
        const CameraSystem::Camera& rightC = cameraSystem()->cameraAtIndex(v.cameraIndices[1]);

        fs.write("nativeWidth", static_cast<int>(cameraSystem()->cameraProvider()->streamWidth()));
        fs.write("nativeHeight", static_cast<int>(cameraSystem()->cameraProvider()->streamHeight()));

        // View transform for view 0 is user input; transforms for subsequent views are stored relative to view 0.
        // We skip the user-input transform on view 0 (write zeros), so all views' transforms end up in the same space, where view 0 is the origin.
        fs.write("viewTranslationMeters", cv::Mat(cvVec3FromGlm(viewIdx == 0 ? glm::vec3(0.0f) : v.viewTranslation)));
        fs.write("viewRotationDegrees", cv::Mat(cvVec3FromGlm(viewIdx == 0 ? glm::vec3(0.0f) : v.viewRotation)));

        if (v.isStereo) {
          fs.write("stereoRotation", v.stereoRotation);
          fs.write("stereoTranslation", v.stereoTranslation);
          fs.write("stereoDisparityToDepth", v.stereoDisparityToDepth);
          fs.write("leftStereoRectification", v.stereoRectification[0]);
          fs.write("rightStereoRectification", v.stereoRectification[1]);
          fs.write("leftStereoProjection", v.stereoProjection[0]);
          fs.write("rightStereoProjection", v.stereoProjection[1]);
        }

        fs.write("leftCameraIntrinsicMatrix", leftC.intrinsicMatrix);
        fs.write("rightCameraIntrinsicMatrix", rightC.intrinsicMatrix);
        fs.write("leftCameraDistortionCoeffs", leftC.distCoeffs);
        fs.write("rightCameraDistortionCoeffs", rightC.distCoeffs);


        depthMapGenerator()->internalWriteTrainingAnnotationsForView(viewIdx, fs);

        cv::String viewDataStr = fs.releaseAndGetString();
        if (!writeFully(viewDataFd, viewDataStr.data(), viewDataStr.size())) {
          printf("DisparityTrainingWriter::setActive(): Error writing to viewData.yml: %s\n", strerror(errno));
          close(viewDataFd);
          closeOutputDescriptors();
          return;
        }
      }

      close(viewDataFd);
    }

    // Generate view metadata

    m_frameIndex = 0;
    m_writtenSamples = 0;
    m_droppedSamples = 0;
    m_active = true;
  } else {
    // Stop accepting new samples, then block until the in-flight writes drain so we can safely close
    // the output fds they reference.
    m_active = false;
    m_dumpRing->drainBlocking();
    closeOutputDescriptors();
    printf("DisparityTrainingWriter: wrote %u samples, dropped %u\n", m_writtenSamples, m_droppedSamples);
  }
}

void DisparityTrainingWriter::beginFrame() {
  if (!isActive())
    return;

  ++m_frameIndex;

  if (m_frameLimit && m_frameIndex > m_frameLimit) {
    // Frame limit reached.
    setActive(false);
  }
}

void DisparityTrainingWriter::closeOutputDescriptors() {
  for (ViewOutput& vo : m_viewOutput) {
    if (vo.metadataFd >= 0) {
      close(vo.metadataFd);
      vo.metadataFd = -1;
    }
    if (vo.dirFd >= 0) {
      close(vo.dirFd);
      vo.dirFd = -1;
    }
  }
  if (m_rootDirFd >= 0) {
    close(m_rootDirFd);
    m_rootDirFd = -1;
  }
}

AsyncGpuDumpRing::Slot* DisparityTrainingWriter::acquireSlot() {
  if (!m_active)
    return nullptr;

  AsyncGpuDumpRing::Slot* slot = m_dumpRing->acquire();
  if (!slot)
    m_droppedSamples += 1; // Not draining the ring fast enough -- drop this sample.
  return slot;
}

// Issues a tightly-packed DtoH copy of a pitched device mat into a slot host region.
static void copyDeviceMatToHost(void* dstHost, const cv::cuda::GpuMat& src, size_t rowBytes, size_t height, CUstream stream) {
  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.srcDevice = (CUdeviceptr) src.cudaPtr();
  copyDescriptor.srcPitch = src.step;
  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.dstHost = dstHost;
  copyDescriptor.dstPitch = rowBytes; // tightly-packed host destination
  copyDescriptor.WidthInBytes = rowBytes;
  copyDescriptor.Height = height;
  CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
}

void DisparityTrainingWriter::copyColor(AsyncGpuDumpRing::Slot* slot, const cv::cuda::GpuMat& leftRGB, const cv::cuda::GpuMat& rightRGB, CUstream stream) {
  uint8_t* base = static_cast<uint8_t*>(slot->hostPtr);
  copyDeviceMatToHost(base + m_leftRGBOffset, leftRGB, static_cast<size_t>(m_algoInputWidth) * 3, m_algoInputHeight, stream);
  copyDeviceMatToHost(base + m_rightRGBOffset, rightRGB, static_cast<size_t>(m_algoInputWidth) * 3, m_algoInputHeight, stream);
}

void DisparityTrainingWriter::copyDisparity(AsyncGpuDumpRing::Slot* slot, const cv::cuda::GpuMat& disparityU16, CUstream stream) {
  uint8_t* base = static_cast<uint8_t*>(slot->hostPtr);
  copyDeviceMatToHost(base + m_disparityOffset, disparityU16, static_cast<size_t>(m_internalWidth) * 2, m_internalHeight, stream);
}

void DisparityTrainingWriter::copyCost(AsyncGpuDumpRing::Slot* slot, const cv::cuda::GpuMat& costU8, CUstream stream) {
  uint8_t* base = static_cast<uint8_t*>(slot->hostPtr);
  copyDeviceMatToHost(base + m_costOffset, costU8, m_internalWidth, m_internalHeight, stream);
}

void DisparityTrainingWriter::submit(AsyncGpuDumpRing::Slot* slot, size_t viewIndex, uint64_t frameIndex, uint64_t timestampNs,
  const ICameraProvider::FrameMetadata& leftFrameMetadata,
  const ICameraProvider::FrameMetadata& rightFrameMetadata,
  const glm::quat& interframeRotation, CUstream stream) {

  // Record completion of the DtoH copies the caller just issued on this stream.
  CUDA_CHECK(cuEventRecord(slot->copyDoneEvent, stream));

  // Metadata row is tiny: write it synchronously here (main thread) so worker threads never contend
  // on the CSV fd. It references frameIndex, which keys the image filenames written asynchronously.
  int metadataFd = m_viewOutput[viewIndex].metadataFd;
  if (metadataFd >= 0) {
    char row[512];
    int rowLength = snprintf(row, sizeof(row), "%016lu,%016lu,%.12f,%.12f,%.12f,%.12f,%lu,%lu,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%u,%u,\n",
      static_cast<unsigned long>(frameIndex),
      timestampNs,
      interframeRotation.w, interframeRotation.x, interframeRotation.y, interframeRotation.z,
      leftFrameMetadata.sensorExposureTimeNs, rightFrameMetadata.sensorExposureTimeNs,
      leftFrameMetadata.sensorSensitivityISO, rightFrameMetadata.sensorSensitivityISO,
      leftFrameMetadata.ispDigitalGain, rightFrameMetadata.ispDigitalGain,
      leftFrameMetadata.sensorAnalogGain, rightFrameMetadata.sensorAnalogGain,
      leftFrameMetadata.sceneLux, rightFrameMetadata.sceneLux,
      leftFrameMetadata.awbCct, rightFrameMetadata.awbCct);

    writeFully(metadataFd, row, rowLength);
  }

  int viewDirFd = m_viewOutput[viewIndex].dirFd;
  m_dumpRing->dispatch(slot, [this, viewDirFd, frameIndex](AsyncGpuDumpRing::Slot* writeSlot) {
    writeSample(writeSlot, viewDirFd, frameIndex);
  });
  m_writtenSamples += 1;
}

// Opens `filename` under viewDirFd and writes an ASCII header followed by a tightly-packed pixel
// plane. Matches the raw, no-compression dump strategy of CalibrationWriter.
static void writeImageFile(int viewDirFd, const char* filename, const char* header, const void* pixels, size_t pixelBytes) {
  int fd = openat(viewDirFd, filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    printf("DisparityTrainingWriter: openat(\"%s\") error: %s\n", filename, strerror(errno));
    return;
  }
  if (!writeFully(fd, header, strlen(header)) || !writeFully(fd, pixels, pixelBytes))
    printf("DisparityTrainingWriter: write(\"%s\") error: %s\n", filename, strerror(errno));
  close(fd);
}

void DisparityTrainingWriter::writeSample(AsyncGpuDumpRing::Slot* slot, int viewDirFd, uint64_t frameIndex) {
  uint8_t* base = static_cast<uint8_t*>(slot->hostPtr);

  // The 16-bit disparity plane must be big-endian (MSB first) for a spec-conformant 16-bit PGM.
  // The Orin is little-endian, so byte-swap the plane in place before writing. This runs on a
  // worker thread, off the capture hot path.
  uint16_t* disparity = reinterpret_cast<uint16_t*>(base + m_disparityOffset);
  size_t disparitySampleCount = static_cast<size_t>(m_internalWidth) * m_internalHeight;
  for (size_t i = 0; i < disparitySampleCount; ++i)
    disparity[i] = __builtin_bswap16(disparity[i]);

  char filename[64];
  char header[64];

  snprintf(filename, sizeof(filename), "%016lu_left.ppm", static_cast<unsigned long>(frameIndex));
  snprintf(header, sizeof(header), "P6\n%u %u\n255\n", m_algoInputWidth, m_algoInputHeight);
  writeImageFile(viewDirFd, filename, header, base + m_leftRGBOffset, m_rgbBytes);

  snprintf(filename, sizeof(filename), "%016lu_right.ppm", static_cast<unsigned long>(frameIndex));
  // header (P6, algoInput size) is unchanged from the left view.
  writeImageFile(viewDirFd, filename, header, base + m_rightRGBOffset, m_rgbBytes);

  snprintf(filename, sizeof(filename), "%016lu_disparity.pgm", static_cast<unsigned long>(frameIndex));
  snprintf(header, sizeof(header), "P5\n%u %u\n65535\n", m_internalWidth, m_internalHeight);
  writeImageFile(viewDirFd, filename, header, base + m_disparityOffset, m_disparityBytes);

  snprintf(filename, sizeof(filename), "%016lu_cost.pgm", static_cast<unsigned long>(frameIndex));
  snprintf(header, sizeof(header), "P5\n%u %u\n255\n", m_internalWidth, m_internalHeight);
  writeImageFile(viewDirFd, filename, header, base + m_costOffset, m_costBytes);
}

void DisparityTrainingWriter::renderIMGUI() {
  if (m_active) {
    if (ImGui::Button("Stop training capture"))
      setActive(false);
    ImGui::SameLine();
    ImGui::Text("Written: %u || Dropped: %u", m_writtenSamples, m_droppedSamples);
  } else {
    if (ImGui::Button("Start training capture (100 frames)")) {
      m_frameLimit = 100;
      setActive(true);
    }
    if (ImGui::Button("Start training capture (300 frames)")) {
      m_frameLimit = 300;
      setActive(true);
    }
    if (ImGui::Button("Start training capture (unlimited frames)")) {
      m_frameLimit = 0;
      setActive(true);
    }
  }
}
