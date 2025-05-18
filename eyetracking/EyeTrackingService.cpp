#include "EyeTrackingService.h"
#include "eyeProcessing.h"
#include "common/Timing.h"
#include "common/mmfile.h"
#include "imgui.h"
#include "SingleEyeFitter/projection.h"
#include <cuda.h>
#include <npp.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <NvInfer.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <math.h>

#if 0
#define FRAME_DEBUG_LOG printf
#else
#define FRAME_DEBUG_LOG(...)
#endif

const char* calibrationFilename = "eyetracking-calibration.yml";

static inline glm::vec2 toGlm(const cv::Point2f& p) { return glm::vec2(p.x, p.y); }
static inline cv::Point2f toCv(const glm::vec2& p) { return cv::Point2f(p[0], p[1]); }

static inline glm::vec2 vec2AtAngleDeg(float deg) { return glm::vec2(cosf(glm::radians(deg)), sinf(glm::radians(deg))); }
static inline glm::vec2 vec2AtAngle(float rad) { return glm::vec2(cosf(rad), sinf(rad)); }

template <typename T> void anglesToVector(T roll, T pitch, T yaw, T* outVec) {
  // Mathematica: evaluated EulerMatrix[{roll, pitch, yaw}, {3, 1, 2}] . {0, 0, 1}
  // result: {Cos[yaw] Sin[pitch] Sin[roll] + Cos[roll] Sin[yaw], -Cos[roll] Cos[yaw] Sin[pitch] + Sin[roll] Sin[yaw], Cos[pitch] Cos[yaw]}
  outVec[0] = (cos(yaw)*sin(pitch)*sin(roll)) + (cos(roll)*sin(yaw));
  outVec[1] = (-cos(roll)*cos(yaw)*sin(pitch)) + (sin(roll)*sin(yaw));
  outVec[2] = (cos(pitch)*cos(yaw));
}

template <typename T> void vectorToAngles(const T* vec, T& outPitch, T& outYaw, bool toDegrees) {
  // Note: argument ordering is atan2(y, x)

  // Pitch is rotation around / flattening along the X axis, where the 2d plane is Z, Y
  outPitch = atan2(-vec[1], vec[2]);

  // Yaw is rotation around / flattening along the Y axis, where the 2d plane is Z, X
  //outYaw = atan2(vec[0], vec[2]); // simple, but doesn't round-trip
  outYaw = atan2(vec[0], sqrt((vec[2] * vec[2]) + (vec[1] * vec[1]))); // based on Mathematica's ToSphericalCoordinates[], round-trips with anglesToVector (when roll == 0)
  if (outYaw >= M_PI)
    outYaw -= (2.0 * M_PI);

  if (toDegrees) {
    outPitch = glm::degrees(outPitch);
    outYaw = glm::degrees(outYaw);
  }
}

class InferLogger : public nvinfer1::ILogger {
public:
  InferLogger() = default;
  virtual ~InferLogger() = default;

  virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {

    const char* severityTable[] = {
      "InternalError",
      "Error",
      "Warning",
      "Info",
      "Verbose"
    };

    if (severity >= Severity::kVERBOSE)
      return;

    fprintf(stderr, "[%s] %s\n", severityTable[(int) severity], msg);
  }
};

static void printDims(const nvinfer1::Dims& d) {
  printf("[%d]{", d.nbDims);

  for (int32_t i = 0; i < d.nbDims; ++i) {
    printf("%ld%s", d.d[i], (i == (d.nbDims - 1) ? "" : ", "));
  }
  printf("}");
}

static void dumpIOBindings(nvinfer1::ICudaEngine* engine) {
  for (int32_t ioTensorIdx = 0; ioTensorIdx < engine->getNbIOTensors(); ++ioTensorIdx) {
    const char* dataTypes[] = {"fp32", "fp16", "int8", "int32", "bool8", "uint8"};

    const char* n = engine->getIOTensorName(ioTensorIdx);
    bool input = engine->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT;

    nvinfer1::Dims d = engine->getTensorShape(n);

    nvinfer1::DataType dt = engine->getTensorDataType(n);
    nvinfer1::TensorLocation loc = engine->getTensorLocation(n);

    printf("[%d] %s %s Loc=%s Dims=",
      ioTensorIdx, n, input ? "(input)" : "(output)",
      loc == nvinfer1::TensorLocation::kDEVICE ? "(device)" : "(host)");
    printDims(d);

    printf(" DataType=%s Format=%u (%s)", dataTypes[(int) dt], (uint32_t) engine->getTensorFormat(n), engine->getTensorFormatDesc(n));

    int32_t vDim = engine->getTensorVectorizedDim(n);
    if (vDim >= 0) {
      printf(" VectorizedDim=%d", vDim);
      printf(" Bytes/Component=%d Components/Element=%d", engine->getTensorBytesPerComponent(n), engine->getTensorComponentsPerElement(n));
    }
    printf("\n");
  }
}

const char* getIOTensorName(nvinfer1::ICudaEngine* engine, nvinfer1::TensorIOMode ioMode) {
  int32_t nbIOTensors = engine->getNbIOTensors();
  for (int32_t i = 0; i < nbIOTensors; ++i) {
    const char* name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) == ioMode)
      return name;
  }
  assert(false && "getIOTensorName: no IO tensors matching given mode");
  return nullptr;
}

const char* getSingleInputTensorName(nvinfer1::ICudaEngine* engine) { return getIOTensorName(engine, nvinfer1::TensorIOMode::kINPUT); }
const char* getSingleOutputTensorName(nvinfer1::ICudaEngine* engine) { return getIOTensorName(engine, nvinfer1::TensorIOMode::kOUTPUT); }

template <typename T> T clamp(T value, T min_, T max_) {
  return std::min<T>(max_, std::max<T>(min_, value));
}

// Fast conversion of 0....255 range uint8 values to -1...1 range fp16 values
// Has a tiny precision loss around input 127 / output zero:
//
// Input  Expected fp16     Actual fp16
// ------------------------------------
//   124  -0.02745 a707   -0.02747 a708
//   125  -0.01961 a505   -0.01962 a506
//   126  -0.01176 a206   -0.01178 a208
//   127  -0.00392 9c04   -0.00394 9c08
//   128   0.00392 1c04    0.00391 1c00
//   129   0.01176 2206    0.01175 2204
//   130   0.01961 2505    0.01959 2504
//
// All other values are bitwise-identical to the reference algorithm, which converts/rescales in fp32 and then downcasts to fp16:
//     _Float16 f16 = static_cast<_Float16>((static_cast<float>(input_u8) / 127.5f) - 1.0f)
//
// (The accuracy of doing the convert/rescale in fp16 is worse than this version)
//
// Requires compiler ARM NEON FP16 support to be enabled: -march=armv8.2-a+fp16 (for both gcc and clang)
//
void convertUnorm8ToSnormFp16(const uint8_t* inU8, void* outFP16, const size_t elementCount) {
  // Only support chunks of 8 elements right now
  assert((elementCount & 7) == 0);

  float16x8_t* vectorOut = reinterpret_cast<float16x8_t*>(outFP16);
  const uint16x8_t signFixup = vdupq_n_u16(0x8000);
  for (size_t chunkIdx = 0; chunkIdx < (elementCount / 8); ++chunkIdx) {
    // Load 8x 8-bit values and widen to 16 bit
    uint16x8_t ux16 = vmovl_u8(vld1_u8(inU8 + (chunkIdx * 8)));

    // Replicate low 8 to high 8
    ux16 = vsliq_n_u16(ux16, ux16, 8);

    // Fixup sign bit for unorm to snorm conversion, then reinterpret as s16
    int16x8_t x = vreinterpretq_s16_u16(veorq_u16(ux16, signFixup));

    // Convert signed fixed-point to fp16
    float16x8_t f = vcvtq_n_f16_s16(x, 15);

    // Write output
    vectorOut[chunkIdx] = f;
  }
}

// Convert 0...255 uint8 values to DLA-int8 -127...127 range
// Conversion equation: out = max(in - 128, -127)
// The input value 0 would map to -128, but that is not valid for the DLA's narrow int8 range;
// input 0 is clipped to output -127.

void convertUnorm8ToDLAInt8(const uint8_t* inU8, void* outDLAInt8, size_t elementCount) {
  // Only support chunks of 16 elements right now
  assert((elementCount & 15) == 0);

  int8x16_t* vectorOut = reinterpret_cast<int8x16_t*>(outDLAInt8);
  const uint8x16_t signFixup = vdupq_n_u8(0x80);
  const int8x16_t cutoff = vdupq_n_s8(-127);

  for (size_t chunkIdx = 0; chunkIdx < (elementCount / 16); ++chunkIdx) {

    // Load 16x 8-bit values
    uint8x16_t ux = vld1q_u8(inU8 + (chunkIdx * 16));

    // Fixup sign bit for unorm to snorm conversion, then reinterpret as s8
    int8x16_t x = vreinterpretq_s8_u8(veorq_u8(ux, signFixup));

    // Max with cutoff value to remove out-of-bounds -128 value (DLA int8 is narrow, so -127...127)
    x = vmaxq_s8(x, cutoff);

    // Write output
    vectorOut[chunkIdx] = x;
  }
}


EyeTrackingService::EyeTrackingService() {

  loadCalibrationData();

  // Figure out the CUDA stream priority range and create a low-priority stream,
  // then create an NPP stream context associated with that stream
  {
    int leastPriority = 0, greatestPriority = 0;
    CUDA_CHECK(cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority));
    printf("CUDA stream priority range: %d least -> %d greatest\n", leastPriority, greatestPriority);

    PER_EYE {
      CUDA_CHECK(cuStreamCreateWithPriority(&m_processingState[eyeIdx].m_cuStream, CU_STREAM_NON_BLOCKING, leastPriority));

      // Create NPP stream context for our stream
      CUstream prevNppStream = nppGetStream();
      NPP_CHECK(nppSetStream(m_processingState[eyeIdx].m_cuStream));
      NPP_CHECK(nppGetStreamContext(&m_processingState[eyeIdx].m_nppContext));
      nppSetStream(prevNppStream);
    }
  }

  // Detect unified addressing support
  // (used for optimizing the GPU->CPU transfer of the TensorRT output buffer)
  int unifiedAddressingSupport = 0;
  {
    CUdevice dev;
    CUDA_CHECK(cuCtxGetDevice(&dev));
    cuDeviceGetAttribute(&unifiedAddressingSupport, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
    printf("unifiedAddressingSupport = %d\n", unifiedAddressingSupport);
  }

  PER_EYE {
    // Profiling events
    CUDA_CHECK(cuEventCreate(&m_processingState[eyeIdx].m_frameProcessingStartEvent, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&m_processingState[eyeIdx].m_frameROIEndEvent, CU_EVENT_BLOCKING_SYNC));
    CUDA_CHECK(cuEventCreate(&m_processingState[eyeIdx].m_frameSegmentationStartEvent, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&m_processingState[eyeIdx].m_framePostProcessingStartEvent, CU_EVENT_DEFAULT));
    CUDA_CHECK(cuEventCreate(&m_processingState[eyeIdx].m_frameProcessingEndEvent, CU_EVENT_BLOCKING_SYNC));

    // Record good initial state for events
    CUDA_CHECK(cuEventRecord(m_processingState[eyeIdx].m_frameProcessingStartEvent, m_processingState[eyeIdx].m_cuStream));
    CUDA_CHECK(cuEventRecord(m_processingState[eyeIdx].m_frameROIEndEvent, m_processingState[eyeIdx].m_cuStream));
    CUDA_CHECK(cuEventRecord(m_processingState[eyeIdx].m_frameSegmentationStartEvent, m_processingState[eyeIdx].m_cuStream));
    CUDA_CHECK(cuEventRecord(m_processingState[eyeIdx].m_framePostProcessingStartEvent, m_processingState[eyeIdx].m_cuStream));
    CUDA_CHECK(cuEventRecord(m_processingState[eyeIdx].m_frameProcessingEndEvent, m_processingState[eyeIdx].m_cuStream));
  }


  m_logger.reset(new InferLogger());
  m_inferRuntime.reset(nvinfer1::createInferRuntime(*m_logger));

  // Load segmentation engine
  {
    mmfile fp("eyetracking/models/segmentation.engine");
    m_segmentationEngine.reset(m_inferRuntime->deserializeCudaEngine(fp.data(), fp.size()));
    m_segInputTensorName = getSingleInputTensorName(m_segmentationEngine.get());
    m_segOutputTensorName = getSingleOutputTensorName(m_segmentationEngine.get());
  }

  // Load ROI engine
  {
    mmfile fp("eyetracking/models/roi.engine");
    m_roiEngine.reset(m_inferRuntime->deserializeCudaEngine(fp.data(), fp.size()));
    m_roiInputTensorName = getSingleInputTensorName(m_roiEngine.get());
    m_roiOutputTensorName = getSingleOutputTensorName(m_roiEngine.get());
  }

  // Create TensorRT execution contexts
  PER_EYE {
    m_processingState[eyeIdx].m_segmentationExec.reset(m_segmentationEngine->createExecutionContext());
    m_processingState[eyeIdx].m_roiExec.reset(m_roiEngine->createExecutionContext());
  }

  // Debug: Dump out information about the I/O bindings
  dumpIOBindings(m_segmentationEngine.get());
  printf("Segmentation Input strides: "); printDims(m_processingState[0].m_segmentationExec->getTensorStrides(m_segInputTensorName)); printf("\n");
  printf("Segmentation Output strides: "); printDims(m_processingState[0].m_segmentationExec->getTensorStrides(m_segOutputTensorName)); printf("\n");

  dumpIOBindings(m_roiEngine.get());
  printf("ROI Input strides: "); printDims(m_processingState[0].m_roiExec->getTensorStrides(m_roiInputTensorName)); printf("\n");
  printf("ROI Output strides: "); printDims(m_processingState[0].m_roiExec->getTensorStrides(m_roiOutputTensorName)); printf("\n");

  // Get the input size
  {
    // Segmentation
    nvinfer1::Dims inSize = m_processingState[0].m_segmentationExec->getTensorShape(m_segInputTensorName);
    assert(inSize.nbDims >= 2);
    m_segInputWidth = inSize.d[inSize.nbDims - 1];
    m_segInputHeight = inSize.d[inSize.nbDims - 2];
    assert(m_segInputWidth > 1 && m_segInputHeight > 1);
    printf("Segmentation image dimensions: %ux%u\n", m_segInputWidth, m_segInputHeight);

    // ROI
    inSize = m_processingState[0].m_roiExec->getTensorShape(m_roiInputTensorName);
    assert(inSize.nbDims >= 2);
    m_roiInputWidth = inSize.d[inSize.nbDims - 1];
    m_roiInputHeight = inSize.d[inSize.nbDims - 2];
    assert(m_roiInputWidth > 1 && m_roiInputHeight > 1);
    printf("ROI image dimensions: %ux%u\n", m_roiInputWidth, m_roiInputHeight);
  }

  // Get the output size
  {
    // Segmentation
    assert(m_segmentationEngine->getTensorDataType(m_segOutputTensorName) == nvinfer1::DataType::kHALF);
    nvinfer1::Dims outSize = m_processingState[0].m_segmentationExec->getTensorShape(m_segOutputTensorName);
    assert(outSize.nbDims >= 3);
    // Ensure output width and height match
    assert(m_segInputWidth == outSize.d[outSize.nbDims - 1]);
    assert(m_segInputHeight == outSize.d[outSize.nbDims - 2]);

    // Output should have 1 channel
    assert(1 == outSize.d[outSize.nbDims - 3]);

    nvinfer1::Dims strides = m_processingState[0].m_segmentationExec->getTensorStrides(m_segOutputTensorName);
    m_segOutputSizeBytes = strides.d[0] * (/*sizeof(fp16)=*/ 2);

    m_segOutputRowPitchElements = strides.d[strides.nbDims - 2];
    m_segOutputPlanePitchElements = strides.d[strides.nbDims - 3];

    printf("Segmentation output is %zu bytes. Row pitch is %zu elements, plane pitch is %zu elements\n", m_segOutputSizeBytes, m_segOutputRowPitchElements, m_segOutputPlanePitchElements);
  }

  {
    // ROI

    // Only support same input and output types
    auto roiIOType = m_roiEngine->getTensorDataType(m_roiInputTensorName);
    assert(m_roiEngine->getTensorDataType(m_roiOutputTensorName) == roiIOType);

    if (roiIOType == nvinfer1::DataType::kINT8) {
      m_roiIOIsInt8 = true;
    } else if (roiIOType == nvinfer1::DataType::kHALF) {
      m_roiIOIsInt8 = false;
    } else {
      assert(false && "ROI engine: unsupported I/O type (must be kHALF or kINT8)");
    }

    nvinfer1::Dims outSize = m_processingState[0].m_roiExec->getTensorShape(m_roiOutputTensorName);
    assert(outSize.nbDims >= 3);

    // Output W/H depends on network config
    m_roiOutputWidth = outSize.d[outSize.nbDims - 1];
    m_roiOutputHeight = outSize.d[outSize.nbDims - 2];
    // Output should have 1 channel
    assert(1 == outSize.d[outSize.nbDims - 3]);

    nvinfer1::Dims strides = m_processingState[0].m_roiExec->getTensorStrides(m_roiOutputTensorName);
    size_t roiOutputSizeElements = strides.d[0];

    m_roiOutputSizeBytes = roiOutputSizeElements * roiElementSize();
    printf("ROI Output is %ux%u, %zu bytes\n", m_roiOutputWidth, m_roiOutputHeight, m_roiOutputSizeBytes);
  }


  applyCalibrationData();

  // Set up per-eye output buffers and TensorRT dispatches
  // We cook the per-eye output buffer address into the TRT dispatch,
  // so each eye needs its own version of the exec graph.

  PER_EYE {
    ProcessingState& ps = m_processingState[eyeIdx];

    {
      // ROI network setup
      CUDA_CHECK(cuMemHostAlloc(&ps.m_roiInputTensorPtr, m_roiInputWidth * m_roiInputHeight * roiElementSize(), /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));
      CUDA_CHECK(cuMemHostAlloc(&ps.m_roiOutputTensorPtr, m_roiOutputSizeBytes, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));

      // Wire tensor I/O buffers into execution context
      assert(ps.m_roiExec->setTensorAddress(m_roiInputTensorName, (void*) ps.m_roiInputTensorPtr));
      assert(ps.m_roiExec->setTensorAddress(m_roiOutputTensorName, (void*) ps.m_roiOutputTensorPtr));

      // Enqueue one run to initialize internal data structures -- required before the graph recording
      // The inital run takes longer than subsequent ones, so we should pay that startup cost now
      // instead of during the frame loop.
      assert(ps.m_roiExec->enqueueV3(ps.m_cuStream));
    }

    {
      // Segmentation network setup

      // Segmentation mask postprocessing buffers
      CUDA_CHECK(cuMemHostAlloc((void**) &ps.m_pupilMask1, m_segInputWidth * m_segInputHeight, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));
      CUDA_CHECK(cuMemHostAlloc((void**) &ps.m_pupilMask2, m_segInputWidth * m_segInputHeight, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));

      // Segmentation network input buffer
      {
        nvinfer1::Dims strides = ps.m_segmentationExec->getTensorStrides(m_segInputTensorName);

        ps.m_segInputTensorStrideElements = strides.d[2];
        CUDA_CHECK(cuMemHostAlloc((void**) &m_processingState[eyeIdx].m_segInputTensorPtr, strides.d[0] * /*sizeof(fp16)=*/ 2, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));
      }

      // Segmentation network output allocation
      CUDA_CHECK(cuMemHostAlloc((void**) &m_processingState[eyeIdx].m_segOutputTensorPtr, m_segOutputSizeBytes, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));

      CUdeviceptr segOutputTensorDevicePtr;

      if (unifiedAddressingSupport) {
        segOutputTensorDevicePtr = (CUdeviceptr) m_processingState[eyeIdx].m_segOutputTensorPtr;
      } else {
        CUDA_CHECK(cuMemHostGetDevicePointer(&segOutputTensorDevicePtr, &m_processingState[eyeIdx].m_segOutputTensorPtr, /*flags=*/ 0));
      }

      // Wire tensor I/O buffers into execution context
      assert(ps.m_segmentationExec->setTensorAddress(m_segInputTensorName, (void*) m_processingState[eyeIdx].m_segInputTensorPtr));
      assert(ps.m_segmentationExec->setTensorAddress(m_segOutputTensorName, (void*) segOutputTensorDevicePtr));

      // Enqueue one run to initialize internal data structures
      // The inital run takes longer than subsequent ones, so we should pay that startup cost now
      // instead of during the frame loop.
      assert(ps.m_segmentationExec->enqueueV3(ps.m_cuStream));
    }
  } // PER_EYE
}

EyeTrackingService::~EyeTrackingService() {
  PER_EYE {
    // Shut down processing threads
    m_processingState[eyeIdx].m_processingThread.interrupt();
    m_processingState[eyeIdx].m_processingThread.join();

    // Ensure that the nvinfer1::IExecutionContext objects are destroyed before the nvinfer1::ICudaEngines that they're based on
    m_processingState[eyeIdx].m_segmentationExec.reset();
    m_processingState[eyeIdx].m_roiExec.reset();
  }
}

bool EyeTrackingService::loadCalibrationData() {

  cv::FileStorage fs(calibrationFilename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("Unable to open calibration data file %s\n", calibrationFilename);
    return false;
  }

  return loadCalibrationData(fs);
}

void EyeTrackingService::saveCalibrationData() {
  cv::FileStorage fs(calibrationFilename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  saveCalibrationData(fs);
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)
bool EyeTrackingService::loadCalibrationData(cv::FileStorage& fs) {
  try {
    readNode(fs, focalLength);
    readNode(fs, pixelPitchMicrons);
    readNode(fs, eyeZ);
    cv::read(fs["rollOffsetL"], m_rollOffsetDeg[0], m_rollOffsetDeg[0]);
    cv::read(fs["rollOffsetR"], m_rollOffsetDeg[1], m_rollOffsetDeg[1]);
    readNode(fs, filterMinCutoff);
    readNode(fs, filterBetaExponent);
    readNode(fs, filterDCutoff);

  } catch (const std::exception& ex) {
    printf("Unable to load calibration data: %s\n", ex.what());
    return false;
  }
  return true;
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void EyeTrackingService::saveCalibrationData(cv::FileStorage& fs) {
  writeNode(fs, focalLength);
  writeNode(fs, pixelPitchMicrons);
  writeNode(fs, eyeZ);
  fs.write("rollOffsetL", m_rollOffsetDeg[0]);
  fs.write("rollOffsetR", m_rollOffsetDeg[1]);
  writeNode(fs, filterMinCutoff);
  writeNode(fs, filterBetaExponent);
  writeNode(fs, filterDCutoff);
}
#undef writeNode


void EyeTrackingService::postprocessOneEye(size_t eyeIdx) {
  ProcessingState& ps = m_processingState[eyeIdx];

  bool fitEllipse = postprocessOneEye_fitEllipse(eyeIdx);

  // Update counters for calibration state machine
  if (fitEllipse) {
    ps.m_contiguousValidFrameCounter += 1;
    ps.m_contiguousInvalidFrameCounter = 0;
  } else {
    ps.m_contiguousValidFrameCounter = 0;
    ps.m_contiguousInvalidFrameCounter += 1;
  }

  switch (ps.m_calibrationState) {
    case kWaitingForValidFrames: {
      if (ps.m_contiguousValidFrameCounter > 25) {
        printf("EyeTrackingService::postprocessOneEye(%zu): waiting -> centering\n", eyeIdx);
        ps.m_calibrationSamples.clear();
        ps.m_calibrationState = kCentering;
      }
    } break;

    case kCentering: {
      // Add valid samples to the scrolling buffer.
      // We throw away the first couple samples after a run of invalid samples to reduce noise
      if (fitEllipse && ps.m_contiguousValidFrameCounter >= 3) {
        ps.m_calibrationSamples.push_back(ps.m_pupilEllipse);

        if (ps.m_calibrationSamples.full()) {
          // Collected enough samples, see if they're consistent enough
          glm::vec2 mean(0.0f, 0.0f);
          for (size_t i = 0; i < ps.m_calibrationSamples.size(); ++i) {
            mean.x += ps.m_calibrationSamples[i].center.x;
            mean.y += ps.m_calibrationSamples[i].center.y;
          }
          mean /= static_cast<float>(ps.m_calibrationSamples.size());

          float maxDeviation = 0.0f;

          float minDeviation = FLT_MAX;
          size_t minDeviationIdx = 0;

          for (size_t i = 0; i < ps.m_calibrationSamples.size(); ++i) {
            glm::vec2 p = glm::vec2(ps.m_calibrationSamples[i].center.x, ps.m_calibrationSamples[i].center.y);
            float lp = glm::length(p - mean);
            maxDeviation = std::max<float>(lp, maxDeviation);

            if (lp < minDeviation) {
              minDeviation = lp;
              minDeviationIdx = i;
            }
          }

          printf("Center calibration: %zu samples, mean = {%.3f, %.3f}, deviation range = [%.3f, %.3f]\n",
            ps.m_calibrationSamples.size(), mean.x, mean.y, minDeviation, maxDeviation);

          // TODO adjust threshold
          if (maxDeviation < 3.0f) {
            printf("Deviation below threshold, accepting calibration\n");
            ps.m_centerCalibrationSample = ps.m_calibrationSamples[minDeviationIdx];


            // Clear the eye fitter model
            ps.m_eyeModelFitter.reset();
            ps.m_eyeFitterSamples.clear();

            // Move to 'calibrated' state.
            // We still need to rebuild the eye fitter model, but that'll happen over time as the user looks around.
            ps.m_calibrationState = kCalibrated;

          }
        }
      }
    } break;

    case kCalibrated: {
      if (ps.m_contiguousInvalidFrameCounter > 50) {
        printf("EyeTrackingService::postprocessOneEye(%zu): Could not fit pupil for %u frames, resetting calibration.\n", eyeIdx, ps.m_contiguousInvalidFrameCounter);
        ps.m_calibrationState = kWaitingForValidFrames;
      }

    } break;

    default:
      printf("EyeTrackingService::postprocessOneEye(%zu): invalid calibration state %u\n", eyeIdx, ps.m_calibrationState);
      ps.m_calibrationState = kWaitingForValidFrames;
      break;
  };

}


// Returns true if an ellipse was fit this frame, even if the full eye model fit/unproject wasn't successful.
bool EyeTrackingService::postprocessOneEye_fitEllipse(size_t eyeIdx) {

  ProcessingState& ps = m_processingState[eyeIdx];
  cv::Mat pupilMask(m_segInputHeight, m_segInputWidth, CV_8UC1, ps.m_pupilMask1, /*step=*/ m_segInputWidth);

  // Find contours in the pupil mask
  struct Contour {
    std::vector<cv::Point> points;
    float area;
    float perimeter;
  };
  std::vector<Contour> filteredContours;


  // Collect stats and filter contours
  {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContoursLinkRuns(pupilMask, contours);


    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx) {
      auto& points = contours[contourIdx];
      if (points.size() < 20)
        continue; // Not enough points

      Contour c;
      c.area = fabs(cv::contourArea(points));
      if (c.area < 1000.0f)
        continue; // Not enough area

      c.perimeter = cv::arcLength(points, /*is_closed=*/ true);

      c.points = std::move(points);
      filteredContours.push_back(c);
    }

    // Sort filtered contours by area descending
    if (filteredContours.size() > 1)
      std::sort(filteredContours.begin(), filteredContours.end(), [](const Contour& left, const Contour& right) { return left.area > right.area; } );
  }

  bool didFitEllipse = false;
  std::vector<cv::Point2f> transformedContour;

  for (size_t contourIdx = 0; contourIdx < filteredContours.size(); ++contourIdx) {
    Contour& contour = filteredContours[contourIdx];
    FRAME_DEBUG_LOG("Contour %zu/%zu: area = %f, perimeter = %f, %zu points\n", contourIdx, filteredContours.size(), contour.area, contour.perimeter, contour.points.size());

    // Circularity
    // 75% circularity seems to be a good threshold for valid ellipses even at extreme angles
    float circularity = (4.0f * M_PI * contour.area) / (contour.perimeter * contour.perimeter);
    if (fabs(1.0f - circularity) > 0.25f) {
      FRAME_DEBUG_LOG(" -- Failed on circularity test (ratio %.3f)\n", circularity);
      continue;
    }

    // Convexity
    std::vector<cv::Point> hull;
    cv::convexHull(contour.points, hull);
    float convexHullArea = fabs(cv::contourArea(hull));
    if (convexHullArea < 1.0f) {
      FRAME_DEBUG_LOG(" -- Failed to fit a convex hull\n");
      continue;
    }

    float convexity = contour.area / convexHullArea;
    if (fabs(1.0f - convexity) > 0.10f) {
      FRAME_DEBUG_LOG(" -- Failed on convexity test (ratio %.3f)\n", convexity);
      continue;
    }

    // Tests passed


    // Find center of points for sector angle filtering
    glm::vec2 boundsCenter;
    {
      glm::vec2 boundsMin = toGlm(contour.points[0]);
      glm::vec2 boundsMax = toGlm(contour.points[0]);
      for (size_t i = 1; i < contour.points.size(); ++i) {
        const cv::Point2f& p = contour.points[i];
        boundsMin = glm::min(toGlm(p), boundsMin);
        boundsMax = glm::max(toGlm(p), boundsMax);
      }
      boundsCenter = (boundsMin + boundsMax) * 0.5f;
    }

    // Save bounds center for sector cutoff gizmo drawing
    ps.m_debugBoundsCenter = boundsCenter + glm::vec2(ps.m_lastSegROIToCaptureMatOffset.x, ps.m_lastSegROIToCaptureMatOffset.y);

    // Vertical vector and cutoff angle
    glm::vec2 verticalVec = vec2AtAngleDeg(m_rollOffsetDeg[eyeIdx] + 90.0f);
    float cosAngleCutoff = cosf(glm::radians(m_sectorCutoffAngleDeg));

    // Transform contour points from ROI into full image space prior to ellipse fitting.
    // (easier to transform points than the ellipse equation, given that the scale can be nonuniform)

    transformedContour.reserve(contour.points.size());
    for (size_t i = 0; i < contour.points.size(); ++i) {
      cv::Point2f np = contour.points[i];

      // Sector-angle cutoff filter
      float cosAngle = glm::abs(glm::dot(glm::normalize(toGlm(np) - boundsCenter), verticalVec));
      if (cosAngle > cosAngleCutoff)
        continue; // Skip point -- failed sector cutoff check

      // Add ROI offset
      np.x += static_cast<float>(ps.m_lastSegROIToCaptureMatOffset.x);
      np.y += static_cast<float>(ps.m_lastSegROIToCaptureMatOffset.y);

      transformedContour.push_back(np);
    }

    // Minimum of 6 points is required to fit an ellipse. Ensure that we have enough points after cutoff filtering
    if (transformedContour.size() < 6) {
      transformedContour.clear(); // discard failed points for next attempt
      continue;
    }

    ps.m_pupilEllipse = cv::fitEllipse(transformedContour);
    ps.m_debugTransformedContour = transformedContour;
    didFitEllipse = true;
    // Only need to fit one ellipse.
    break;
  }

  if (!didFitEllipse) {
    // Clear pupil ellipse, since we didn't find anything useful this round.
    ps.m_pupilEllipse = cv::RotatedRect();
    ps.m_eyeFitterOutputsValid = false;

    // No valid ellipse this frame.
    return false;
  }


  // See if we need to add more samples to the model
  if (ps.m_eyeModelFitter.pupils.size() < 50) {
    bool isNovelSample = true;
    if (ps.m_eyeModelFitter.pupils.size()) {
      float minDist = FLT_MAX;
      for (const auto& pupilSample : ps.m_eyeModelFitter.pupils) {
        cv::Point2f delta = cv::Point2f(
          ps.m_pupilEllipse.center.x - (pupilSample.observation.ellipse.centre[0] + ps.m_captureCenterOffset.x),
          ps.m_pupilEllipse.center.y - (pupilSample.observation.ellipse.centre[1] + ps.m_captureCenterOffset.y));
        float dist = sqrtf((delta.x * delta.x) + (delta.y * delta.y));
        minDist = std::min<float>(minDist, dist);
      }
      // FRAME_DEBUG_LOG("Min sample distance = %.3f\n", minDist);
      isNovelSample = (minDist > 3.0f);
    }

    if (isNovelSample) {
      // Add this observation to the eye model fitter
      ps.m_eyeFitterSamples.push_back(ps.m_pupilEllipse);

      // pupil_inliers needs to be in camera space, so apply captureCenterOffset
      std::vector<cv::Point2f> pupil_inliers;
      pupil_inliers.resize(transformedContour.size());
      for (size_t i = 0; i < pupil_inliers.size(); ++i) {
        pupil_inliers[i] = transformedContour[i] - ps.m_captureCenterOffset;
      }

      // The eyefitter works in a coordinate system where the center of the image is at (0, 0),
      // so we need to offset the ellipse center coordinate (via toEllipseWithOffset)

      ps.m_eyeModelFitter.add_observation(
        /*image (unused)=*/cv::Mat(),
        /*pupil=*/ singleeyefitter::toEllipseWithOffset<double>(ps.m_pupilEllipse, ps.m_captureCenterOffset),
        /*inliers=*/ pupil_inliers);

      // Try and fit the model
      if (ps.m_eyeModelFitter.pupils.size() > 20) {
        FRAME_DEBUG_LOG("Attempting eye model fit. ps.m_eyeFitterSamples.size()=%zu ps.m_eyeModelFitter.pupils.size()=%zu\n",
          ps.m_eyeFitterSamples.size(), ps.m_eyeModelFitter.pupils.size());

        if (ps.m_eyeModelFitter.unproject_observations(pupilRadius(), initialEyeZ())) {
          ps.m_eyeModelFitter.initialise_model();
          ps.m_eyeModelFitter.refine_with_inliers();


          // Use the center calibration sample to find angle offsets
          if (ps.m_eyeModelFitter.unproject_single_observation(ps.m_centerPupilCircle, singleeyefitter::toEllipseWithOffset<double>(ps.m_centerCalibrationSample, ps.m_captureCenterOffset), pupilRadius())) {
            glm::vec3 pupil = ps.centerPupilNormal();

            vectorToAngles(glm::value_ptr(pupil), ps.m_centerPitchDeg, ps.m_centerYawDeg, /*toDegrees=*/ true);

            printf("Center calibration sample pitch=%.3f yaw=%.3f (n=%.3f %.3f %.3f)\n",
              ps.m_centerPitchDeg, ps.m_centerYawDeg,
              pupil.x, pupil.y, pupil.z);
          } else {
            printf("Center calibration sample invalid!\n");
            // TODO: try and recover from this?
          }

        } else {
          FRAME_DEBUG_LOG("Eye model fit failed; unproject_observations() returned false.\n");
        }
      }
    }
  }

  // Apply model to ellipse to generate 3d fit
  if (ps.m_eyeModelFitter.hasEyeModel()) {
    ps.m_eyeFitterOutputsValid = ps.m_eyeModelFitter.unproject_single_observation(ps.m_fitPupilCircle, singleeyefitter::toEllipseWithOffset<double>(ps.m_pupilEllipse, ps.m_captureCenterOffset), pupilRadius());
    if (ps.m_eyeFitterOutputsValid) {
      // Original coordinate system:
      // +x is left
      // -y is up
      // -z is forward

      glm::vec3 pupil = ps.fitPupilNormal();
      vectorToAngles(glm::value_ptr(pupil), ps.m_pupilRawPitchDeg, ps.m_pupilRawYawDeg, /*toDegrees=*/ true);

      FRAME_DEBUG_LOG("n=%.3f %.3f %.3f\n", pupil.x, pupil.y, pupil.z);
      FRAME_DEBUG_LOG("pitch = %.3f, yaw = %.3f\n", ps.m_pupilRawPitchDeg, ps.m_pupilRawYawDeg);

      // Add sample to filter
      double ts = static_cast<double>(currentTimeNs() / 1000ULL) / 1'000'000.0;
      ps.m_pupilFilteredPitchDeg = ps.m_pitchFilter(ps.m_pupilRawPitchDeg, ts);
      ps.m_pupilFilteredYawDeg = ps.m_yawFilter(ps.m_pupilRawYawDeg, ts);
    }
  } else {
    ps.m_eyeFitterOutputsValid = false;
  }

  // Had a valid ellipse observation this frame
  return true;
}


void polyline(cv::Mat& img, const cv::Point2f* points, size_t startIdx, size_t endIdx, const cv::Scalar& color, bool loop = false) {
  for (size_t i = startIdx; i < endIdx; ++i) {
    cv::line(img, points[i], points[i + 1], color);
  }
  if (loop)
    cv::line(img, points[endIdx], points[startIdx], color);
}

void lineCenterDirectionLength(cv::Mat& img, const glm::vec2& center, const glm::vec2& direction, float length, const cv::Scalar& color, bool bidirectional = true) {
  glm::vec2 v1 = center + (direction * length);
  glm::vec2 v2 = bidirectional ? (center - (direction * length)) : center;
  cv::line(img, cv::Point(v1.x, v1.y), cv::Point(v2.x, v2.y), color);
}

cv::Point2f toImgCoord(const cv::Point2f& point, const cv::Point2f& centerOffset) {
  return point + centerOffset;
}
cv::Point toImgCoord(const cv::Point& point, const cv::Point2f& centerOffset) {
    return cv::Point(
        static_cast<int>(centerOffset.x) + point.x,
        static_cast<int>(centerOffset.y) + point.y);
}
cv::RotatedRect toImgCoord(const cv::RotatedRect& rect, const cv::Point2f& centerOffset) {
    return cv::RotatedRect(toImgCoord(rect.center, centerOffset),
        cv::Size2f(rect.size.width, rect.size.height), rect.angle);
}


void EyeTrackingService::eyeProcessingThreadFn(size_t eyeIdx) {
  ProcessingState& ps = m_processingState[eyeIdx];

  ps.m_processingThreadAlive = true;

  cv::Mat captureMat;

  cv::Mat rgbDebugMat;

  while (true) {
    if (boost::this_thread::interruption_requested())
      break;

    // Capture frame
    if (!ps.m_capture.readFrame()) {
      printf("EyeTrackingService::eyeProcessingThreadFn(%zu)::captureWorkerThread: readFrame() returned false, terminating\n", eyeIdx);
      break;
    }

    ps.m_capture.lumaPlane().copyTo(captureMat);

    if (captureMat.empty()) {
      printf("EyeTrackingService::eyeProcessingThreadFn(%zu)::captureWorkerThread: mat is empty, terminating\n", eyeIdx);
      break;
    }

    ps.m_lastCaptureTimestampNs = currentTimeNs();

    // Update the capture center offset, now that we know the frame dimensions.
    ps.m_captureCenterOffset = cv::Point2f(
      static_cast<float>(captureMat.cols) / 2.0f,
      static_cast<float>(captureMat.rows) / 2.0f);

    // Scale from capture mat to the input size for the ROI prediction network
    cv::resize(captureMat, ps.m_roiScaleMat, cv::Size(m_roiInputWidth, m_roiInputHeight));

    // Convert CV_U8 from the scale output to int8 or half for the ROI network input

    assert(ps.m_roiScaleMat.isContinuous()); // Conversion assumes continuous input and output mats
    if (m_roiIOIsInt8) {
      convertUnorm8ToDLAInt8(ps.m_roiScaleMat.ptr<uint8_t>(0), ps.m_roiInputTensorPtr, m_roiInputWidth * m_roiInputHeight);
    } else {
      convertUnorm8ToSnormFp16(ps.m_roiScaleMat.ptr<uint8_t>(0), ps.m_roiInputTensorPtr, m_roiInputWidth * m_roiInputHeight);
    }

    // Run ROI network
    CUDA_CHECK(cuEventRecord(ps.m_frameProcessingStartEvent, ps.m_cuStream));

    assert(ps.m_roiExec->enqueueV3(ps.m_cuStream));

    CUDA_CHECK(cuEventRecord(ps.m_frameROIEndEvent, ps.m_cuStream));

    // Wait for ROI to finish processing
    CUDA_CHECK(cuStreamSynchronize(ps.m_cuStream));


    // Compute center of ROI heatmap
    float roiOutput[2]; // 0...1 coordinate range

    if (m_roiIOIsInt8) {
      assert(false && "TODO: ROI heatmap center computation for int8 i/o");
    } else {
      _Float16* roiBasePtr = reinterpret_cast<_Float16*>(ps.m_roiOutputTensorPtr);

      // Weighted sampling:

      // Find the max value. TODO: This can be vectorized
      _Float16 maxValue = 0.0f;
      for (uint32_t y = 0; y < m_roiOutputHeight; ++y) {
        _Float16* roiRowPtr = roiBasePtr + (y * m_roiOutputWidth);
        for (uint32_t x = 0; x < m_roiOutputWidth; ++x) {
          maxValue = std::max<_Float16>(roiRowPtr[x], maxValue);
        }
      }

      // Gather all sample points >= 0.85x max value and compute their average position.
      _Float16 threshold = maxValue * 0.85f16;

      uint32_t sampleCount = 0;
      uint32_t xAccum = 0;
      uint32_t yAccum = 0;

      for (uint32_t y = 0; y < m_roiOutputHeight; ++y) {
        _Float16* roiRowPtr = roiBasePtr + (y * m_roiOutputWidth);
        for (uint32_t x = 0; x < m_roiOutputWidth; ++x) {
          if (roiRowPtr[x] >= threshold) {
            sampleCount += 1;
            xAccum += x;
            yAccum += y;
          }
        }
      }

      if (sampleCount > 0) {
        roiOutput[0] = (static_cast<float>(xAccum) / static_cast<float>(sampleCount)) / static_cast<float>(m_roiOutputWidth);
        roiOutput[1] = (static_cast<float>(yAccum) / static_cast<float>(sampleCount)) / static_cast<float>(m_roiOutputHeight);
      } else {
        // No samples? Default to center of image.
        roiOutput[0] = 0.5f;
        roiOutput[1] = 0.5f;
      }
    }

    // Use center coordinates computed from the ROI network to compute the ROI aligned inside the original capture mat
    // ROI Rect needs to be fixed size of the segmentation network input (no edge clipping allowed)

    FRAME_DEBUG_LOG("ROI computed center (0...1): (%f, %f)\n", roiOutput[0], roiOutput[1]);

    // Rescale to 0...1f and multiply by the actual source w/h
    cv::Point2i roiCenter_captureRelative = cv::Point2i(
      clamp<int32_t>(roiOutput[0] * static_cast<float>(captureMat.cols - 1), 0, captureMat.cols - 1),
      clamp<int32_t>(roiOutput[1] * static_cast<float>(captureMat.rows - 1), 0, captureMat.rows - 1)
    );

    // Clip the capture-relative ROI center to the capture dimensions inset by half of the segmentation network input size. This should ensure that the
    // segmentation ROI rect fits entirely within the capture region.

    roiCenter_captureRelative.x = clamp<int32_t>(roiCenter_captureRelative.x, (m_segInputWidth / 2), (captureMat.cols - 1) - (m_segInputWidth / 2));
    roiCenter_captureRelative.y = clamp<int32_t>(roiCenter_captureRelative.y, (m_segInputHeight / 2), (captureMat.rows - 1) - (m_segInputHeight / 2));

    // Build segmentation input crop rect in capture mat coordinates
    cv::Point2i segROIRect_tl = (roiCenter_captureRelative - cv::Point2i(m_segInputWidth / 2, m_segInputHeight / 2));
    // Just in case, clamp the top left corner so it doesn't go negative
    segROIRect_tl.x = std::max<int32_t>(segROIRect_tl.x, 0);
    segROIRect_tl.y = std::max<int32_t>(segROIRect_tl.y, 0);

    cv::Rect segROIRect = cv::Rect(segROIRect_tl, cv::Size(m_segInputWidth, m_segInputHeight));
    ps.m_lastSegROIToCaptureMatOffset = segROIRect_tl; // save for eyefitter processing

    cv::Mat segROIMat = cv::Mat(captureMat, segROIRect);

    // Convert ROI window to fp16 to populate ps.m_segInputTensor
    // The ROI input is known not to be contiguous, so we do it row-by-row.
    for (size_t y = 0; y < m_segInputHeight; ++y) {
      convertUnorm8ToSnormFp16(segROIMat.ptr<uint8_t>(y, 0), ps.m_segInputTensorPtr + (y * ps.m_segInputTensorStrideElements), m_segInputWidth);
    }

    // Launch TRT processing
    CUDA_CHECK(cuEventRecord(ps.m_frameSegmentationStartEvent, ps.m_cuStream));

    assert(ps.m_segmentationExec->enqueueV3(ps.m_cuStream));

    CUDA_CHECK(cuEventRecord(ps.m_framePostProcessingStartEvent, ps.m_cuStream));

    // Run CUDA postprocessing operations:
    {
      NppiSize sz;
      sz.width = m_segInputWidth;
      sz.height = m_segInputHeight;

      NppiPoint zero;
      zero.x = 0;
      zero.y = 0;

      // Run threshold over the return from the segmentation network to find the pupil mask
      union {
        _Float16 fval;
        int16_t ival;
      } a;
      a.fval = 0.5f;

      int16_t threshold = a.ival;

      // NppStatus nppiCompareC_8u_C1R_Ctx(
      //   const Npp8u *pSrc, int nSrcStep,
      //   const Npp8u nConstant,
      //   Npp8u *pDst, int nDstStep,
      //   NppiSize oSizeROI,
      //   NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
      NPP_CHECK(nppiCompareC_16s_C1R_Ctx(
        reinterpret_cast<const Npp16s*>(ps.m_segOutputTensorPtr), m_segInputWidth * sizeof(uint16_t),
        threshold,
        ps.m_pupilMask1, m_segInputWidth,
        sz,
        NPP_CMP_GREATER_EQ, ps.m_nppContext));

      // Run closure operation to fill holes in the mask -- 3x3 dilation followed by 3x3 erosion

      // NppStatus nppiDilate3x3Border_8u_C1R_Ctx(
      //   const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
      //         Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
      //         NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
      //
      NPP_CHECK(nppiDilate3x3Border_8u_C1R_Ctx(
        ps.m_pupilMask1, m_segInputWidth, sz, zero,
        ps.m_pupilMask2, m_segInputWidth, sz,
        NPP_BORDER_REPLICATE, ps.m_nppContext));

      // NppStatus nppiErode3x3Border_8u_C1R_Ctx(
      //   const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
      //         Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
      //         NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
      NPP_CHECK(nppiErode3x3Border_8u_C1R_Ctx(
        ps.m_pupilMask2, m_segInputWidth, sz, zero,
        ps.m_pupilMask1, m_segInputWidth, sz,
        NPP_BORDER_REPLICATE, ps.m_nppContext));
    }

    CUDA_CHECK(cuEventRecord(ps.m_frameProcessingEndEvent, ps.m_cuStream));


    // Wait for processing to finish
    CUDA_CHECK(cuStreamSynchronize(ps.m_cuStream));

    // Update stats
    cuEventElapsedTime(&ps.m_lastFrameROITimeMs, ps.m_frameProcessingStartEvent, ps.m_frameROIEndEvent);
    cuEventElapsedTime(&ps.m_lastFrameSegmentationTimeMs, ps.m_frameSegmentationStartEvent, ps.m_framePostProcessingStartEvent);
    cuEventElapsedTime(&ps.m_lastFrameROIToSegmentationLatencyMs, ps.m_frameROIEndEvent, ps.m_frameSegmentationStartEvent);

    cuEventElapsedTime(&ps.m_lastFrameTotalInferenceLatencyMs, ps.m_frameProcessingStartEvent, ps.m_framePostProcessingStartEvent);
    cuEventElapsedTime(&ps.m_lastFramePostProcessingTimeMs, ps.m_framePostProcessingStartEvent, ps.m_frameProcessingEndEvent);

    uint64_t postStartTimeNs = currentTimeNs();

    postprocessOneEye(eyeIdx);

    float postTimeMs = deltaTimeMs(postStartTimeNs, currentTimeNs());
    if (postTimeMs > 0.25f) {
      FRAME_DEBUG_LOG("Eye %zu CPU postprocess took %.3fms\n", eyeIdx, postTimeMs);
    }


    // Update debug view

    // Promote full capture buffer to RGB
    cv::cvtColor(/*src=*/ captureMat, /*dst=*/ rgbDebugMat, cv::COLOR_GRAY2BGR);
    if (m_debugDrawOverlays) {

      // Segmentation ROI view of the RGB debug mat
      cv::Mat debugROIViewRGB = cv::Mat(rgbDebugMat, segROIRect);

#if 0
      // Draw segmentation mask colors
      for (size_t row = 0; row < debugROIViewRGB.rows; ++row) {
        uint8_t* pupilRowPtr = ps.m_pupilMask1 + (row * m_segInputWidth);
        for (size_t col = 0; col < debugROIViewRGB.cols; ++col) {
          if (pupilRowPtr[col])
            debugROIViewRGB.ptr<uint8_t>(row, col)[/*red channel=*/2] = 0xcc;
        }
      }
#endif

      // Now operating on the full view; coordinates that are relative to the ROI will need to be translated.

#if 0
      // Draw eye-fitter sample ellipses
      for (const auto& el : ps.m_eyeFitterSamples) {
        cv::ellipse(rgbDebugMat, el, cv::Scalar(0x3f, 0, 0x3f), /*thickness=*/ 2);
      }
#endif

      // Draw pupil ellipse, if present
      if (!ps.m_pupilEllipse.size.empty()) {
        cv::ellipse(rgbDebugMat, ps.m_pupilEllipse, cv::Scalar(0xff, 0, 0xff), /*thickness=*/ 2);

        // Draw sector gizmo
        glm::vec2 verticalVec = vec2AtAngleDeg(m_rollOffsetDeg[eyeIdx] + 90.0f);
        glm::vec2 sector1Vec = vec2AtAngleDeg((m_rollOffsetDeg[eyeIdx] + 90.0f) + m_sectorCutoffAngleDeg);
        glm::vec2 sector2Vec = vec2AtAngleDeg((m_rollOffsetDeg[eyeIdx] + 90.0f) - m_sectorCutoffAngleDeg);

        lineCenterDirectionLength(rgbDebugMat, ps.m_debugBoundsCenter, verticalVec, 80.0f, cv::Scalar(255, 0, 0), /*bidirectional=*/ true);
        lineCenterDirectionLength(rgbDebugMat, ps.m_debugBoundsCenter, sector1Vec,  80.0f, cv::Scalar(0, 0, 255), /*bidirectional=*/ true);
        lineCenterDirectionLength(rgbDebugMat, ps.m_debugBoundsCenter, sector2Vec,  80.0f, cv::Scalar(0, 0, 255), /*bidirectional=*/ true);
      }

      if (ps.m_eyeFitterOutputsValid) {
        // Try-catch block avoids cv drawing functions crashing the app if we pass NaNs or something
        try {
          singleeyefitter::Conic<double> pupil_conic = singleeyefitter::project(ps.m_fitPupilCircle, ps.m_eyeModelFitter.focal_length);
          singleeyefitter::Ellipse2D<double> eye_ellipse = singleeyefitter::project(ps.m_eyeModelFitter.eye, ps.m_eyeModelFitter.focal_length);

          cv::RotatedRect pupilEllipseImg = toImgCoord(toRotatedRect(singleeyefitter::Ellipse2D<double>(pupil_conic)), ps.m_captureCenterOffset);
          // pupil ellipse was already drawn above
          //cv::ellipse(rgbDebugMat, pupilEllipseImg, cv::Scalar(60, 60, 0), /*thickness=*/ 2);

          cv::RotatedRect eyeEllipseImg = toImgCoord(toRotatedRect(eye_ellipse), ps.m_captureCenterOffset);
          cv::ellipse(rgbDebugMat, eyeEllipseImg, cv::Scalar(0, 60, 60), /*thickness=*/ 2);

          // order is _bottomLeft_, _topLeft_, topRight, bottomRight
          cv::Point2f rectPoints[4];
          eyeEllipseImg.points(rectPoints);

          // Draw crosshairs through the eye-ellipse
          cv::line(rgbDebugMat,
            (rectPoints[0] + rectPoints[1]) * 0.5f,
            (rectPoints[2] + rectPoints[3]) * 0.5f,
            cv::Scalar(0, 60, 60), /*thickness=*/2);

          cv::line(rgbDebugMat,
            (rectPoints[1] + rectPoints[2]) * 0.5f,
            (rectPoints[0] + rectPoints[3]) * 0.5f,
            cv::Scalar(0, 60, 60), /*thickness=*/2);

          // Draw a small marker on the eye center point
          cv::circle(rgbDebugMat, eyeEllipseImg.center, /*r=*/ 3, cv::Scalar(255, 0, 0), /*thickness=*/ -1);

          // Line from the eye center through the pupil center
          cv::line(rgbDebugMat, eyeEllipseImg.center, pupilEllipseImg.center, cv::Scalar(0, 255, 0), /*thickness=*/ 1);

        } catch (...) {}

        char buf[64];
        snprintf(buf, 64, "n=%.3f %.3f %.3f", ps.m_fitPupilCircle.normal[0], ps.m_fitPupilCircle.normal[1], ps.m_fitPupilCircle.normal[2]);

        cv::putText(rgbDebugMat, buf, cv::Point2f(/*x=*/ 5, /*y=*/ rgbDebugMat.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

        //FRAME_DEBUG_LOG("Ellipse: center=%.3f %.3f\n width=%.3f height=%.3f\n",
        //    ps.m_pupilEllipse.center.x, ps.m_pupilEllipse.center.y,
        //    ps.m_pupilEllipse.size.width, ps.m_pupilEllipse.size.height);

      }

      // Draw the ROI centroid
      cv::circle(rgbDebugMat, roiCenter_captureRelative, /*r=*/ 3, cv::Scalar(192, 0, 192));

      // Draw the ROI rectangle
      cv::rectangle(rgbDebugMat, segROIRect.tl(), segROIRect.br(), cv::Scalar(0, 255, 0));

    } // Debug overlay drawing

    // Swap debug mat with the one in processing state
    // (tries to avoid the main thread getting a partially-drawn debug mat)
    cv::swap(ps.m_debugViewRGB, rgbDebugMat);

  } // Frame loop

  ps.m_processingThreadAlive = false;
}


bool EyeTrackingService::processFrame() {
  FRAME_DEBUG_LOG("\x1b[2J\x1b[H"); // Clear terminal per frame

  // Processing thread maintenance:
  // Start threads if they're not running, clean up after exited threads

  PER_EYE {
    ProcessingState& ps = m_processingState[eyeIdx];

    if (!ps.m_processingThreadAlive) {
      if (!ps.m_inputFilename.empty()) {
        // Try re-starting processing, ratelimited to once a second
        if (deltaTimeMs(ps.m_lastCaptureOpenAttemptTimeNs, currentTimeNs()) > 1000.0f) {
          ps.m_lastCaptureOpenAttemptTimeNs = currentTimeNs();
          if (ps.m_capture.tryOpenSensor(ps.m_inputFilename.c_str())) {
            // Capture is open, restart the processing thread
            ps.m_processingThread = boost::thread(boost::bind(&EyeTrackingService::eyeProcessingThreadFn, this, eyeIdx));
            printf("EyeTrackingService: Successfully opened capture of \"%s\" for eye %zu\n", ps.m_inputFilename.c_str(), eyeIdx);
          }
        }
      }
    }
  }

  if (!(m_processingState[0].m_processingThreadAlive || m_processingState[1].m_processingThreadAlive)) {
    // printf("EyeTrackingService: no processing threads alive\n");
    return false;
  }


  return true;
}

cv::Mat& EyeTrackingService::getDebugViewForEye(size_t eyeIdx, bool withDebugOverlay) {
  assert(eyeIdx < 2);
  return m_processingState[eyeIdx].m_debugViewRGB;
}

void EyeTrackingService::applyCalibrationData() {
  PER_EYE {
    ProcessingState& ps = m_processingState[eyeIdx];

    // Experimentally, roughly 1800.0 is a good ps.m_eyeModelFitter.focal_length value for the OV9281 sensor in 640x480 mode with 60 degree lens.
    ps.m_eyeModelFitter.focal_length = sefFocalLength();

    ps.m_pitchFilter.mincutoff = m_filterMinCutoff;
    ps.m_pitchFilter.beta = powf(10.0f, m_filterBetaExponent);
    ps.m_pitchFilter.dcutoff = m_filterDCutoff;

    ps.m_yawFilter.mincutoff = m_filterMinCutoff;
    ps.m_yawFilter.beta = powf(10.0f, m_filterBetaExponent);
    ps.m_yawFilter.dcutoff = m_filterDCutoff;
  }
}

void EyeTrackingService::renderIMGUI() {
  ImGui::PushID(this);

  bool dirty = false;


  // Roll angles don't require recalibrating
  ImGui::DragFloat("L Roll angle (deg)", &m_rollOffsetDeg[0], /*speed=*/ 0.1f, /*min=*/ -30.0f, /*max=*/ 30.0f, "%.1f");
  ImGui::DragFloat("R Roll angle (deg)", &m_rollOffsetDeg[1], /*speed=*/ 0.1f, /*min=*/ -30.0f, /*max=*/ 30.0f, "%.1f");

  ImGui::Separator();
  ImGui::DragFloat("Sector cutoff angle (deg)", &m_sectorCutoffAngleDeg, /*speed=*/ 1.0f, /*min=*/ 0.0f, /*max=*/ 90.0f, "%.1f");

  // Filter settings also apply immediately, but still require calling applyCalibrationData
  ImGui::Separator();
  dirty |= ImGui::DragFloat("Filter min cutoff", &m_filterMinCutoff, /*speed=*/ 0.01f, /*min=*/ 0.0f, /*max=*/ 2.0f, "%.2f");
  dirty |= ImGui::DragFloat("Filter D cutoff", &m_filterDCutoff, /*speed=*/ 0.01f, /*min=*/ 0.0f, /*max=*/ 2.0f, "%.2f");
  dirty |= ImGui::DragFloat("Filter beta exponent", &m_filterBetaExponent, /*speed=*/ 0.1f, /*min=*/ -10.0f, /*max=*/ 10.0f, "%.1f");

  // All of these settings require recalibrating the model
  ImGui::Separator();
  dirty |= ImGui::DragFloat("Focal Length", &m_focalLength, /*speed=*/ 0.1, /*min=*/ 1.0f, /*max=*/ 20.0f, "%.1f");
  dirty |= ImGui::DragFloat("Distance to eye (mm)", &m_eyeZ, /*speed=*/ 0.5f, /*min=*/ 1.0f, /*max=*/ 50.0f, "%.1f");
  dirty |= ImGui::DragFloat("Sensor pixel pitch (um)", &m_pixelPitchMicrons, /*speed=*/ 0.1f, /*min=*/ 0.1f, /*max=*/ 10.0f, "%.1f");

  if (ImGui::Button("Save Settings")) {
    saveCalibrationData();
  }

  if (dirty) {
    applyCalibrationData();
  }


  ImGui::PopID();
}

glm::vec2 EyeTrackingService::getPitchYawAnglesForEye(size_t eyeIdx) {
  assert(eyeIdx == 0 || eyeIdx == 1);

  glm::vec2 angles = glm::vec2(
    m_processingState[eyeIdx].m_pupilFilteredPitchDeg - m_processingState[eyeIdx].m_centerPitchDeg,
    m_processingState[eyeIdx].m_pupilFilteredYawDeg - m_processingState[eyeIdx].m_centerYawDeg);

  // Apply roll correction
  glm::vec3 rollCorrectionVector;
  anglesToVector<float>(glm::radians(m_rollOffsetDeg[eyeIdx]), glm::radians(angles[0]), glm::radians(angles[1]), glm::value_ptr(rollCorrectionVector));
  vectorToAngles<float>(glm::value_ptr(rollCorrectionVector), angles[0], angles[1], /*toDegrees=*/ true);
  return angles;
}

