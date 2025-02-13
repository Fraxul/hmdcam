#include "EyeTrackingService.h"
#include "eyeProcessing.h"
#include "common/Timing.h"
#include "common/mmfile.h"
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

#include <NvInfer.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>

// CUDA graph can only be used if the model doesn't run on the DLA
#define USE_CUDA_GRAPH 0

const char* kInputImageTensorName = "input";
const char* kOutputTensorName = "output";

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


EyeTrackingService::EyeTrackingService() {

  // Figure out the CUDA stream priority range and create a low-priority stream,
  // then create an NPP stream context associated with that stream
  {
    int leastPriority = 0, greatestPriority = 0;
    CUDA_CHECK(cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority));
    printf("CUDA stream priority range: %d least -> %d greatest\n", leastPriority, greatestPriority);
    CUDA_CHECK(cuStreamCreateWithPriority(&m_cuStream, CU_STREAM_NON_BLOCKING, leastPriority));

    // Create NPP stream context for our stream
    CUstream prevNppStream = nppGetStream();
    NPP_CHECK(nppSetStream(m_cuStream));
    NPP_CHECK(nppGetStreamContext(&m_nppContext));
    nppSetStream(prevNppStream);

    // Wrap stream object for OpenCV funcs
    m_cvStream = cv::cuda::wrapStream((size_t) m_cuStream);
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

  // Profiling events
  CUDA_CHECK(cuEventCreate(&m_frameProcessingStartEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&m_framePostProcessingStartEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&m_frameProcessingEndEvent, CU_EVENT_BLOCKING_SYNC));

  // Record good initial state for events
  CUDA_CHECK(cuEventRecord(m_frameProcessingStartEvent, m_cuStream));
  CUDA_CHECK(cuEventRecord(m_framePostProcessingStartEvent, m_cuStream));
  CUDA_CHECK(cuEventRecord(m_frameProcessingEndEvent, m_cuStream));


  m_logger = new InferLogger();
  m_inferRuntime = nvinfer1::createInferRuntime(*m_logger);

  mmfile fp("eyetracking/models/model-dla.engine");
  m_inferEngine = m_inferRuntime->deserializeCudaEngine(fp.data(), fp.size());

  // Create TensorRT execution contexts
  PER_EYE {
    m_processingState[eyeIdx].m_exec = m_inferEngine->createExecutionContext();
  }


  // Debug: Dump out information about the I/O bindings
  for (int32_t ioTensorIdx = 0; ioTensorIdx < m_inferEngine->getNbIOTensors(); ++ioTensorIdx) {
    const char* dataTypes[] = {"fp32", "fp16", "int8", "int32", "bool8", "uint8"};

    const char* n = m_inferEngine->getIOTensorName(ioTensorIdx);
    bool input = m_inferEngine->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT;

    nvinfer1::Dims d = m_inferEngine->getTensorShape(n);

    nvinfer1::DataType dt = m_inferEngine->getTensorDataType(n);
    nvinfer1::TensorLocation loc = m_inferEngine->getTensorLocation(n);

    printf("[%d] %s %s Loc=%s Dims=",
      ioTensorIdx, n, input ? "(input)" : "(output)",
      loc == nvinfer1::TensorLocation::kDEVICE ? "(device)" : "(host)");
    printDims(d);

    printf(" DataType=%s Format=%u (%s) Strides=", dataTypes[(int) dt], (uint32_t) m_inferEngine->getTensorFormat(n), m_inferEngine->getTensorFormatDesc(n));
    nvinfer1::Dims strides = m_processingState[0].m_exec->getTensorStrides(n);
    printDims(strides);

    int32_t vDim = m_inferEngine->getTensorVectorizedDim(n);
    if (vDim >= 0) {
      printf(" VectorizedDim=%d", vDim);
      printf(" Bytes/Component=%d Components/Element=%d", m_inferEngine->getTensorBytesPerComponent(n), m_inferEngine->getTensorComponentsPerElement(n));
    }

    printf("\n");
  }


  // Setup buffers
  printf("Input strides: "); printDims(m_processingState[0].m_exec->getTensorStrides(kInputImageTensorName)); printf("\n");
  printf("Output strides: "); printDims(m_processingState[0].m_exec->getTensorStrides(kOutputTensorName)); printf("\n");

  // Get the input size
  {
    nvinfer1::Dims inSize = m_processingState[0].m_exec->getTensorShape(kInputImageTensorName);
    assert(inSize.nbDims >= 2);
    m_trtInputWidth = inSize.d[inSize.nbDims - 1];
    m_trtInputHeight = inSize.d[inSize.nbDims - 2];
    assert(m_trtInputWidth > 1 && m_trtInputHeight > 1);
    printf("Input image dimensions: %ux%u\n", m_trtInputWidth, m_trtInputHeight);
  }

  // Get the output size
  {
    assert(m_inferEngine->getTensorDataType(kOutputTensorName) == nvinfer1::DataType::kHALF);
    nvinfer1::Dims outSize = m_processingState[0].m_exec->getTensorShape(kOutputTensorName);
    assert(outSize.nbDims >= 3);
    // Ensure output width and height match
    assert(m_trtInputWidth == outSize.d[outSize.nbDims - 1]);
    assert(m_trtInputHeight == outSize.d[outSize.nbDims - 2]);
    // Output should have 4 channels
    assert(4 == outSize.d[outSize.nbDims - 3]);

    nvinfer1::Dims strides = m_processingState[0].m_exec->getTensorStrides(kOutputTensorName);
    m_trtOutputSizeBytes = strides.d[0] * (/*sizeof(fp16)=*/ 2);

    m_trtOutputRowPitchElements = strides.d[strides.nbDims - 2];
    m_trtOutputPlanePitchElements = strides.d[strides.nbDims - 3];

    printf("Output is %zu bytes. Row pitch is %zu elements, plane pitch is %zu elements\n", m_trtOutputSizeBytes, m_trtOutputRowPitchElements, m_trtOutputPlanePitchElements);
  }

  // Set up per-eye output buffers and TensorRT dispatches
  // We cook the per-eye output buffer address into the TRT dispatch,
  // so each eye needs its own version of the exec graph.

  PER_EYE {
    ProcessingState& ps = m_processingState[eyeIdx];

    // Init eye-fitter
    // Focal length computation is taken from the SingleEyeFitter sample tool
    double fov = 45.0; // TODO parameterize
    double fov_radians = fov * (M_PI / 180.0);
    ps.m_eyeModelFitter.focal_length = static_cast<double>(m_trtInputWidth / 2) / std::tan(fov_radians / 2.0);

    // Pre-create m_preHistEq and m_postHistEq to be the same size/dimensions as the TRT input mat
    // (histogram equalization happens after warp/resize, right before TRT processing)
    ps.m_preHistEqMat.create(m_trtInputHeight, m_trtInputWidth, CV_8U);
    ps.m_postHistEqMat.create(m_trtInputHeight, m_trtInputWidth, CV_8U);

    // CLAHE processor, which has some internal allocations
    ps.m_clahe = cv::cuda::createCLAHE(/*clipLimit=*/ 1.5, /*tileGridSize=*/ cv::Size(8, 8));

    // Postprocessing buffers
    CUDA_CHECK(cuMemHostAlloc((void**) &ps.m_classIndex, m_trtInputWidth * m_trtInputHeight, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostAlloc((void**) &ps.m_pupilMask1, m_trtInputWidth * m_trtInputHeight, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostAlloc((void**) &ps.m_pupilMask2, m_trtInputWidth * m_trtInputHeight, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));

    // CV GpuMat for the TensorRT input
    // This needs to be manually allocated to match the stride requirements set by TensorRT.
    {
      nvinfer1::Dims strides = ps.m_exec->getTensorStrides(kInputImageTensorName);

      size_t inputStrideBytes = strides.d[2] * sizeof(_Float16);
      CUDA_CHECK(cuMemAlloc(&m_processingState[eyeIdx].m_trtInputMatPtr, strides.d[0]));
      m_processingState[eyeIdx].m_trtInputMat = cv::cuda::GpuMat(m_trtInputHeight, m_trtInputWidth, CV_16U, (void*) m_processingState[eyeIdx].m_trtInputMatPtr, inputStrideBytes);
      //printf("trtInputMat: step=%zu step1=%zu\n", m_processingState[eyeIdx].m_trtInputMat.step, m_processingState[eyeIdx].m_trtInputMat.step1());
    }

    // TensorRT output allocation
    CUDA_CHECK(cuMemHostAlloc((void**) &m_processingState[eyeIdx].m_trtOutputHostPtr, m_trtOutputSizeBytes, /*flags=*/ CU_MEMHOSTALLOC_DEVICEMAP));

    CUdeviceptr trtOutputDevicePtr;

    if (unifiedAddressingSupport) {
      trtOutputDevicePtr = (CUdeviceptr) m_processingState[eyeIdx].m_trtOutputHostPtr;
    } else {
      CUDA_CHECK(cuMemHostGetDevicePointer(&trtOutputDevicePtr, &m_processingState[eyeIdx].m_trtOutputHostPtr, /*flags=*/ 0));
    }

    // Wire tensor I/O buffers into execution context
    assert(ps.m_exec->setTensorAddress(kInputImageTensorName, m_processingState[eyeIdx].m_trtInputMat.cudaPtr()));
    assert(ps.m_exec->setTensorAddress(kOutputTensorName, (void*) trtOutputDevicePtr));

    // Enqueue one run to initialize internal data structures -- required before the graph recording
    // The inital run takes longer than subsequent ones, so we should pay that startup cost now
    // instead of during the frame loop.
    assert(ps.m_exec->enqueueV3(m_cuStream));

#if USE_CUDA_GRAPH
    // Compile the TensorRT dispatch into a CUDA graph
    // Begin recording graph
    CUDA_CHECK(cuStreamBeginCapture(m_cuStream, CU_STREAM_CAPTURE_MODE_GLOBAL));

    // Record TensorRT dispatch
    assert(ps.m_exec->enqueueV3(m_cuStream));

    // End recording and instantiate executable graph
    CUDA_CHECK(cuStreamEndCapture(m_cuStream, &m_processingState[eyeIdx].m_frameProcessingGraph));
    CUDA_CHECK(cuGraphInstantiateWithFlags(&m_processingState[eyeIdx].m_frameProcessingGraphExec, m_processingState[eyeIdx].m_frameProcessingGraph, /*flags=*/ 0));
    CUDA_CHECK(cuGraphUpload(m_processingState[eyeIdx].m_frameProcessingGraphExec, m_cuStream));
#endif
  } // PER_EYE

  // Create the u8 -> fp16 LUT for edgaze input processing
  // The LUT applies a gamma curve of 0.8, then remaps to -1...1 range in fp16.
  _Float16 lut[256];
  for (size_t lutIdx = 0; lutIdx < 256; ++lutIdx) {
	  lut[lutIdx] = static_cast<_Float16>((pow(static_cast<float>(lutIdx) / 255.0f, 0.8f) * 2.0f) - 1.0f);
	}
  CUDA_CHECK(cuMemAlloc(&m_inputLUT, 256 * sizeof(_Float16)));
  CUDA_CHECK(cuMemcpyHtoD(m_inputLUT, lut, 256 * sizeof(_Float16)));
}

EyeTrackingService::~EyeTrackingService() {
  cuStreamSynchronize(m_cuStream);

  PER_EYE {
    m_processingState[eyeIdx].releaseResources();
  }

  delete m_logger; m_logger = nullptr;
  delete m_inferEngine; m_inferEngine = nullptr;
  delete m_inferRuntime; m_inferRuntime = nullptr;

  cuStreamDestroy(m_cuStream);
  cuEventDestroy(m_frameProcessingStartEvent);
  cuEventDestroy(m_framePostProcessingStartEvent);
  cuEventDestroy(m_frameProcessingEndEvent);
  CUDA_SAFE_FREE(m_inputLUT);
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)
void EyeTrackingService::internalLoadSettings(cv::FileStorage& fs) {
  cv::FileNode trt = fs["tensorrt"];
  if (trt.isMap()) {
    //cv::read(trt["confidenceThreshold"], m_params.confidenceThreshold, m_params.confidenceThreshold);
    //cv::read(trt["quality"], m_params.quality, m_params.quality);
  }
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void EyeTrackingService::internalSaveSettings(cv::FileStorage& fs) {
  fs.startWriteStruct(cv::String("tensorrt"), cv::FileNode::MAP, cv::String());
    //fs.write("confidenceThreshold", m_params.confidenceThreshold);
    //fs.write("quality", m_params.quality);

  fs.endWriteStruct();
}
#undef writeNode


bool EyeTrackingService::processFrame() {

  // Capture thread maintenance:
  // Start captures if they're not running, clean up after exited threads

  PER_EYE {
    CaptureState& cs = m_captureState[eyeIdx];
    ProcessingState& ps = m_processingState[eyeIdx];

    if (cs.m_captureThreadAlive) {
      // Update buffer from the capture thread
      size_t bufIdx = ps.olderCaptureBufferIdx();
      ps.m_captureBuffers[bufIdx] = cs.m_captureBufferMailbox.exchange(ps.m_captureBuffers[bufIdx]);
      //printf("captureThreadAlive(%zu) buf %zu => %p\n", eyeIdx, bufIdx, ps.m_captureBuffers[bufIdx]);
    } else {
      if (!cs.m_inputFilename.empty()) {
        // Try re-opening capture, ratelimited to once a second
        if (deltaTimeMs(cs.m_lastCaptureOpenAttemptTimeNs, currentTimeNs()) > 1000.0f) {
          cs.m_lastCaptureOpenAttemptTimeNs = currentTimeNs();
          if (cs.m_capture.open(cs.m_inputFilename)) {
            // Capture is open, restart the capture thread
            cs.m_captureThread = boost::thread(boost::bind(&EyeTrackingService::CaptureState::captureWorkerThread, &cs));
            printf("EyeTrackingService: Successfully opened capture of \"%s\" for eye %zu. Backend: %s\n", cs.m_inputFilename.c_str(), eyeIdx, cs.m_capture.getBackendName().c_str());
          }
        }
      }
    }
  }

  if (!(m_captureState[0].m_captureThreadAlive || m_captureState[1].m_captureThreadAlive)) {
    // printf("EyeTrackingService: no capture threads alive\n");
    return false;
  }


  // Wait for previous processing to finish.
  cuEventSynchronize(m_frameProcessingEndEvent);
  cuEventElapsedTime(&m_lastFrameProcessingTimeMs, m_frameProcessingStartEvent, m_framePostProcessingStartEvent);
  cuEventElapsedTime(&m_lastFramePostProcessingTimeMs, m_framePostProcessingStartEvent, m_frameProcessingEndEvent);

  // Do postprocessing on whichever eye needed it
  PER_EYE {
    ProcessingState& ps = m_processingState[eyeIdx];
    if (!ps.m_requiresPostProcessing)
      continue;

    ps.m_requiresPostProcessing = false;
    uint64_t startTimeNs = currentTimeNs();

    // Find contours in the pupil mask
    cv::Mat pupilMask(m_trtInputHeight, m_trtInputWidth, CV_8UC1, ps.m_pupilMask1, /*step=*/ m_trtInputWidth);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContoursLinkRuns(pupilMask, contours);

    bool didFitEllipse = false;
    if (!contours.empty()) {
      std::vector<float> contourAreas;
      contourAreas.resize(contours.size());

      size_t bestContourIdx = 0;
      float bestContourArea = 0;

      for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx) {
        float area = cv::contourArea(contours[contourIdx]);
        contourAreas[contourIdx] = area;
        printf("Contour %zu: area = %f, %zu points\n", contourIdx, area, contours[contourIdx].size());

        if (area > bestContourArea) {
          bestContourArea = area;
          bestContourArea = contourIdx;
        }
      }

      // The algorithm needs a bare minimum of 5px to fit an ellipse
      // Try and give it a healthy margin
      if (contours[bestContourIdx].size() >= 20) {
        const auto& bestContour = contours[bestContourIdx];

        ps.m_pupilContourArea = bestContourArea;
        ps.m_pupilEllipse = cv::fitEllipse(bestContour);
        didFitEllipse = true;

        std::vector<cv::Point2f> pupil_inliers;
        pupil_inliers.resize(bestContour.size());
        for (size_t i = 0; i < pupil_inliers.size(); ++i) {
          pupil_inliers[i] = cv::Point2f(bestContour[i].x, bestContour[i].y);
        }

        // Add this observation to the eye model fitter
        ps.m_eyeModelFitter.add_observation(
          /*image (unused)=*/cv::Mat(),
          /*pupil=*/ singleeyefitter::toEllipse<double>(ps.m_pupilEllipse),
          /*inliers=*/ pupil_inliers);

        // Try and fit the model
        if (((ps.m_eyeModelFitter.pupils.size() & 63) == 0) && ps.m_eyeModelFitter.model_version == 0) {
          printf("Attempting eye model fit\n");
          if (ps.m_eyeModelFitter.unproject_observations()) {
            ps.m_eyeModelFitter.initialise_model();
          } else {
            printf("Eye model fit failed; unproject_observations() returned false.\n");
          }
        }
      }
    }

    if (!didFitEllipse) {
      // Clear pupil ellipse, since we didn't find anything useful this round.
      ps.m_pupilEllipse = cv::RotatedRect();
    }

    float postTimeMs = deltaTimeMs(startTimeNs, currentTimeNs());
    printf("Eye %zu CPU postprocess took %.3fms\n", eyeIdx, postTimeMs);

    //printf("Eye %zu postprocess required. coordinates:\n", eyeIdx);
    //for (size_t i = 0; i < m_trtOutputKeypointCount; ++i) {
    //  printf("  %.4f %.4f\n", ps.m_trtOutputHostPtr[(i * 2) + 0], ps.m_trtOutputHostPtr[(i * 2) + 1]);
    //}

  } // PER_EYE postprocessing


  // Pick which eye we're working on:
  // Use the eye with the oldest processing timestamp, but only if the capture is newer than that processing timestamp.
  m_currentlyProcessingEyeIdx = 0;
  if (m_processingState[1].m_lastProcessingTimeNs < m_processingState[0].m_lastProcessingTimeNs) {
    // Eye 1 is older, make sure that we have a valid buffer
    if (m_processingState[1].newerCaptureBufferTimestamp() > m_processingState[1].m_lastProcessingTimeNs) {
      // Eye 1 was processed less recently and has a new enough buffer on deck.
      m_currentlyProcessingEyeIdx = 1;
    }
  }

  ProcessingState& ps = m_processingState[m_currentlyProcessingEyeIdx];
  CaptureBuffer* capture = ps.m_captureBuffers[ps.newerCaptureBufferIdx()];

  if (capture == nullptr) {
    // sanity check
    printf("EyeTrackingService::processFrame(): buffer for eye %zu is null!\n", m_currentlyProcessingEyeIdx);
    return false;
  }
  ps.m_lastProcessingTimeNs = capture->timestamp;


  // Submit the next eye's image
  if (m_enableProfiling) {
    CUDA_CHECK(cuEventRecord(m_frameProcessingStartEvent, m_cuStream));
  }

  // Upload to m_preWarpGpuMat
  ps.m_preWarpGpuMat.create(capture->mat.size(), capture->mat.type());
  {
    CUDA_MEMCPY2D copy;
    memset(&copy, 0, sizeof(copy));
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = capture->mat.data;
    copy.srcPitch = capture->mat.step;

    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = (CUdeviceptr) ps.m_preWarpGpuMat.cudaPtr();
    copy.dstPitch = ps.m_preWarpGpuMat.step;

    copy.WidthInBytes = capture->mat.cols * capture->mat.elemSize();
    copy.Height = capture->mat.rows;

    cuMemcpy2DAsync(&copy, m_cuStream);
  }

  {
    NppiSize srcsz;
    srcsz.height = ps.m_preWarpGpuMat.rows;
    srcsz.width = ps.m_preWarpGpuMat.cols;

    NppiRect srcroi;
    srcroi.x = 0;
    srcroi.y = 0;
    srcroi.height = ps.m_preWarpGpuMat.rows;
    srcroi.width = ps.m_preWarpGpuMat.cols;

    NppiSize dstsz;
    dstsz.height = ps.m_preHistEqMat.rows;
    dstsz.width = ps.m_preHistEqMat.cols;

    NppiRect dstroi;
    dstroi.x = 0;
    dstroi.y = 0;
    dstroi.height = ps.m_preHistEqMat.rows;
    dstroi.width = ps.m_preHistEqMat.cols;

#if 0
    // Crop/rotate the capture buffer

    double coeffs[2][3];
    Mat coeffsMat(2, 3, CV_64F, (void*)coeffs);
    M.convertTo(coeffsMat, coeffsMat.type());

    nppiWarpAffine_8u_C1R_Ctx(
      ps.m_preWarpGpuMat.ptr<Npp8u>(), srcsz, static_cast<int>(ps.m_preWarpGpuMat.step), srcroi,
      ps.m_preHistEqMat.ptr<Npp8u>(), static_cast<int>(ps.m_preHistEqMat.step), dstroi,
      coeffs, NPPI_INTER_LINEAR, m_nppContext);

#else
    // Just resize the capture buffer
    // NppStatus nppiResize_8u_C1R_Ctx(
    //    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
    //    Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
    //    int eInterpolation, NppStreamContext nppStreamCtx)

    NPP_CHECK(nppiResize_8u_C1R_Ctx(
      ps.m_preWarpGpuMat.ptr<Npp8u>(), static_cast<int>(ps.m_preWarpGpuMat.step), srcsz, srcroi,
      ps.m_preHistEqMat.ptr<Npp8u>(), static_cast<int>(ps.m_preHistEqMat.step), dstsz, dstroi,
      NPPI_INTER_LINEAR, m_nppContext));
#endif

  }


#if 0
  // CLAHE Histogram equalization
  //ps.m_clahe->apply(/*src=*/ ps.m_preHistEqMat, /*dst=*/ ps.m_postHistEqMat, m_cvStream);
  // TODO: Enable CLAHE (maybe as an option).
  cv::cuda::equalizeHist(/*src=*/ ps.m_preHistEqMat, /*dst=*/ ps.m_postHistEqMat, m_cvStream);

  // Apply format conversion for TRT input
  ApplyLUT8to16(ps.m_postHistEqMat, ps.m_trtInputMat, (const ushort*) m_inputLUT, m_cuStream);

#else
  // Format conversion only, no histogram equalization
  ApplyLUT8to16(ps.m_preHistEqMat, ps.m_trtInputMat, (const ushort*) m_inputLUT, m_cuStream);
  ps.m_postHistEqMat = ps.m_preHistEqMat;

#endif

#if USE_CUDA_GRAPH
  // Launch TRT processing graph for the currently selected eye.
  CUDA_CHECK(cuGraphLaunch(ps.m_frameProcessingGraphExec, m_cuStream));
#else
  // Launch TRT processing (no graph)
  assert(ps.m_exec->enqueueV3(m_cuStream));
#endif

  if (m_enableProfiling) {
    CUDA_CHECK(cuEventRecord(m_framePostProcessingStartEvent, m_cuStream));
  }

  // Run CUDA postprocessing operations:

  // Run argmax over the return from the segmentation network to find the class index
  ComputeClassIndex(ps.m_trtOutputHostPtr, ps.m_classIndex, m_trtInputWidth, m_trtInputHeight, m_cuStream);

  {
    NppiSize sz;
    sz.width = m_trtInputWidth;
    sz.height = m_trtInputHeight;

    NppiPoint zero;
    zero.x = 0;
    zero.y = 0;


    // Generate a binary mask from the pupil class

    // NppStatus nppiCompareC_8u_C1R_Ctx(
    //   const Npp8u *pSrc, int nSrcStep,
    //   const Npp8u nConstant,
    //   Npp8u *pDst, int nDstStep,
    //   NppiSize oSizeROI,
    //   NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx)
    NPP_CHECK(nppiCompareC_8u_C1R_Ctx(
      ps.m_classIndex, m_trtInputWidth,
      /*pupil class=*/ 3,
      ps.m_pupilMask1, m_trtInputWidth,
      sz,
      NPP_CMP_EQ, m_nppContext));

    // Run closure operation to fill holes in the mask -- 3x3 dilation followed by 3x3 erosion

    // NppStatus nppiDilate3x3Border_8u_C1R_Ctx(
    //   const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    //         Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
    //         NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
    //
    NPP_CHECK(nppiDilate3x3Border_8u_C1R_Ctx(
      ps.m_pupilMask1, m_trtInputWidth, sz, zero,
      ps.m_pupilMask2, m_trtInputWidth, sz,
      NPP_BORDER_REPLICATE, m_nppContext));

    // NppStatus nppiErode3x3Border_8u_C1R_Ctx(
    //   const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    //         Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
    //         NppiBorderType eBorderType, NppStreamContext nppStreamCtx)
    NPP_CHECK(nppiErode3x3Border_8u_C1R_Ctx(
      ps.m_pupilMask2, m_trtInputWidth, sz, zero,
      ps.m_pupilMask1, m_trtInputWidth, sz,
      NPP_BORDER_REPLICATE, m_nppContext));

  }

  ps.m_requiresPostProcessing = true;

  if (m_enableProfiling) {
    CUDA_CHECK(cuEventRecord(m_frameProcessingEndEvent, m_cuStream));
  }

  return true;
}

void EyeTrackingService::CaptureState::captureWorkerThread() {
  m_captureThreadAlive = true;

  CaptureBuffer* nextBuf = nullptr;

  cv::Mat capRgbMat;

  bool doFramePacing = ((int) m_capture.get(cv::CAP_PROP_BACKEND)) == cv::CAP_FFMPEG;
  double fps = m_capture.get(cv::CAP_PROP_FPS);
  uint64_t frameTimeNs = 1'000'000'000.0 / fps;

  if (doFramePacing) {
      printf("EyeTrackingService::CaptureState(%p): Frame pacing enabled, FPS = %.3f, target frametime is %luns\n", this, fps, frameTimeNs);
  }

  uint64_t decodeStartTimeNs = currentTimeNs();
  uint64_t frameCount = 0;

  while (true) {
    // Ensure we have a buffer to capture into
    if (!nextBuf)
      nextBuf = new CaptureBuffer();

    // Capture frame
    if (!m_capture.read(capRgbMat)) {
      printf("EyeTrackingService::CaptureState(%p)::captureWorkerThread: read() returned false, terminating\n", this);
      break;
    }
    if (capRgbMat.empty()) {
      printf("EyeTrackingService::CaptureState(%p)::captureWorkerThread: mat is empty, terminating\n", this);
      break;
    }
    cv::cvtColor(/*src=*/ capRgbMat, /*dst=*/ nextBuf->mat, cv::COLOR_BGR2GRAY);
    uint64_t now = currentTimeNs();
    nextBuf->timestamp = now;

    // Hand off to main thread
    nextBuf = m_captureBufferMailbox.exchange(nextBuf);

    ++frameCount;
    if (doFramePacing) {
      uint64_t targetTimeNs = (frameTimeNs * frameCount) + decodeStartTimeNs;
      if (now < targetTimeNs) {
        delayNs(targetTimeNs - now);
      }
    }
  }

  // Cleanup
  if (nextBuf)
    delete nextBuf;

  m_capture.release();
  m_captureThreadAlive = false;

}

void polyline(cv::Mat& img, const cv::Point2f* points, size_t startIdx, size_t endIdx, const cv::Scalar& color, bool loop = false) {
  for (size_t i = startIdx; i < endIdx; ++i) {
    cv::line(img, points[i], points[i + 1], color);
  }
  if (loop)
    cv::line(img, points[endIdx], points[startIdx], color);
}


#if 0
cv::Point2f toImgCoord(const cv::Point2f& point, uint32_t width, uint32_t height) {
    return cv::Point2f(static_cast<float>((width/2 + point.x)),
        static_cast<float>((height/2 + point.y)));
}
cv::Point toImgCoord(const cv::Point& point, uint32_t width, uint32_t height) {
    return cv::Point(static_cast<int>((width/2 + point.x)),
        static_cast<int>((height/2 + point.y)));
}
cv::RotatedRect toImgCoord(const cv::RotatedRect& rect, uint32_t width, uint32_t height) {
    return cv::RotatedRect(toImgCoord(rect.center, width, height),
        cv::Size2f(rect.size.width, rect.size.height), rect.angle);
}
#else
cv::Point2f toImgCoord(const cv::Point2f& point, uint32_t width, uint32_t height) {
    return point;
}
cv::Point toImgCoord(const cv::Point& point, uint32_t width, uint32_t height) {
    return point;
}
cv::RotatedRect toImgCoord(const cv::RotatedRect& rect, uint32_t width, uint32_t height) {
    return rect;
}
#endif

cv::Mat& EyeTrackingService::getDebugViewForEye(size_t eyeIdx) {
  assert(eyeIdx < 2);
  ProcessingState& ps = m_processingState[eyeIdx];

  if (ps.m_postHistEqMat.empty()) {
    ps.m_debugViewRGB = cv::Mat(); // clear to empty mat
    return ps.m_debugViewRGB;
  }

  // Download last GPU copy of the greyscale mat that was the TensorRT input (after all cropping/scaling, before LUT/format conversion)
  ps.m_postHistEqMat.download(ps.m_debugViewGrey);

  // Promote greyscale mat to RGB so we can draw markers on it
  cv::cvtColor(/*src=*/ ps.m_debugViewGrey, /*dst=*/ ps.m_debugViewRGB, cv::COLOR_GRAY2BGR);

  // Draw segmentation class colors
#if 1
  for (size_t row = 0; row < ps.m_debugViewRGB.rows; ++row) {
    uint8_t* pupilRowPtr = ps.m_pupilMask1 + (row * m_trtInputWidth);
    for (size_t col = 0; col < ps.m_debugViewRGB.cols; ++col) {
      if (pupilRowPtr[col])
        ps.m_debugViewRGB.ptr<uint8_t>(row, col)[/*red channel=*/2] = 0x7f;
    }
  }

#else
  for (size_t row = 0; row < ps.m_debugViewRGB.rows; ++row) {
    uint8_t* classRowPtr = ps.m_classIndex + (row * m_trtInputWidth);
    for (size_t col = 0; col < ps.m_debugViewRGB.cols; ++col) {
      uint8_t classIdx = classRowPtr[col];
      if (classIdx > 0) { // non-BG
        // Saturate one of the color channels matching the class index
        ps.m_debugViewRGB.ptr<uint8_t>(row, col)[classIdx - 1] = 0xff;
      }
      ps.m_debugViewRGB.ptr<uint8_t>(row, col)[/*red channel=*/0] = 0xff;
    }
  }
#endif

  // Draw pupil ellipse, if present
  if (!ps.m_pupilEllipse.size.empty()) {
    cv::ellipse(ps.m_debugViewRGB, ps.m_pupilEllipse, cv::Scalar(0xff, 0, 0xff), /*thickness=*/ 2);
  }

  if (ps.m_eyeModelFitter.hasEyeModel()) {
    singleeyefitter::Circle3D<double> circle;
    if (ps.m_eyeModelFitter.unproject_single_observation(circle, singleeyefitter::toEllipse<double>(ps.m_pupilEllipse))) {
      singleeyefitter::Conic<double> pupil_conic = singleeyefitter::project(circle, ps.m_eyeModelFitter.focal_length);
      singleeyefitter::Ellipse2D<double> eye_ellipse = singleeyefitter::project(ps.m_eyeModelFitter.eye, ps.m_eyeModelFitter.focal_length);


      cv::ellipse(ps.m_debugViewRGB, toImgCoord(toRotatedRect(singleeyefitter::Ellipse2D<double>(pupil_conic)), m_trtInputWidth, m_trtInputHeight), cv::Scalar(60, 60, 0), /*thickness=*/ 2);
      cv::ellipse(ps.m_debugViewRGB, toImgCoord(toRotatedRect(eye_ellipse), m_trtInputWidth, m_trtInputHeight), cv::Scalar(0, 60, 60), /*thickness=*/ 2);

      char buf[64];
      snprintf(buf, 64, "n=%.3f %.3f %.3f", circle.normal[0], circle.normal[1], circle.normal[2]);

      cv::putText(ps.m_debugViewRGB, buf, cv::Point2f(/*x=*/ 5, /*y=*/ m_trtInputHeight - 16), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

      //printf("Ellipse: center=%.3f %.3f\n width=%.3f height=%.3f\n",
      //    ps.m_pupilEllipse.center.x, ps.m_pupilEllipse.center.y,
      //    ps.m_pupilEllipse.size.width, ps.m_pupilEllipse.size.height);

    }
  }

  // Convert markers to pixel coordinates
  //for (size_t markerIdx = 0; markerIdx < m_trtOutputKeypointCount; ++markerIdx) {
  //  markers[markerIdx] = cv::Point2f(
  //    ps.m_trtOutputHostPtr[(markerIdx * 2) + 0] * static_cast<float>(ps.m_debugViewRGB.cols),
  //    ps.m_trtOutputHostPtr[(markerIdx * 2) + 1] * static_cast<float>(ps.m_debugViewRGB.rows));
  //}

  // Draw dots on all  markers
  //for (size_t markerIdx = 0; markerIdx < m_trtOutputKeypointCount; ++markerIdx) {
  //  cv::circle(ps.m_debugViewRGB, markers[markerIdx], /*radius=*/ 3, cv::Scalar(255, 255,   0), /*thickness=*/ -1);
  //}

  // Draw lines to surround the upper and lower eyelids
  // markers 0-16, 33 are the top eyelid
  // markers 0, 17-33 are the bottom eyelid
  // 34-41 are the pupil surround
  // 42 is the pupil center

  //const cv::Scalar red = cv::Scalar(0, 0, 255);
  //const cv::Scalar green = cv::Scalar(0, 255, 0);
  //const cv::Scalar blue = cv::Scalar(255, 0, 0);

  // upper
  //polyline(ps.m_debugViewRGB, markers, 0, 16, green);
  //cv::line(ps.m_debugViewRGB, markers[16], markers[1], green);

  // lower
  //cv::line(ps.m_debugViewRGB, markers[0], markers[17], green);
  //polyline(ps.m_debugViewRGB, markers, 17, 33, green);

  // pupil surround
  //polyline(ps.m_debugViewRGB, markers, 34, 41, blue, /*loop=*/ true);

  // pupil center
  //cv::circle(ps.m_debugViewRGB, markers[42], /*radius=*/ 3, red, /*thickness=*/ 1);
/*
  for (size_t markerIdx = 0; markerIdx < 7; ++markerIdx) {
    char textBuf[16];
    textBuf[0] = 'a' + markerIdx; // a, b, c, d, e, f, g
    textBuf[1] = '\0';
    cv::putText(ps.m_debugViewRGB, textBuf, marker + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }
*/

  return ps.m_debugViewRGB;
}

