#include "FaceTrackingService.h"
#include "CuDLAStandaloneRunner.h"
#include "VectorOps.h"
#include "common/FxThreading.h"
#include "common/Timing.h"
#include "common/mmfile.h"
#include "imgui.h"
#include "implot/implot.h"
#include "stb/stb_image_write.h"
#include "rhi/RHIResources.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <math.h>
#include <dirent.h>


static const char* calibrationFilename = "facetracking-calibration.yml";
static const char* kCaptureDirName = "facetracking-captures";

FaceTrackingService::FaceTrackingService() {

  loadCalibrationData();

  m_processingState.m_service = this;
  m_processingState.m_captureDirName = kCaptureDirName;
  m_processingState.loadTrackingEngine();
  // Throttle down the face tracking camera to 60fps
  m_processingState.m_capture.setTargetFramerate(60);

  applyCalibrationData();
}

FaceTrackingService::~FaceTrackingService() {
  m_processingState.shutdownThread();
}

void FaceTrackingService::ProcessingState::loadTrackingEngine() {
  // Load engine
  {
    mmfile fp("eyetracking/models/facetracking-dla-standalone.engine");
    m_trackingModel.reset(new CuDLAStandaloneRunner(0, reinterpret_cast<const uint8_t*>(fp.data()), fp.size()));
  }

  // Get the input size
  {
    const cudlaModuleTensorDescriptor& desc = m_trackingModel->inputTensorDescriptor(0);
    m_inputWidth = desc.w;
    m_inputHeight = desc.h;

    m_inputRowStrideElements = desc.stride[1] / desc.stride[0];
    printf("Tracking image dimensions: %ux%u. Row stride is %u elements\n", m_inputWidth, m_inputHeight, m_inputRowStrideElements);
  }

  // Get the output size
  {
    const cudlaModuleTensorDescriptor& desc = m_trackingModel->outputTensorDescriptor(0);

    // Output should have w/h == 1, batch size == 1, and some number of channels.
    assert(1 == desc.w);
    assert(1 == desc.h);
    assert(desc.c >= 1);
    assert(1 == desc.n);

    m_trackingOutputChannels = desc.c;
    m_trackingOutputChannelPitchElements = desc.stride[2] / desc.stride[0];

    printf("Tracking output is %u channels, channel pitch is %u elements\n", m_trackingOutputChannels, m_trackingOutputChannelPitchElements);
    m_channelData = new float[m_trackingOutputChannels];

    // Check I/O datatypes
    if (desc.dataType == CUDLA_DATA_TYPE_INT8) {
      m_ioIsInt8 = true;
    } else if (desc.dataType == CUDLA_DATA_TYPE_HALF) {
      m_ioIsInt8 = false;
    } else {
      assert(false && "Facetracking engine: unsupported I/O type (must be kHALF or kINT8)");
    }

    // Only support same input and output types
    assert(m_trackingModel->inputTensorDescriptor(0).dataType == desc.dataType);
  }

}

bool FaceTrackingService::loadCalibrationData() {

  cv::FileStorage fs(calibrationFilename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("Unable to open calibration data file %s\n", calibrationFilename);
    return false;
  }

  return loadCalibrationData(fs);
}

void FaceTrackingService::saveCalibrationData() {
  cv::FileStorage fs(calibrationFilename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  saveCalibrationData(fs);
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)
bool FaceTrackingService::loadCalibrationData(cv::FileStorage& fs) {
  try {
    readNode(fs, filterMinCutoff);
    readNode(fs, filterBetaExponent);
    readNode(fs, filterDCutoff);
    cv::read(fs["faceCamera"], m_processingState.m_cameraDeviceName, m_processingState.m_cameraDeviceName);

  } catch (const std::exception& ex) {
    printf("Unable to load calibration data: %s\n", ex.what());
    return false;
  }
  return true;
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void FaceTrackingService::saveCalibrationData(cv::FileStorage& fs) {
  writeNode(fs, filterMinCutoff);
  writeNode(fs, filterBetaExponent);
  writeNode(fs, filterDCutoff);
  fs.write("faceCamera", m_processingState.m_cameraDeviceName);

}
#undef writeNode


FaceTrackingService::ProcessingState::~ProcessingState() {
  if (m_channelData)
    delete[] m_channelData;
}


float interpolateChannels(size_t channelCount, const float* weights, const float* values) {
  if (channelCount == 0)
    return 0;
  else if (channelCount == 1)
    return values[0];

  // Find max weight index
  size_t maxWeightIdx = 0;
  {
    float maxWeight = 0.0f;
    for (size_t i = 0; i < channelCount; ++i) {
      if (weights[i] > maxWeight) {
        maxWeight = weights[i];
        maxWeightIdx = i;
      }
    }
  }

  // Find the larger adjacent value
  size_t adjacentValueIdx;
  if (maxWeightIdx == 0) {
    // Special case: [0] is largest, so the largest adjacent value index must be 1
    adjacentValueIdx = 1;
  } else if (maxWeightIdx == (channelCount - 1)) {
    // Special case: End of the array is largest, so the largest adjacent value list must be second-to-end.
    adjacentValueIdx = maxWeightIdx - 1;
  } else {
    adjacentValueIdx = (weights[maxWeightIdx - 1] > weights[maxWeightIdx + 1]) ? (maxWeightIdx - 1) : (maxWeightIdx + 1);
  }

  // Interpolate between the selected largest and second-largest values.
  float t = weights[adjacentValueIdx] / (weights[maxWeightIdx] + weights[adjacentValueIdx]);

  // Adjustment for the curve tending to stick around the category crossover points
  t = sqrtf(t);

  return glm::mix(values[maxWeightIdx], values[adjacentValueIdx], t);
}

void FaceTrackingService::ProcessingState::internalUpdateStateOnCaptureOpen() {
  // Compute capture mat crop rect.
  {
    uint32_t targetStreamWidth = 960;
    uint32_t targetStreamHeight = 720;
    uint32_t cropOffsetX = (m_capture.streamWidth() - targetStreamWidth) / 2;
    uint32_t cropOffsetY = (m_capture.streamHeight() - targetStreamHeight) / 2;
    m_captureCropRect = cv::Rect(cropOffsetX, cropOffsetY, targetStreamWidth, targetStreamHeight);
    printf("FaceTrackingService::ProcessingState::internalUpdateStateOnCaptureOpen(): Capture dimensions are %ux%u, crop is (%ux%u @ %u,%u)\n",
      m_capture.streamWidth(), m_capture.streamHeight(),
      targetStreamWidth, targetStreamHeight,
      cropOffsetX, cropOffsetY);
  }
}

void FaceTrackingService::ProcessingState::internalProcessOneCapture() {
  PerfTimer perfTimer;

  // Extract crop region from the luma plane
  cv::Mat captureMat = cv::Mat(m_capture.lumaPlane(), m_captureCropRect);
  // cv::Mat captureMat = m_capture.lumaPlane();

  // Scale from capture mat to the input size for the classification network
  cv::resize(captureMat, m_inputScaleMat, cv::Size(m_inputWidth, m_inputHeight), 0, 0, cv::INTER_LINEAR);

  // Convert CV_U8 from the scale output to int8 or half for the network input
  if (m_ioIsInt8) {
    int8_t* inputTensor = m_trackingModel->inputTensorPtr<int8_t>(0);
    for (size_t row = 0; row < m_inputHeight; ++row) {
      convertUnorm8ToDLAInt8(m_inputScaleMat.ptr<uint8_t>(row), inputTensor + (m_inputRowStrideElements * row), m_inputWidth);
    }
  } else {
    _Float16* inputTensor = m_trackingModel->inputTensorPtr<_Float16>(0);
    for (size_t row = 0; row < m_inputHeight; ++row) {
      convertUnorm8ToSnormFp16(m_inputScaleMat.ptr<uint8_t>(row), inputTensor + (m_inputRowStrideElements * row), m_inputWidth);
    }
  }

  // Launch classification network.
  m_trackingModel->asyncStartInference();

  // Cache this flag since it may be written async on the main thread.
  bool populateDebugView = m_service->m_debugShowFeedbackView;

  if (populateDebugView) {
    // While the DLA is running the network (~1 ms), convert the capture buffer to RGBA in preparation for debug drawing.
    // cv::cvtColor(/*src=*/ captureMat, /*dst=*/ m_tempRGBDebugMat, cv::COLOR_GRAY2RGBA); // Original
    cv::cvtColor(/*src=*/ m_inputScaleMat, /*dst=*/ m_tempRGBDebugMat, cv::COLOR_GRAY2RGBA); // Rescaled input
  }

  // Wait for classification network to finish
  m_trackingModel->asyncFinishInference();


  if (m_ioIsInt8) {
    assert(false && "TODO: output processing for int8 i/o");
  } else {
    _Float16* basePtr = m_trackingModel->outputTensorPtr<_Float16>(0);
    for (uint32_t channelIdx = 0; channelIdx < m_trackingOutputChannels; ++channelIdx) {
      m_channelData[channelIdx] = static_cast<float>(basePtr[channelIdx * m_trackingOutputChannelPitchElements]);
    }
  }

  // Process/filter channel data
  GraphData graphDataFrame;


  {
    float neutral = m_channelData[0];
    float brow_down_partial = m_channelData[1];
    float brow_down = m_channelData[2];
    float brow_up_partial = m_channelData[3];
    float brow_up = m_channelData[4];

    float weights[] = {brow_down, brow_down_partial, neutral, brow_up_partial, brow_up};
    float values[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};

    float browPositionRaw = interpolateChannels(5, weights, values);
    // Add sample to filter
    double ts = static_cast<double>(currentTimeNs() / 1000ULL) / 1'000'000.0;
    graphDataFrame.browPosition = browPositionRaw;
    m_browPosition = m_browPositionFilter(browPositionRaw, ts);
  }


  // Update data graphs
  m_graphData.push_back(graphDataFrame);

  // ---

  if (populateDebugView) {
    // Swap debug mat with the one in processing state
    // (tries to avoid the main thread getting a partially-drawn debug mat)
    cv::swap(m_debugViewRGB, m_tempRGBDebugMat);
  }

  m_lastFrameTotalProcessingTimeMs = perfTimer.totalElapsedTime();
}

const char* FaceTrackingService::ProcessingState::getDebugPerfStats() {
  char* buf = m_debugPerfStatsBuffer;
  constexpr size_t len = sizeof(m_debugPerfStatsBuffer);

  if (processingThreadAlive()) {
    snprintf(buf, len - 1,
      "Processing time: %.3fms",
      m_lastFrameTotalProcessingTimeMs);
  } else {
    snprintf(buf, len - 1,
      "Processing thread not running");
  }

  buf[len - 1] = '\0';
  return m_debugPerfStatsBuffer;
}

void FaceTrackingService::renderIMGUI() {
  ImGui::PushID(this);

  bool dirty = false;
  ImGui::Checkbox("FT capture freeze", &m_debugFreezeCapture);
  ImGui::Checkbox("FT camera feedback view", &m_debugShowFeedbackView);


  if (ImGui::CollapsingHeader("Data graphs")) {
    int plotFlags = ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;

    if (ImPlot::BeginPlot("##BrowPosition", ImVec2(-1,150), /*flags=*/ plotFlags)) {
      ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
      ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_Lock);
      ImPlot::SetupAxisLimits(ImAxis_X1, 0, m_processingState.m_graphData.size(), ImPlotCond_Always);
      ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImPlotCond_Always);
      ImPlot::SetupFinish();

      ImPlot::PlotLine("Brow Position", &m_processingState.m_graphData.data()[0].browPosition, m_processingState.m_graphData.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, m_processingState.m_graphData.offset(), sizeof(ProcessingState::GraphData));
      ImPlot::EndPlot();
    }
  }

  // Filter settings require calling applyCalibrationData
  ImGui::Separator();
  dirty |= ImGui::DragFloat("Filter min cutoff", &m_filterMinCutoff, /*speed=*/ 0.01f, /*min=*/ 0.0f, /*max=*/ 2.0f, "%.2f");
  dirty |= ImGui::DragFloat("Filter D cutoff", &m_filterDCutoff, /*speed=*/ 0.01f, /*min=*/ 0.0f, /*max=*/ 2.0f, "%.2f");
  dirty |= ImGui::DragFloat("Filter beta exponent", &m_filterBetaExponent, /*speed=*/ 0.1f, /*min=*/ -10.0f, /*max=*/ 10.0f, "%.1f");


  if (ImGui::Button("Save Settings")) {
    saveCalibrationData();
  }

  if (dirty) {
    applyCalibrationData();
  }

  ImGui::PopID();
}

void FaceTrackingService::renderSceneGizmos_preUI(FxRenderView* renderViews) {
  // TODO parameterize? or pull from external config
  const float feedbackViewDepth = 0.5f;

  // Render debug view
  if (m_debugShowFeedbackView) {
    const float feedbackViewScale = 0.175f;

    ProcessingState& ps = m_processingState;
    const cv::Mat& debugView = ps.m_debugViewRGB;

    if (debugView.cols && debugView.rows) {

      if (!ps.m_debugTexture || (ps.m_debugTexture->width() != debugView.cols) || (ps.m_debugTexture->height() != debugView.rows)) {
        ps.m_debugTexture = rhi()->newTexture2D(debugView.cols, debugView.rows, kSurfaceFormat_RGBA8);
      }

      // Eyetracking debug view is drawn in RGBA, so we just have to upload it.
      rhi()->loadTextureData(ps.m_debugTexture, kVertexElementTypeUByte4N, debugView.ptr());


      rhi()->bindBlendState(disabledBlendState);
      rhi()->bindDepthStencilState(disabledDepthStencilState);
      rhi()->bindRenderPipeline(uiLayerStereoPipeline);
      rhi()->loadTexture(ksImageTex, ps.m_debugTexture, linearClampSampler);
      // rhi()->setViewports(eyeViewports, 2); // should already be set

      UILayerStereoUniformBlock ub;
      glm::mat4 modelMatrix = glm::translate(glm::vec3(0.0f, 0.0f, -feedbackViewDepth)) * glm::scale(glm::vec3(feedbackViewScale * (static_cast<float>(ps.m_debugTexture->width()) / static_cast<float>(ps.m_debugTexture->height())), -feedbackViewScale, feedbackViewScale));
      ub.modelViewProjection[0] = renderViews[0].viewProjectionMatrix * modelMatrix;
      ub.modelViewProjection[1] = renderViews[1].viewProjectionMatrix * modelMatrix;
      ub.tint = glm::vec4(m_debugFeedbackBrightness, m_debugFeedbackBrightness, m_debugFeedbackBrightness, 1.0f);

      rhi()->loadUniformBlockImmediate(ksUILayerStereoUniformBlock, &ub, sizeof(ub));
      rhi()->drawNDCQuad();
    }
  }
}

void FaceTrackingService::renderSceneGizmos_postUI(FxRenderView* renderViews) {
  // TODO nothing to render here right now. Should add an overlay with bar graphs for the expression parameters or whatever
}

void FaceTrackingService::requestCapture() {
  // Make sure the target directory exists
  if (mkdir(kCaptureDirName, 0777) < 0) {
    // EEXIST is ok, anything else is fatal
    if (errno != EEXIST) {
      printf("FaceTrackingService::requestCapture(): can't create capture directory \"%s\": %s\n", kCaptureDirName, strerror(errno));
      return;
    }
  }

  // Select a consistent capture index across all sessions.
  uint64_t captureIndex = currentRealTimeMs();

  // Trigger capture in any active sessions
  ProcessingState& ps = m_processingState;
  if (ps.processingThreadAlive()) {
    ps.m_captureFileIndex = captureIndex;
  }

}

bool FaceTrackingService::processFrame() {
  // Copy some flags from the service to the tracking thread base
  m_processingState.m_debugFreezeCapture = m_debugFreezeCapture;

  m_processingState.processFrameHook();

  return m_processingState.processingThreadAlive();
}

void FaceTrackingService::CANTransmitTrackingData() {
  constexpr uint16_t kPortID = 202;

  // Signed 8-bit integer for brow position; 0 is neutral, 127 is full up, -127 is full down.
  int8_t browPosition = static_cast<int8_t>(glm::clamp<float>(m_processingState.m_browPosition, -1.0f, 1.0f) * 127.0f);


  SerializationBuffer buf;
  buf.reserve(1);

  buf.put_i8(browPosition);

  canbus()->transmitMessage(kPortID, buf);
}

void FaceTrackingService::applyCalibrationData() {
  m_processingState.m_browPositionFilter.mincutoff = m_filterMinCutoff;
  m_processingState.m_browPositionFilter.beta = powf(10.0f, m_filterBetaExponent);
  m_processingState.m_browPositionFilter.dcutoff = m_filterDCutoff;
}

