#pragma once
#include "common/CANBus.h"
#include "common/FxRenderView.h"
#include "common/ScrollingBuffer.h"
#include "TrackingThreadBase.h"
#include "one_euro_filter.h"
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <glm/glm.hpp>
#include <opencv2/core.hpp>
#include "CuDLAStandaloneRunner.h"
#include "V4L2Camera.h"

#include <algorithm>
#include <memory>
#include <vector>

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

class FaceTrackingService {
public:
  FaceTrackingService();
  ~FaceTrackingService();

  bool loadCalibrationData();
  bool loadCalibrationData(cv::FileStorage&);

  void saveCalibrationData();
  void saveCalibrationData(cv::FileStorage&);

  void renderIMGUI();

  void renderSceneGizmos_preUI(FxRenderView* renderViews);
  void renderSceneGizmos_postUI(FxRenderView* renderViews);

  void setInputDeviceOverride(const std::string& s) {
    m_processingState.m_cameraDeviceNameOverride = s;
  }

  bool processFrame(); // Called from main thread

  void CANTransmitTrackingData();

  void requestCapture();

  bool m_debugShowFeedbackView = false; // Draw camera(s) over the scene in renderSceneGizmos(). Required for getDebugView() to return anything.
  bool m_debugFreezeCapture = false;
  float m_debugFeedbackBrightness = 1.0f;
  const char* getDebugPerfStats() { return m_processingState.getDebugPerfStats(); }

  class ProcessingState : public TrackingThreadBase {
  public:
    void loadTrackingEngine();
    virtual ~ProcessingState();

    FaceTrackingService* m_service = nullptr; // Ref to containing service

    virtual void internalUpdateStateOnCaptureOpen();
    virtual void internalProcessOneCapture();


    // Output data.
    float m_browPosition = 0.0f; // -1 to 1 range
    // Filter initializer values are overwritten in applyCalibrationData
    one_euro_filter<float, double> m_browPositionFilter = {/*freq=*/ 120, /*minCutoff=*/ 1, /*beta=*/ 0.1, /*dcutoff=*/ 1};

    float* m_channelData = nullptr;


    // Profiling stats
    float m_lastFrameTotalProcessingTimeMs = 0.0f;

    // Debug view support
    bool m_debugShowFeedbackView = false;
    cv::Mat m_debugViewRGB; // RGB debug view, optionally with debug overlays drawn on it

    cv::Mat m_tempRGBDebugMat; // Draw to this one, swap with m_debugViewRGB to avoid main thread getting a partially-drawn debug view.
    RHISurface::ptr m_debugTexture;

    const char* getDebugPerfStats();


    // DLA model
    std::unique_ptr<CuDLAStandaloneRunner> m_trackingModel;
    cv::Rect m_captureCropRect;
    cv::Mat m_inputScaleMat;

    uint32_t m_inputWidth = 0, m_inputHeight = 0, m_inputRowStrideElements = 0;
    uint32_t m_trackingOutputChannels = 0, m_trackingOutputChannelPitchElements = 0;

    bool m_ioIsInt8 = false;


    // Additional debugging data
    struct GraphData {
      float browPosition = 0.0f;
    };
    bool m_freezeGraphData = false;
    ScrollingBuffer<GraphData> m_graphData { 120 /*samples*/ };


private:
    char m_debugPerfStatsBuffer[256]; // Accessed/populated through getDebugPerfStats()
  };

  ProcessingState m_processingState;

protected:

  void processingThreadFn();

  void applyCalibrationData();

  float m_browPositionScale = 1.0f;
  float m_browPositionExponent = 1.0f;

  float m_filterMinCutoff = 0.2;
  float m_filterDCutoff = 0.2;
  float m_filterBetaExponent = -0.2; // Filter beta is pow(10.0, m_filterBetaExponent)
};

