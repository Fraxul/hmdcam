#pragma once
#include "common/DepthMapGenerator.h"

class DepthMapGeneratorMock : public DepthMapGenerator {
public:
  DepthMapGeneratorMock();
  virtual ~DepthMapGeneratorMock();

protected:

  virtual void internalLoadSettings(cv::FileStorage&);
  virtual void internalSaveSettings(cv::FileStorage&);
  virtual void internalProcessFrame();
  virtual void internalRenderIMGUI();
  virtual void internalRenderIMGUIPerformanceGraphs();

  bool m_didChangeSettings = true; // force initial algorithm setup
  int m_disparityBytesPerPixel = 1;

  struct ViewDataMock : public ViewData {
    ViewDataMock() {}
    virtual ~ViewDataMock() {}

    size_t m_shmViewIndex;

    cv::Mat fakeDisparity;
  };
  virtual ViewData* newEmptyViewData() { return new ViewDataMock(); }
  virtual void internalUpdateViewData();

private:
  ViewDataMock* viewDataAtIndex(size_t index) { return static_cast<ViewDataMock*>(m_viewData[index]); }
};

