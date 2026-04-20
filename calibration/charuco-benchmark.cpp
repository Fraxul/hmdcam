#include <cstdio>
#include <cstdint>
#include <cassert>
#include <set>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "common/Timing.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 14, CV_64F);

// ChAruCo target pattern config
const cv::aruco::PredefinedDictionaryType s_charucoDictionaryName = cv::aruco::DICT_5X5_100;
const unsigned int s_charucoBoardSquareCountX = 12;
const unsigned int s_charucoBoardSquareCountY = 9;
const float s_charucoBoardSquareSideLengthMeters = 0.060f;
// markers are 7x7 pixels, squares are 9x9 pixels (add 1px border), so the marker size is 7/9 of the square size
const float s_charucoBoardMarkerSideLengthMeters = s_charucoBoardSquareSideLengthMeters * (7.0f / 9.0f);

cv::aruco::Dictionary charucoDictionary() { return cv::aruco::getPredefinedDictionary(s_charucoDictionaryName); }
cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard;
cv::aruco::DetectorParameters s_detectorParams;

cv::aruco::CharucoDetector createCharucoDetector(cv::Mat cameraMatrix = cv::Mat(), cv::Mat distCoeffs = cv::Mat()) {
  cv::aruco::CharucoParameters chParams;
  chParams.cameraMatrix = cameraMatrix;
  chParams.distCoeffs = distCoeffs;
  chParams.tryRefineMarkers = true;
  return cv::aruco::CharucoDetector(*s_charucoBoard, chParams, s_detectorParams);
}


int main(int argc, char** argv) {

  if (argc < 2) {
    printf("usage: %s filename\n", argv[0]);
    return -1;
  }

  const char* filename = argv[1];

  // Initialize ChAruCo data on first use
  s_charucoBoard = new cv::aruco::CharucoBoard(cv::Size(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY), s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, charucoDictionary());

  // Set some default detection parameters
  s_detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; // Enable subpixel refinement for higher precision

  // Cache board points
  std::vector<cv::Point3f> boardPoints = s_charucoBoard->getChessboardCorners();
  const size_t totalCorners = s_charucoBoard->getChessboardSize().width * s_charucoBoard->getChessboardSize().height;

  // Load image
  cv::Mat inputImage = cv::imread(filename, cv::IMREAD_GRAYSCALE);
  cv::Mat image;
  cv::resize(/*src=*/ inputImage, /*dst=*/ image, cv::Size(640, 360));
  printf("Image: %u x %u\n", image.cols, image.rows);

  for (size_t run = 0; run < 10; ++run) {
    PerfTimer perfTimer;

    auto detector = createCharucoDetector();

    std::vector<int> objectPointIds; // IDs of points on the calibration target
    std::vector<cv::Point3f> objectPoints; // Points in object space, on the calibration target
    std::vector<cv::Point2f> imagePoints; // Points in image space

    cv::Mat currentCharucoCorners;
    std::vector<int> currentCharucoIds;
    detector.detectBoard(image, currentCharucoCorners, currentCharucoIds);

    bool found = (currentCharucoIds.size() >= (totalCorners / 3));
    if (found) {
     
      objectPoints.reserve(currentCharucoIds.size()); 
      imagePoints.reserve(currentCharucoIds.size());

      // Convert currentCharucoCorners mat to vector<Point2f>
      assert(currentCharucoCorners.cols == 1);
      for (size_t i = 0; i < currentCharucoCorners.rows; ++i) {
        imagePoints.push_back(cv::Point2f(currentCharucoCorners.ptr<cv::Point2f>()[i]));
      }

      // Extract object points from the board definition

      for (size_t pointIdx = 0; pointIdx < currentCharucoIds.size(); ++pointIdx) {
        const cv::Point3f& boardPoint = boardPoints[currentCharucoIds[pointIdx]];
        objectPoints.push_back(boardPoint);
      }

      // Save point ids
      objectPointIds = std::move(currentCharucoIds);
    }

    // Time end (don't count the printf())
    printf("Detection done in %.3f ms\n\n", perfTimer.checkpoint());
    printf("Found=%u cornerCount=%zu\n", found, currentCharucoCorners.total());
  }

  return 0;
}

