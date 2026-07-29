#pragma once
#include "Estimator.h"
#include <string>
#include <vector>

namespace CameraImuCalib {

// Write the detailed per-camera YAML result with an explicit conventions header, and a
// plotting CSV of the aligned magnitude series and the per-axis money plot.
// lineDelaySeconds (the sensor line duration from RollingShutterTiming) converts the solved
// time offset into the line-domain form the base system consumes.
bool writeResults(const std::string& yamlPath, const std::string& plotCsvPath,
  const EstimatorResult& result, int cameraIndex, double lineDelaySeconds);

// Print a human-readable summary (and any warnings) to stdout.
void printResultSummary(const EstimatorResult& result);

// One camera's entry in the consolidated, production-ready imuCalibration.yml.
struct CameraCalibration {
  int cameraIndex = -1;
  bool valid = false; // false if estimation failed for this camera
  double lineDelaySeconds = 0.0; // sensor line duration used to express the time offset in lines
  EstimatorResult result;
};

// Write the consolidated imuCalibration.yml the base system consumes: a top-level `cameras`
// sequence (indexed to match calibration.yml) carrying R_imu_to_cam_matrix,
// time_offset_delta_s and gyro_bias_deg_s per camera. Entries are written in cameraIndex
// order; gaps are filled with valid:0 placeholders so the array stays index-aligned.
bool writeImuCalibrationFile(const std::string& path, const std::vector<CameraCalibration>& cameras);

} // namespace CameraImuCalib
