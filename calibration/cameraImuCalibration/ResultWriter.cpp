#include "ResultWriter.h"
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <algorithm>
#include <cstdio>

namespace CameraImuCalib {

namespace {
  cv::Mat rotationMatrix3x3(const Eigen::Quaterniond& q) {
    const Eigen::Matrix3d m = q.toRotationMatrix();
    cv::Mat r(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r.at<double>(i, j) = m(i, j);
    return r;
  }
  cv::Mat columnVector3(const Eigen::Vector3d& v) {
    cv::Mat c(3, 1, CV_64F);
    c.at<double>(0) = v.x();
    c.at<double>(1) = v.y();
    c.at<double>(2) = v.z();
    return c;
  }
} // namespace

bool writeResults(const std::string& yamlPath, const std::string& plotCsvPath,
  const EstimatorResult& result, int cameraIndex) {
  cv::FileStorage fs(yamlPath, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    fprintf(stderr, "writeResults: cannot open '%s' for writing\n", yamlPath.c_str());
    return false;
  }

  fs.writeComment(
    "\nCamera-IMU autosync calibration result.\n"
    "Conventions:\n"
    "  Camera frame: x right, y down, z forward (optical axis).\n"
    "  R_imu_to_cam rotates an IMU-frame vector into the camera frame: v_cam = R_imu_to_cam * v_imu.\n"
    "  Time offset: t_imu = t_cam + time_offset_delta_s.\n"
    "  Gyro bias is in deg/s; omega_cam = R_imu_to_cam * (omega_imu - gyro_bias_deg_s).\n",
    0);

  fs << "cameraIndex" << cameraIndex;

  // R_imu_to_cam as quaternion (w, x, y, z) and as a 3x3 matrix.
  const Eigen::Quaterniond& q = result.rImuToCam;
  fs << "R_imu_to_cam_quaternion_wxyz"
     << "[:" << q.w() << q.x() << q.y() << q.z() << "]";
  cv::Mat rMat(3, 3, CV_64F);
  const Eigen::Matrix3d m = result.rImuToCam.toRotationMatrix();
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      rMat.at<double>(r, c) = m(r, c);
  fs << "R_imu_to_cam_matrix" << rMat;

  fs << "time_offset_relation"
     << "t_imu = t_cam + time_offset_delta_s";
  fs << "time_offset_delta_s" << result.deltaSeconds;
  fs << "delta_coarse_s" << result.deltaCoarseSeconds;
  fs << "delta_refined_s" << result.deltaSeconds;

  fs << "gyro_bias_deg_s"
     << "[:" << result.biasDeg.x() << result.biasDeg.y() << result.biasDeg.z() << "]";
  fs << "gyro_bias_refined" << (result.biasRefined ? 1 : 0);

  fs << "residual_rms_deg" << result.residualRmsDeg;
  fs << "residual_rms_per_axis_deg"
     << "[:" << result.perAxisResidualRmsDeg.x()
     << result.perAxisResidualRmsDeg.y() << result.perAxisResidualRmsDeg.z() << "]";
  fs << "pairs_used" << static_cast<int>(result.pairsUsed);
  fs << "pairs_rejected" << static_cast<int>(result.pairsRejected);
  fs << "stage1_median_inlier_ratio" << result.medianInlierRatio;
  fs << "stage2_correlation_peak" << result.stage2CorrelationPeak;
  fs << "converged" << (result.converged ? 1 : 0);

  fs << "warnings"
     << "[";
  for (const std::string& w : result.warnings)
    fs << w;
  fs << "]";
  fs.release();

  // Plotting CSV: aligned magnitudes + per-axis omega_cam vs IMU prediction.
  FILE* csv = fopen(plotCsvPath.c_str(), "w");
  if (csv) {
    fprintf(csv, "t,omega_cam_mag,omega_imu_mag,"
                 "omega_cam_x,omega_cam_y,omega_cam_z,"
                 "omega_imu_pred_x,omega_imu_pred_y,omega_imu_pred_z\n");
    for (const PairPlotRecord& p : result.plot)
      fprintf(csv, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        p.tMid, p.omegaCamMag, p.omegaImuMag,
        p.omegaCam.x(), p.omegaCam.y(), p.omegaCam.z(),
        p.omegaImuPredicted.x(), p.omegaImuPredicted.y(), p.omegaImuPredicted.z());
    fclose(csv);
  } else {
    fprintf(stderr, "writeResults: cannot open plot CSV '%s'\n", plotCsvPath.c_str());
  }

  printf("Wrote results to %s and plot data to %s\n", yamlPath.c_str(), plotCsvPath.c_str());
  return true;
}

void printResultSummary(const EstimatorResult& result) {
  const Eigen::Quaterniond& q = result.rImuToCam;
  printf("\n================ camera-IMU calibration result ================\n");
  printf("R_imu_to_cam (w,x,y,z): %.6f, %.6f, %.6f, %.6f\n", q.w(), q.x(), q.y(), q.z());
  printf("time offset delta     : %.4f ms   (t_imu = t_cam + delta)\n", result.deltaSeconds * 1e3);
  printf("  coarse vs refined   : %.4f ms -> %.4f ms\n",
    result.deltaCoarseSeconds * 1e3, result.deltaSeconds * 1e3);
  printf("gyro bias (deg/s)     : %.4f, %.4f, %.4f  [%s]\n",
    result.biasDeg.x(), result.biasDeg.y(), result.biasDeg.z(),
    result.biasRefined ? "refined (motion)" : "static-segment (fixed)");
  printf("residual RMS          : %.4f deg  (per-axis %.4f, %.4f, %.4f)\n",
    result.residualRmsDeg, result.perAxisResidualRmsDeg.x(),
    result.perAxisResidualRmsDeg.y(), result.perAxisResidualRmsDeg.z());
  printf("pairs used / rejected : %zu / %zu\n", result.pairsUsed, result.pairsRejected);
  printf("Stage-1 median inlier : %.3f   Stage-2 corr peak: %.3f\n",
    result.medianInlierRatio, result.stage2CorrelationPeak);
  printf("converged             : %s\n", result.converged ? "yes" : "NO");

  if (!result.warnings.empty()) {
    printf("\n!!! WARNINGS (%zu) !!!\n", result.warnings.size());
    for (const std::string& w : result.warnings)
      printf("  * %s\n", w.c_str());
  }
  printf("===============================================================\n");
}

bool writeImuCalibrationFile(const std::string& path, const std::vector<CameraCalibration>& cameras) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    fprintf(stderr, "writeImuCalibrationFile: cannot open '%s' for writing\n", path.c_str());
    return false;
  }

  fs.writeComment(
    "\nCamera-IMU calibration for the base system. One entry per camera, indexed to match\n"
    "calibration.yml's 'cameras' sequence. Conventions:\n"
    "  v_cam = R_imu_to_cam_matrix * v_imu        (camera frame: x right, y down, z forward)\n"
    "  t_imu = t_cam + time_offset_delta_s\n"
    "  omega_cam = R_imu_to_cam_matrix * (omega_imu - gyro_bias_deg_s)   [gyro_bias in deg/s]\n",
    0);

  // Emit entries in cameraIndex order, filling any gaps with valid:0 placeholders so the
  // sequence stays index-aligned with calibration.yml.
  int maxIndex = -1;
  for (const CameraCalibration& c : cameras)
    maxIndex = std::max(maxIndex, c.cameraIndex);

  fs << "cameras"
     << "[";
  for (int idx = 0; idx <= maxIndex; ++idx) {
    const CameraCalibration* entry = nullptr;
    for (const CameraCalibration& c : cameras)
      if (c.cameraIndex == idx) {
        entry = &c;
        break;
      }

    fs << "{";
    fs << "cameraIndex" << idx;
    if (entry && entry->valid) {
      fs << "valid" << 1;
      fs << "R_imu_to_cam_matrix" << rotationMatrix3x3(entry->result.rImuToCam);
      fs << "time_offset_delta_s" << entry->result.deltaSeconds;
      fs << "gyro_bias_deg_s" << columnVector3(entry->result.biasDeg);
    } else {
      // Placeholder: identity / zero so the file is still well-formed and index-aligned.
      fs << "valid" << 0;
      fs << "R_imu_to_cam_matrix" << cv::Mat::eye(3, 3, CV_64F);
      fs << "time_offset_delta_s" << 0.0;
      fs << "gyro_bias_deg_s" << cv::Mat::zeros(3, 1, CV_64F);
    }
    fs << "}";
  }
  fs << "]";
  fs.release();
  printf("Wrote consolidated calibration to %s\n", path.c_str());
  return true;
}

} // namespace CameraImuCalib
