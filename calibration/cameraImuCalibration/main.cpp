// Camera-IMU autosync calibrator: estimates R_imu_to_cam, the camera<->IMU time offset
// delta, and a constant gyro bias from a recorded IMU trace and OFA dense optical flow over
// a recorded frame sequence.
//
// Operates on a CalibrationWriter dataset directory (imu.csv + cameraN/*.pgm). By default it
// calibrates every camera in the dataset and writes, INTO that directory:
//   - imuCalibration-cameraN.yml / -plot.csv : detailed per-camera result + plot data
//   - imuCalibration.yml                     : consolidated, production-ready result the base
//                                              system consumes (top-level `cameras` sequence).
// Outputs are written next to their inputs (so results are correlated with the capture) and
// never overwrite the production calibration -- move imuCalibration.yml into place once you
// have confirmed it.

#include "Estimator.h"
#include "FrameSequence.h"
#include "ImuTrace.h"
#include "Intrinsics.h"
#include "ResultWriter.h"
#include "RollingShutterTiming.h"
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef IS_TEGRA
#include "OfaFlowSource.h"
#endif

using namespace CameraImuCalib;

namespace {

void usage(const char* argv0) {
  printf("usage: %s --dataset DIR [options]\n", argv0);
  printf("  --dataset DIR         CalibrationWriter dataset (imu.csv + cameraN/*.pgm)  [required]\n");
  printf("  --calibration PATH    intrinsics YAML                              [default calibration.yml]\n");
  printf("  --camera-index N      calibrate only camera N (default: all cameras in the dataset)\n");
  printf("  --downsample-shift N  OFA processing resolution = full >> N (0=full,1=half,2=quarter) [default 1]\n");
  printf("  --stride N            subsampling of the flow field                 [default 4]\n");
  printf("  --refine-bias         let the optimizer adjust gyro bias during motion (default: use\n");
  printf("                        the static-segment bias, held fixed)\n");
  printf("  --rolling-shutter     Row-timed RS-aware flow fit (time each vector by its row)\n");
  printf("  --rs-alpha-reg F      RS-aware alpha (intra-frame gradient) regularization        [default 1.0]\n");
  printf("\nOutputs are written into DIR. Move DIR/imuCalibration.yml into place once confirmed.\n");
}

const char* argValue(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    fprintf(stderr, "missing value for %s\n", argv[i]);
    exit(2);
  }
  return argv[++i];
}

bool isDirectory(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Calibrate one camera and write its detailed per-camera result into the dataset directory.
// Returns false on a hard failure (missing intrinsics/frames/hardware); a non-converged
// estimate still returns true with out.valid reflecting convergence.
bool calibrateCamera(const std::string& dataset, const std::string& calibration, int cameraIndex,
  const ImuTrace& imu, const EstimatorConfig& config, int downsampleShift,
  int stride, CameraCalibration& out) {
  out.cameraIndex = cameraIndex;
  out.valid = false;

  RollingShutterTiming timing;
  Intrinsics intrinsics;
  if (!intrinsics.load(calibration, cameraIndex, timing))
    return false;

  const std::string imageDir = dataset + "/camera" + std::to_string(cameraIndex);
  FrameSequence frames;
  if (!frames.load(imageDir))
    return false;
  timing.visibleHeight = frames.height(); // line-delay timing uses the actual frame height

#ifdef IS_TEGRA
  OfaFlowSource flow(frames, stride, downsampleShift);
  if (!flow.initialize()) {
    fprintf(stderr, "camera %d: OFA flow source init failed\n", cameraIndex);
    return false;
  }
  Estimator estimator(intrinsics, timing, imu, config);
  if (!estimator.run(flow, out.result)) {
    fprintf(stderr, "camera %d: estimation failed\n", cameraIndex);
    return false;
  }
  printResultSummary(out.result);
  const std::string detail = dataset + "/imuCalibration-camera" + std::to_string(cameraIndex) + ".yml";
  const std::string plot = dataset + "/imuCalibration-camera" + std::to_string(cameraIndex) + "-plot.csv";
  writeResults(detail, plot, out.result, cameraIndex);
  out.valid = out.result.converged;
  return true;
#else
  (void) imu;
  (void) config;
  (void) downsampleShift;
  (void) stride;
  fprintf(stderr, "cameraImuCalibration requires the Tegra OFA flow source (build on Jetson).\n");
  return false;
#endif
}

} // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0); // line-buffered so progress is visible when piped
  int cameraIndex = -1; // -1 = all cameras in the dataset
  int downsampleShift = 1; // OFA processing resolution = full >> shift (1 = half-res)
  int stride = 4; // subsample the flow field (half-res / stride 4 -> ~32k samples)
  std::string dataset, calibration = "calibration.yml";
  EstimatorConfig config;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--dataset"))
      dataset = argValue(argc, argv, i);
    else if (!strcmp(argv[i], "--calibration"))
      calibration = argValue(argc, argv, i);
    else if (!strcmp(argv[i], "--camera-index"))
      cameraIndex = atoi(argValue(argc, argv, i));
    else if (!strcmp(argv[i], "--stride"))
      stride = atoi(argValue(argc, argv, i));
    else if (!strcmp(argv[i], "--downsample-shift"))
      downsampleShift = atoi(argValue(argc, argv, i));
    else if (!strcmp(argv[i], "--max-pairs"))
      config.maxPairs = static_cast<size_t>(atol(argValue(argc, argv, i)));
    else if (!strcmp(argv[i], "--refine-bias"))
      config.refineBias = true;
    else if (!strcmp(argv[i], "--rolling-shutter"))
      config.rollingShutterAware = true;
    else if (!strcmp(argv[i], "--rs-alpha-reg"))
      config.rsAlphaRegularization = atof(argValue(argc, argv, i));
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
  }

  if (dataset.empty()) {
    fprintf(stderr, "--dataset is required\n");
    usage(argv[0]);
    return 2;
  }
  if (!isDirectory(dataset)) {
    fprintf(stderr, "dataset '%s' is not a directory\n", dataset.c_str());
    return 2;
  }

  ImuTrace imu;
  if (!imu.load(dataset + "/imu.csv"))
    return 1;

  // Determine which cameras to process: the requested one, or every cameraN/ present.
  std::vector<int> cameraIndices;
  if (cameraIndex >= 0) {
    cameraIndices.push_back(cameraIndex);
  } else {
    for (int idx = 0; isDirectory(dataset + "/camera" + std::to_string(idx)); ++idx)
      cameraIndices.push_back(idx);
  }
  if (cameraIndices.empty()) {
    fprintf(stderr, "no cameraN/ directories found in '%s'\n", dataset.c_str());
    return 1;
  }

  std::vector<CameraCalibration> results;
  for (int idx : cameraIndices) {
    printf("\n######## calibrating camera %d ########\n", idx);
    CameraCalibration c;
    if (!calibrateCamera(dataset, calibration, idx, imu, config, downsampleShift, stride, c)) {
      c.cameraIndex = idx;
      c.valid = false;
    }
    results.push_back(c);
  }

  // The consolidated file is only (re)written when calibrating all cameras, so a single-camera
  // re-run can't clobber a complete imuCalibration.yml with placeholders.
  if (cameraIndex < 0) {
    writeImuCalibrationFile(dataset + "/imuCalibration.yml", results);
    printf("\nReview the per-camera results, then move %s/imuCalibration.yml into place to use it.\n",
      dataset.c_str());
  } else {
    printf("\nSingle-camera run: wrote the detailed result only. Re-run without --camera-index "
           "to regenerate %s/imuCalibration.yml.\n",
      dataset.c_str());
  }

  bool allValid = true;
  for (const CameraCalibration& c : results)
    allValid = allValid && c.valid;
  return allValid ? 0 : 1;
}
