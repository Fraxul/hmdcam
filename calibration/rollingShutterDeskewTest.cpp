// Offline rolling-shutter de-skew visualization harness.
//
// Replays a CalibrationWriter capture (a directory with imu.csv at the root and
// camera<N>/<timestamp-ns>.pgm frames) through the same rolling-shutter de-skew math
// the live pipeline uses (CameraSystem::processFrame), and writes out, per selected
// frame: the plain undistorted image (no RS correction) and the de-skewed image.
//
// The dataset stores absolute nanosecond timestamps for both frames and IMU samples
// (a shared, zeroed timebase) rather than the per-frame line offsets the live IMU
// service produces. So this harness integrates the IMU gyro stream in absolute time
// and converts each output row to an absolute exposure time via the rolling-shutter
// line-delay model -- but every convention-critical step (R_imu_to_cam application,
// gyro-bias subtraction, frame-center reference, exp(+/-theta) rotation, readout-flip
// handling, and the per-row R_y * iR product) is identical to the live path, and the
// exp-map / IMU-rotation primitives are literally the shared header those paths call.
//
// The sign of R_y -- exp(-theta) vs exp(+theta) -- is the de-skew convention under
// test, so this tool renders BOTH polarities. Whichever one straightens a real
// vertical edge during a fast head-turn is correct; the wrong polarity doubles the
// skew. Reality is the oracle here, not agreement with the live code.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <glm/glm.hpp>

#include "common/glmCvInterop.h"
#include "common/RollingShutterDeskew.h"

namespace fs = std::filesystem;

namespace {

constexpr double kDefaultLineDelaySeconds = 1.0 / 112500.0; // 1250 lines/frame * 90 fps.
constexpr float kDegToRad = static_cast<float>(M_PI / 180.0);

struct ImuSample {
  double timeSeconds = 0.0;
  glm::vec3 gyroDPS = glm::vec3(0.0f); // raw, deg/s
  glm::vec3 omegaCam = glm::vec3(0.0f); // bias-compensated, rotated to camera frame, rad/s
};

struct FrameRecord {
  uint64_t timestampNs = 0;
  fs::path path;
};

struct CameraCalibration {
  cv::Mat intrinsicMatrix; // source camera K (3x3, double)
  cv::Mat distCoeffs; // Brown-Conrady k1,k2,p1,p2,k3 (5x1, double)
  bool readoutBottomToTop = false;
  bool imuValid = false;
  glm::mat3 imuToCameraRotation = glm::mat3(1.0f);
  glm::vec3 gyroBias = glm::vec3(0.0f);
};

// ---- argument parsing -------------------------------------------------------

struct Args {
  std::string datasetDir;
  int cameraIndex = 0;
  int topCount = 3;
  std::vector<int> explicitFrames;
  std::string outDir;
  std::string calibrationPath = "calibration.yml";
  std::string imuCalibrationPath = "imuCalibration.yml";
  double alpha = 0.25;
  double lineDelaySeconds = kDefaultLineDelaySeconds;
  bool selfTest = false;
};

std::vector<int> parseIntList(const std::string& s) {
  std::vector<int> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string token = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!token.empty())
      out.push_back(std::stoi(token));
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  return out;
}

bool parseArgs(int argc, char** argv, Args& outArgs) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", name);
        return std::string();
      }
      return argv[++i];
    };

    if (arg == "--camera")
      outArgs.cameraIndex = std::stoi(next("--camera"));
    else if (arg == "--top")
      outArgs.topCount = std::stoi(next("--top"));
    else if (arg == "--frames")
      outArgs.explicitFrames = parseIntList(next("--frames"));
    else if (arg == "--out")
      outArgs.outDir = next("--out");
    else if (arg == "--calibration")
      outArgs.calibrationPath = next("--calibration");
    else if (arg == "--imu-calibration")
      outArgs.imuCalibrationPath = next("--imu-calibration");
    else if (arg == "--alpha")
      outArgs.alpha = std::stod(next("--alpha"));
    else if (arg == "--line-delay")
      outArgs.lineDelaySeconds = std::stod(next("--line-delay"));
    else if (arg == "--self-test")
      outArgs.selfTest = true;
    else if (arg[0] == '-') {
      fprintf(stderr, "Unknown option: %s\n", arg.c_str());
      return false;
    } else if (outArgs.datasetDir.empty())
      outArgs.datasetDir = arg;
    else {
      fprintf(stderr, "Unexpected positional argument: %s\n", arg.c_str());
      return false;
    }
  }

  if (outArgs.selfTest) {
    if (outArgs.outDir.empty())
      outArgs.outDir = "/tmp/rolling-shutter-self-test";
    return true;
  }

  if (outArgs.datasetDir.empty()) {
    fprintf(stderr,
      "Usage: rolling-shutter-deskew-test <datasetDir> [options]\n"
      "  --self-test           run the synthetic ground-truth sign check (no dataset needed)\n"
      "  --camera N            camera stream index (default 0)\n"
      "  --top N               auto-select top N frames by peak angular rate (default 3)\n"
      "  --frames i,j,k        explicit frame indices (overrides --top)\n"
      "  --out DIR             output directory (default <datasetDir>/deskew-test)\n"
      "  --calibration PATH    intrinsics file (default calibration.yml)\n"
      "  --imu-calibration P   IMU extrinsics file (default imuCalibration.yml)\n"
      "  --alpha A             getOptimalNewCameraMatrix alpha (default 0.25)\n"
      "  --line-delay SECONDS  per-line readout time (default 1/112500)\n");
    return false;
  }
  if (outArgs.outDir.empty())
    outArgs.outDir = (fs::path(outArgs.datasetDir) / "deskew-test").string();
  return true;
}

// ---- loaders ----------------------------------------------------------------

bool loadCameraCalibration(const Args& args, CameraCalibration& outCalibration) {
  cv::FileStorage fs(args.calibrationPath, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    fprintf(stderr, "Could not open calibration file %s\n", args.calibrationPath.c_str());
    return false;
  }
  cv::FileNode cameras = fs["cameras"];
  if (!cameras.isSeq() || args.cameraIndex >= static_cast<int>(cameras.size())) {
    fprintf(stderr, "calibration.yml has no camera index %d\n", args.cameraIndex);
    return false;
  }
  cv::FileNode c = cameras[args.cameraIndex];
  c["intrinsicMatrix"] >> outCalibration.intrinsicMatrix;
  c["distortionCoeffs"] >> outCalibration.distCoeffs;
  cv::read(c["readoutBottomToTop"], outCalibration.readoutBottomToTop, false);
  if (outCalibration.intrinsicMatrix.empty() || outCalibration.distCoeffs.empty()) {
    fprintf(stderr, "Camera %d is missing intrinsic calibration\n", args.cameraIndex);
    return false;
  }

  cv::FileStorage imuFs(args.imuCalibrationPath, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!imuFs.isOpened()) {
    fprintf(stderr, "WARNING: could not open %s -- de-skew will be identity (no IMU extrinsics).\n",
      args.imuCalibrationPath.c_str());
    return true;
  }
  cv::FileNode imuCameras = imuFs["cameras"];
  if (imuCameras.isSeq() && args.cameraIndex < static_cast<int>(imuCameras.size())) {
    cv::FileNode ic = imuCameras[args.cameraIndex];
    ic["valid"] >> outCalibration.imuValid;
    cv::Mat rMat, biasMat;
    ic["R_imu_to_cam_matrix"] >> rMat;
    ic["gyro_bias_deg_s"] >> biasMat;
    if (!rMat.empty())
      outCalibration.imuToCameraRotation = glmMat3FromCVMatrix(rMat);
    if (!biasMat.empty())
      outCalibration.gyroBias = glmVec3FromCV(biasMat);
    if (!outCalibration.imuValid)
      fprintf(stderr, "WARNING: camera %d IMU calibration is marked invalid.\n", args.cameraIndex);
  }
  return true;
}

// Parse imu.csv: timestamp(ns),gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z.
bool loadImuSamples(const std::string& path, const CameraCalibration& calibration,
  std::vector<ImuSample>& outSamples) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) {
    fprintf(stderr, "Could not open IMU log %s\n", path.c_str());
    return false;
  }
  char line[512];
  bool firstLine = true;
  while (fgets(line, sizeof(line), f)) {
    if (firstLine) {
      firstLine = false;
      continue;
    } // header row
    unsigned long long tNs = 0;
    double gx, gy, gz, ax, ay, az;
    if (sscanf(line, "%llu,%lf,%lf,%lf,%lf,%lf,%lf", &tNs, &gx, &gy, &gz, &ax, &ay, &az) != 7)
      continue;
    ImuSample s;
    s.timeSeconds = static_cast<double>(tNs) * 1.0e-9;
    s.gyroDPS = glm::vec3(static_cast<float>(gx), static_cast<float>(gy), static_cast<float>(gz));
    s.omegaCam =
      RollingShutterDeskew::applyImuToCameraRotation(calibration.imuToCameraRotation, s.gyroDPS - calibration.gyroBias) * kDegToRad;
    outSamples.push_back(s);
  }
  fclose(f);
  return !outSamples.empty();
}

std::vector<FrameRecord> loadFrameList(const std::string& datasetDir, int cameraIndex) {
  std::vector<FrameRecord> frames;
  fs::path cameraDir = fs::path(datasetDir) / ("camera" + std::to_string(cameraIndex));
  if (!fs::is_directory(cameraDir)) {
    fprintf(stderr, "No such directory: %s\n", cameraDir.c_str());
    return frames;
  }
  for (const auto& entry : fs::directory_iterator(cameraDir)) {
    if (entry.path().extension() != ".pgm")
      continue;
    FrameRecord r;
    r.timestampNs = std::stoull(entry.path().stem().string());
    r.path = entry.path();
    frames.push_back(r);
  }
  std::sort(frames.begin(), frames.end(),
    [](const FrameRecord& a, const FrameRecord& b) { return a.timestampNs < b.timestampNs; });
  return frames;
}

// ---- integration ------------------------------------------------------------

// Cumulative camera-frame rotation vector (radians) at each IMU sample, integrated
// trapezoidally from the first sample. Stored in double: absolute magnitude grows over
// the whole capture, but we only ever read DIFFERENCES across one frame, which
// telescope exactly regardless of the running total -- double keeps that subtraction clean.
struct CumulativeRotation {
  std::vector<double> times;
  std::vector<glm::dvec3> theta;

  void build(const std::vector<ImuSample>& samples) {
    times.resize(samples.size());
    theta.resize(samples.size());
    glm::dvec3 acc(0.0);
    for (size_t i = 0; i < samples.size(); ++i) {
      if (i > 0) {
        const double dt = samples[i].timeSeconds - samples[i - 1].timeSeconds;
        const glm::dvec3 omegaPrev(samples[i - 1].omegaCam);
        const glm::dvec3 omegaCur(samples[i].omegaCam);
        acc += 0.5 * dt * (omegaPrev + omegaCur);
      }
      theta[i] = acc;
      times[i] = samples[i].timeSeconds;
    }
  }

  glm::dvec3 at(double t) const {
    if (t <= times.front()) return theta.front();
    if (t >= times.back()) return theta.back();
    const size_t b = std::upper_bound(times.begin(), times.end(), t) - times.begin();
    const size_t a = b - 1;
    const double span = times[b] - times[a];
    const double frac = (span > 0.0) ? (t - times[a]) / span : 0.0;
    return glm::mix(theta[a], theta[b], frac);
  }
};

// ---- per-frame processing ---------------------------------------------------

// Replicates UndistortRectifyKernel.cu on the CPU: builds absolute-pixel remap maps so
// cv::remap reproduces what the GPU kernel writes into the distortion map + sampling
// pass. `perRowMatrix(y, M)` provides the row-major 3x3 M_y = R_y * iR for output row y.
template <typename RowMatrixFn>
void buildRemapMaps(int width, int height, const CameraCalibration& calibration,
  const RowMatrixFn& perRowMatrix, cv::Mat& outMapX, cv::Mat& outMapY) {
  const float fx = static_cast<float>(calibration.intrinsicMatrix.at<double>(0, 0));
  const float fy = static_cast<float>(calibration.intrinsicMatrix.at<double>(1, 1));
  const float cx = static_cast<float>(calibration.intrinsicMatrix.at<double>(0, 2));
  const float cy = static_cast<float>(calibration.intrinsicMatrix.at<double>(1, 2));
  const float k1 = static_cast<float>(calibration.distCoeffs.at<double>(0, 0));
  const float k2 = static_cast<float>(calibration.distCoeffs.at<double>(1, 0));
  const float p1 = static_cast<float>(calibration.distCoeffs.at<double>(2, 0));
  const float p2 = static_cast<float>(calibration.distCoeffs.at<double>(3, 0));
  const float k3 = static_cast<float>(calibration.distCoeffs.at<double>(4, 0));

  outMapX.create(height, width, CV_32F);
  outMapY.create(height, width, CV_32F);

  for (int y = 0; y < height; ++y) {
    float m[9];
    perRowMatrix(y, m);
    float* mapXRow = outMapX.ptr<float>(y);
    float* mapYRow = outMapY.ptr<float>(y);
    for (int x = 0; x < width; ++x) {
      const float u = static_cast<float>(x);
      const float v = static_cast<float>(y);
      const float xR = (u * m[0]) + (v * m[1]) + m[2];
      const float yR = (u * m[3]) + (v * m[4]) + m[5];
      const float wR = (u * m[6]) + (v * m[7]) + m[8];
      const float invW = 1.0f / wR;
      const float xn = xR * invW;
      const float yn = yR * invW;

      const float r2 = (xn * xn) + (yn * yn);
      const float radial = 1.0f + (r2 * (k1 + (r2 * (k2 + (r2 * k3)))));
      const float xy2 = 2.0f * xn * yn;
      const float xd = (xn * radial) + (p1 * xy2) + (p2 * (r2 + (2.0f * xn * xn)));
      const float yd = (yn * radial) + (p2 * xy2) + (p1 * (r2 + (2.0f * yn * yn)));

      mapXRow[x] = (fx * xd) + cx;
      mapYRow[x] = (fy * yd) + cy;
    }
  }
}

// Multiply row-major 3x3: out = a * b.
void matMul3x3(const float a[9], const float b[9], float out[9]) {
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out[(r * 3) + c] = (a[(r * 3) + 0] * b[(0 * 3) + c]) + (a[(r * 3) + 1] * b[(1 * 3) + c]) + (a[(r * 3) + 2] * b[(2 * 3) + c]);
}

// Per-output-row matrix functor for buildRemapMaps: M_y = R_y * iR, with
// R_y = exp(signMode * (theta(row) - theta(reference))). signMode 0 yields identity R_y
// (plain undistort, no rolling-shutter correction); -1 is the validated de-skew polarity
// (matches CameraSystem::processFrame); +1 doubles the skew. Both the dataset replay and
// the synthetic self-test drive buildRemapMaps through this one functor so they cannot
// diverge in the convention-critical per-row math.
struct DeskewRowMatrix {
  const float* iR;
  const CumulativeRotation* cumulative;
  int height;
  bool readoutBottomToTop;
  double lineDelaySeconds;
  double tFrame;
  glm::dvec3 thetaReference;
  int signMode;

  void operator()(int y, float outM[9]) const {
    if (signMode == 0) {
      std::memcpy(outM, iR, sizeof(float) * 9);
      return;
    }
    // Output row -> sensor row (1:1 here) -> readout line, honoring readout direction.
    const float readoutLine = readoutBottomToTop
      ? static_cast<float>((height - 1) - y)
      : static_cast<float>(y);
    const double tRow = tFrame + (static_cast<double>(readoutLine) * lineDelaySeconds);
    const glm::vec3 thetaRow = glm::vec3(cumulative->at(tRow) - thetaReference);
    float ry[9];
    RollingShutterDeskew::rotationVectorToMatrix(static_cast<float>(signMode) * thetaRow, ry);
    matMul3x3(ry, iR, outM);
  }
};

// row-major inverse of a 3x3 K stored as cv::Mat (double) into float[9].
void invertInto(const cv::Mat& k, float out[9]) {
  cv::Mat inv = k.inv();
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out[(r * 3) + c] = static_cast<float>(inv.at<double>(r, c));
}

// Label a single greyscale panel and draw a true-vertical reference line, for montages.
cv::Mat labelPanel(const cv::Mat& grey, const std::string& text) {
  cv::Mat bgr;
  cv::cvtColor(grey, bgr, cv::COLOR_GRAY2BGR);
  cv::rectangle(bgr, cv::Point(0, 0), cv::Point(bgr.cols - 1, 34), cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(bgr, text, cv::Point(8, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
  cv::line(bgr, cv::Point(bgr.cols / 2, 0), cv::Point(bgr.cols / 2, bgr.rows), cv::Scalar(0, 255, 0), 1);
  return bgr;
}

// Synthetic ground-truth sign check. Builds a straight checkerboard U, shears it into a
// rolling-shutter capture C with a HAND-DERIVED forward model (pure camera-Y yaw: as the
// camera yaws right, later-readout rows show content shifted left -> C row y = U sampled
// at column x + fx*tan(phi(y))), then de-skews C through the same code the dataset path
// uses. The correct polarity reconstructs U; the wrong one doubles the distortion.
// Returns true if exp(-theta) is the clear winner.
bool runSelfTest(const std::string& outDir) {
  const int width = 1280, height = 720;
  const double fx = 800.0, fy = 800.0, cx = 640.0, cy = 360.0;
  const double lineDelaySeconds = kDefaultLineDelaySeconds;
  const double omegaDPS = 400.0; // pure yaw about camera Y (y-down)
  const double omegaRad = omegaDPS * (M_PI / 180.0);
  const double tFrame = 1.0;
  const double tReference = tFrame + (static_cast<double>(height - 1) * 0.5 * lineDelaySeconds);

  CameraCalibration calibration;
  calibration.intrinsicMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  calibration.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
  calibration.readoutBottomToTop = false;
  calibration.imuValid = true; // R_imu_to_cam = I, bias = 0 -> isolates the exp-map sign.

  // Straight reference scene U: checkerboard with bold true-vertical rulings.
  cv::Mat U(height, width, CV_8U, cv::Scalar(0));
  const int sq = 80;
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      if ((((x / sq) + (y / sq)) % 2) == 0)
        U.at<uchar>(y, x) = 200;
  for (int x = 0; x < width; x += sq)
    cv::line(U, cv::Point(x, 0), cv::Point(x, height), cv::Scalar(255), 2);

  // Forward rolling-shutter model (physics, independent of the de-skew code).
  cv::Mat mapX(height, width, CV_32F), mapY(height, width, CV_32F);
  for (int y = 0; y < height; ++y) {
    const double phi = omegaRad * ((tFrame + (y * lineDelaySeconds)) - tReference);
    const float shift = static_cast<float>(fx * std::tan(phi));
    for (int x = 0; x < width; ++x) {
      mapX.at<float>(y, x) = static_cast<float>(x) + shift;
      mapY.at<float>(y, x) = static_cast<float>(y);
    }
  }
  cv::Mat captured;
  cv::remap(U, captured, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_REFLECT);

  // Constant pure-yaw gyro over the readout window, run through the shared rotation.
  std::vector<ImuSample> samples;
  const int sampleCount = 80;
  const double t0 = tFrame - 0.002, t1 = tFrame + (height * lineDelaySeconds) + 0.002;
  for (int k = 0; k < sampleCount; ++k) {
    ImuSample s;
    s.timeSeconds = t0 + ((t1 - t0) * k / (sampleCount - 1));
    s.gyroDPS = glm::vec3(0.0f, static_cast<float>(omegaDPS), 0.0f);
    s.omegaCam = RollingShutterDeskew::applyImuToCameraRotation(calibration.imuToCameraRotation, s.gyroDPS - calibration.gyroBias) * kDegToRad;
    samples.push_back(s);
  }
  CumulativeRotation cumulative;
  cumulative.build(samples);

  float iR[9];
  invertInto(calibration.intrinsicMatrix, iR); // zero distortion -> newK == K -> iR = inv(K)
  const glm::dvec3 thetaReference = cumulative.at(tReference);
  auto makeRow = [&](int signMode) {
    return DeskewRowMatrix{iR, &cumulative, height, calibration.readoutBottomToTop, lineDelaySeconds, tFrame, thetaReference, signMode};
  };

  cv::Mat mx, my, deskewNeg, deskewPos;
  buildRemapMaps(width, height, calibration, makeRow(-1), mx, my);
  cv::remap(captured, deskewNeg, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  buildRemapMaps(width, height, calibration, makeRow(+1), mx, my);
  cv::remap(captured, deskewPos, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

  // RMS vs ground truth over a central crop (avoid reflect-border edge artifacts).
  const cv::Rect crop(200, 120, 880, 480);
  auto rms = [&](const cv::Mat& m) {
    cv::Mat d;
    cv::absdiff(m(crop), U(crop), d);
    d.convertTo(d, CV_32F);
    return std::sqrt(cv::mean(d.mul(d))[0]);
  };
  const double rmsCaptured = rms(captured);
  const double rmsNeg = rms(deskewNeg);
  const double rmsPos = rms(deskewPos);

  fs::create_directories(outDir);
  cv::Mat top, bot, grid;
  cv::hconcat(labelPanel(U, "U: ground truth (straight)"),
    labelPanel(captured, "captured: rolling-shutter shear"), top);
  cv::hconcat(labelPanel(deskewNeg, "deskew exp(-theta): expect STRAIGHT"),
    labelPanel(deskewPos, "deskew exp(+theta): expect doubled"), bot);
  cv::vconcat(top, bot, grid);
  const std::string montagePath = (fs::path(outDir) / "self_test_montage.png").string();
  cv::imwrite(montagePath, grid);

  printf("Rolling-shutter de-skew self-test (synthetic ground truth, pure %.0f deg/s yaw):\n", omegaDPS);
  printf("  RMS vs ground truth U over central crop:\n");
  printf("    captured (uncorrected) = %6.2f\n", rmsCaptured);
  printf("    deskew exp(-theta)     = %6.2f   <- validated polarity\n", rmsNeg);
  printf("    deskew exp(+theta)     = %6.2f\n", rmsPos);
  printf("  Montage: %s\n", montagePath.c_str());

  const bool pass = (rmsNeg < 0.5 * rmsCaptured) && (rmsPos > rmsCaptured);
  printf("  RESULT: %s -- exp(-theta) %s the distortion; exp(+theta) %s it.\n",
    pass ? "PASS" : "FAIL",
    pass ? "removes" : "does NOT clearly remove",
    (rmsPos > rmsCaptured) ? "worsens" : "does not worsen");
  return pass;
}

} // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parseArgs(argc, argv, args))
    return 1;

  if (args.selfTest)
    return runSelfTest(args.outDir) ? 0 : 2;

  CameraCalibration calibration;
  if (!loadCameraCalibration(args, calibration))
    return 1;

  std::vector<ImuSample> imuSamples;
  if (!loadImuSamples((fs::path(args.datasetDir) / "imu.csv").string(), calibration, imuSamples)) {
    fprintf(stderr, "Failed to load IMU samples.\n");
    return 1;
  }

  std::vector<FrameRecord> frames = loadFrameList(args.datasetDir, args.cameraIndex);
  if (frames.empty()) {
    fprintf(stderr, "No frames found for camera %d.\n", args.cameraIndex);
    return 1;
  }

  CumulativeRotation cumulative;
  cumulative.build(imuSamples);

  // Determine output image size from the first frame.
  cv::Mat firstFrame = cv::imread(frames.front().path.string(), cv::IMREAD_GRAYSCALE);
  if (firstFrame.empty()) {
    fprintf(stderr, "Could not read %s\n", frames.front().path.c_str());
    return 1;
  }
  const int width = firstFrame.cols;
  const int height = firstFrame.rows;
  const cv::Size imageSize(width, height);

  // New camera matrix for the undistorted output, matching the live intrinsic path
  // (CameraSystem::updateCameraIntrinsicDistortionParameters). iR = inv(newK) since the
  // mono visualization applies no rectification rotation.
  cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
    calibration.intrinsicMatrix, calibration.distCoeffs, imageSize, args.alpha, imageSize,
    nullptr, /*centerPrincipalPoint=*/ true);
  cv::Mat iRMat = newCameraMatrix.inv();
  float iR[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      iR[(r * 3) + c] = static_cast<float>(iRMat.at<double>(r, c));

  // Peak angular rate (deg/s) over each frame's readout window, for ranking/diagnostics.
  const double readoutWindowSeconds = static_cast<double>(height - 1) * args.lineDelaySeconds;
  auto peakRateForFrame = [&](const FrameRecord& frame) -> double {
    const double tStart = static_cast<double>(frame.timestampNs) * 1.0e-9;
    const double tEnd = tStart + readoutWindowSeconds;
    double peak = 0.0;
    for (const ImuSample& s : imuSamples) {
      if (s.timeSeconds < tStart) continue;
      if (s.timeSeconds > tEnd) break;
      // Report the body-frame rate magnitude in deg/s (bias removed).
      const double rate = glm::length(s.omegaCam) / static_cast<double>(kDegToRad);
      peak = std::max(peak, rate);
    }
    return peak;
  };

  // Select which frames to process.
  std::vector<int> selected;
  if (!args.explicitFrames.empty()) {
    for (int idx : args.explicitFrames) {
      if (idx >= 0 && idx < static_cast<int>(frames.size()))
        selected.push_back(idx);
      else
        fprintf(stderr, "Ignoring out-of-range frame index %d (have %zu frames)\n", idx, frames.size());
    }
  } else {
    std::vector<std::pair<double, int>> ranked;
    for (int i = 0; i < static_cast<int>(frames.size()); ++i)
      ranked.emplace_back(peakRateForFrame(frames[i]), i);
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int i = 0; i < args.topCount && i < static_cast<int>(ranked.size()); ++i)
      selected.push_back(ranked[i].second);
    std::sort(selected.begin(), selected.end());
  }

  if (selected.empty()) {
    fprintf(stderr, "No frames selected.\n");
    return 1;
  }

  fs::create_directories(args.outDir);
  printf("Dataset: %s | camera %d | %dx%d | %zu frames | %zu IMU samples\n",
    args.datasetDir.c_str(), args.cameraIndex, width, height, frames.size(), imuSamples.size());
  printf("readoutBottomToTop=%d | IMU calibration valid=%d | line delay=%.3f us\n",
    calibration.readoutBottomToTop, calibration.imuValid, args.lineDelaySeconds * 1.0e6);
  printf("Output directory: %s\n\n", args.outDir.c_str());

  for (int frameIndex : selected) {
    const FrameRecord& frame = frames[frameIndex];
    cv::Mat src = cv::imread(frame.path.string(), cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
      fprintf(stderr, "Could not read %s -- skipping.\n", frame.path.c_str());
      continue;
    }

    const double tFrame = static_cast<double>(frame.timestampNs) * 1.0e-9;
    const double tReference = tFrame + (static_cast<double>(height - 1) * 0.5 * args.lineDelaySeconds);
    const glm::dvec3 thetaReference = cumulative.at(tReference);

    // Per-output-row matrix M_y = R_y * iR via the shared functor. signMode 0 -> identity
    // R_y (plain undistort), -1 -> validated de-skew polarity, +1 -> doubled (wrong).
    auto makeRow = [&](int signMode) {
      return DeskewRowMatrix{iR, &cumulative, height, calibration.readoutBottomToTop,
        args.lineDelaySeconds, tFrame, thetaReference, signMode};
    };

    cv::Mat mapX, mapY, undistorted, deskewNeg, deskewPos;
    buildRemapMaps(width, height, calibration, makeRow(0), mapX, mapY);
    cv::remap(src, undistorted, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    buildRemapMaps(width, height, calibration, makeRow(-1), mapX, mapY);
    cv::remap(src, deskewNeg, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    buildRemapMaps(width, height, calibration, makeRow(+1), mapX, mapY);
    cv::remap(src, deskewPos, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    // Overlays: baseline (skewed undistort) in green, candidate de-skew in red. Where the
    // de-skew shifts a row horizontally, edges separate into a red/green fringe; the
    // correct polarity is the one that pulls a real vertical edge straight relative to the
    // green baseline (and the wrong one visibly worsens it).
    auto makeOverlay = [&](const cv::Mat& deskew) {
      std::vector<cv::Mat> channels = {cv::Mat::zeros(src.size(), CV_8U), undistorted, deskew}; // B,G,R
      cv::Mat overlay;
      cv::merge(channels, overlay);
      return overlay;
    };

    char stem[64];
    snprintf(stem, sizeof(stem), "frame%04d", frameIndex);
    const fs::path outBase = fs::path(args.outDir) / stem;
    cv::imwrite(outBase.string() + "_baseline.png", undistorted);
    cv::imwrite(outBase.string() + "_deskew_neg.png", deskewNeg);
    cv::imwrite(outBase.string() + "_deskew_pos.png", deskewPos);
    cv::imwrite(outBase.string() + "_overlay_neg.png", makeOverlay(deskewNeg));
    cv::imwrite(outBase.string() + "_overlay_pos.png", makeOverlay(deskewPos));

    // Quantitative: max per-row source-pixel displacement (approx, both axes) at the
    // center column between the validated de-skew and the baseline. Reported per axis
    // because this rig's correction is often dominated by the vertical component.
    const float fxPix = static_cast<float>(calibration.intrinsicMatrix.at<double>(0, 0));
    const float fyPix = static_cast<float>(calibration.intrinsicMatrix.at<double>(1, 1));
    float maxShiftXPx = 0.0f, maxShiftYPx = 0.0f;
    const int centerX = width / 2;
    auto baseRow = makeRow(0);
    auto negRow = makeRow(-1);
    for (int y = 0; y < height; ++y) {
      float baseM[9], negM[9];
      baseRow(y, baseM);
      negRow(y, negM);
      auto ray = [&](const float m[9], float& xn, float& yn) {
        const float u = static_cast<float>(centerX), v = static_cast<float>(y);
        const float invW = 1.0f / ((u * m[6]) + (v * m[7]) + m[8]);
        xn = ((u * m[0]) + (v * m[1]) + m[2]) * invW;
        yn = ((u * m[3]) + (v * m[4]) + m[5]) * invW;
      };
      float bxn, byn, nxn, nyn;
      ray(baseM, bxn, byn);
      ray(negM, nxn, nyn);
      maxShiftXPx = std::max(maxShiftXPx, std::fabs(nxn - bxn) * fxPix);
      maxShiftYPx = std::max(maxShiftYPx, std::fabs(nyn - byn) * fyPix);
    }

    printf("frame %4d  t=%.4f s  peakRate=%7.1f deg/s  max correction at center col ~ x=%.1fpx y=%.1fpx\n",
      frameIndex, tFrame, peakRateForFrame(frame), maxShiftXPx, maxShiftYPx);
  }

  printf("\nDone. Inspect *_overlay_neg.png vs *_overlay_pos.png: the correct polarity straightens\n"
         "real vertical edges relative to the green baseline; the wrong one worsens the skew.\n");
  return 0;
}
