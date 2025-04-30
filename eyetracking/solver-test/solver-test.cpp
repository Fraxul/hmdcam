// clang++ -Wall -I/usr/include/eigen3 solver-test.cpp -o solver-test -lceres -lglog
#include "ceres/ceres.h"

const size_t kCalibrationSampleColumns = 10;
const double data[] = {
-2.500000, 0.000000, -2.240425, -1.838610, -0.011283, -0.598664, 0.800921, -0.035865, -0.629520, 0.776156,
2.500000, 0.000000, 0.566910, -1.389530, -0.016896, -0.637176, 0.770533, -0.035865, -0.629520, 0.776156,
0.000000, 2.500000, -1.594009, 1.097374, -0.051847, -0.607662, 0.792501, -0.035865, -0.629520, 0.776156,
0.000000, -2.500000, -0.477501, -3.798306, 0.015735, -0.623022, 0.782046, -0.035865, -0.629520, 0.776156,
2.500000, 2.500000, 0.063168, 0.368546, -0.040820, -0.630376, 0.775216, -0.035865, -0.629520, 0.776156,
2.500000, -2.500000, 0.188549, -3.340587, 0.009398, -0.632073, 0.774852, -0.035865, -0.629520, 0.776156,
-2.500000, 2.500000, -3.315910, 0.697156, -0.047354, -0.583524, 0.810714, -0.035865, -0.629520, 0.776156,
-2.500000, -2.500000, -3.436768, -3.625138, 0.013903, -0.581810, 0.813206, -0.035865, -0.629520, 0.776156,
-5.000000, 0.000000, -5.505882, -1.779543, -0.012604, -0.552065, 0.833706, -0.035865, -0.629520, 0.776156,
5.000000, 0.000000, 1.837448, -0.716263, -0.025467, -0.654109, 0.755971, -0.035865, -0.629520, 0.776156,
0.000000, 5.000000, -2.002552, 3.651814, -0.087589, -0.601984, 0.793690, -0.035865, -0.629520, 0.776156,
0.000000, -5.000000, -0.653664, -6.219785, 0.048881, -0.620615, 0.782591, -0.035865, -0.629520, 0.776156,
5.000000, 5.000000, 1.525055, 3.145261, -0.076679, -0.649975, 0.756077, -0.035865, -0.629520, 0.776156,
5.000000, -5.000000, 2.043430, -5.626272, 0.039208, -0.656824, 0.753024, -0.035865, -0.629520, 0.776156,
-5.000000, 5.000000, -5.866627, 1.276733, -0.057273, -0.546804, 0.835299, -0.035865, -0.629520, 0.776156,
-5.000000, -5.000000, -4.765537, -6.785785, 0.059677, -0.562793, 0.824441, -0.035865, -0.629520, 0.776156,
-7.500000, 0.000000, -6.976284, -2.854326, 0.003087, -0.530488, 0.847687, -0.035865, -0.629520, 0.776156,
7.500000, 0.000000, 3.444149, 0.343073, -0.038467, -0.675060, 0.736759, -0.035865, -0.629520, 0.776156,
0.000000, -7.500000, -1.250530, -9.039058, 0.088030, -0.612413, 0.785622, -0.035865, -0.629520, 0.776156,
7.500000, 7.500000, 2.898106, 6.692055, -0.120743, -0.667999, 0.734302, -0.035865, -0.629520, 0.776156,
7.500000, -7.500000, 3.376720, -9.743416, 0.091258, -0.674192, 0.732897, -0.035865, -0.629520, 0.776156,
-7.500000, 7.500000, -7.007923, 3.475174, -0.090417, -0.530019, 0.843151, -0.035865, -0.629520, 0.776156,
-7.500000, -7.500000, -7.292316, -9.029900, 0.094583, -0.525804, 0.845331, -0.035865, -0.629520, 0.776156,
-10.000000, 0.000000, -8.777620, -2.991130, 0.005209, -0.503579, 0.863934, -0.035865, -0.629520, 0.776156,
10.000000, 0.000000, 4.925903, -1.269941, -0.017288, -0.693912, 0.719852, -0.035865, -0.629520, 0.776156,
0.000000, 10.000000, -3.536411, 7.081440, -0.137587, -0.580394, 0.802628, -0.035865, -0.629520, 0.776156,
0.000000, -10.000000, -2.136997, -12.291713, 0.134035, -0.600109, 0.788609, -0.035865, -0.629520, 0.776156,
10.000000, 10.000000, 3.001911, 9.813422, -0.160286, -0.669346, 0.725455, -0.035865, -0.629520, 0.776156,
10.000000, -10.000000, 4.313751, -11.107703, 0.107044, -0.686180, 0.719513, -0.035865, -0.629520, 0.776156,
-10.000000, 10.000000, -9.458723, 4.913236, -0.114428, -0.493273, 0.862315, -0.035865, -0.629520, 0.776156,
-10.000000, -10.000000, -9.011292, -11.049239, 0.126560, -0.500051, 0.856698, -0.035865, -0.629520, 0.776156,
-12.500000, 0.000000, -9.853790, -3.011034, 0.005568, -0.487264, 0.873237, -0.035865, -0.629520, 0.776156,
12.500000, 0.000000, 5.991528, 0.337214, -0.036792, -0.707183, 0.706072, -0.035865, -0.629520, 0.776156,
0.000000, 12.500000, -4.402935, 9.481347, -0.172900, -0.568013, 0.804654, -0.035865, -0.629520, 0.776156,
0.000000, -12.500000, -2.098984, -14.263900, 0.161015, -0.600639, 0.783139, -0.035865, -0.629520, 0.776156,
12.500000, 12.500000, 3.679970, 13.001310, -0.198231, -0.678091, 0.707741, -0.035865, -0.629520, 0.776156,
12.500000, -12.500000, 6.276257, -13.600014, 0.133685, -0.710688, 0.690688, -0.035865, -0.629520, 0.776156,
-12.500000, 12.500000, -11.970865, 5.153850, -0.120869, -0.454672, 0.882420, -0.035865, -0.629520, 0.776156,
-12.500000, -12.500000, -11.137892, -13.869254, 0.172051, -0.467572, 0.867050, -0.035865, -0.629520, 0.776156,
-15.000000, 0.000000, -11.171022, -3.013802, 0.005681, -0.467061, 0.884207, -0.035865, -0.629520, 0.776156,
15.000000, 0.000000, 7.491817, 1.210703, -0.046290, -0.725453, 0.686714, -0.035865, -0.629520, 0.776156,
0.000000, 15.000000, -5.202881, 11.495600, -0.202993, -0.556467, 0.805691, -0.035865, -0.629520, 0.776156,
0.000000, -15.000000, -2.160324, -16.643969, 0.193554, -0.599783, 0.776400, -0.035865, -0.629520, 0.776156,
15.000000, 15.000000, 3.607174, 15.686838, -0.231444, -0.677157, 0.698493, -0.035865, -0.629520, 0.776156,
15.000000, -15.000000, 7.832703, -16.959909, 0.169098, -0.729535, 0.662710, -0.035865, -0.629520, 0.776156,
-15.000000, 15.000000, -13.413198, 5.902794, -0.134052, -0.432109, 0.891802, -0.035865, -0.629520, 0.776156,
-15.000000, -15.000000, -12.904640, -15.436973, 0.198806, -0.440097, 0.875666, -0.035865, -0.629520, 0.776156,
#if 1
-17.500000, 0.000000, -13.223038, -2.793746, 0.002327, -0.435100, 0.900379, -0.035865, -0.629520, 0.776156,
17.500000, 0.000000, 9.432602, 1.687415, -0.050116, -0.748346, 0.661413, -0.035865, -0.629520, 0.776156,
0.000000, 17.500000, -7.085482, 13.928070, -0.242092, -0.528871, 0.813441, -0.035865, -0.629520, 0.776156,
0.000000, -17.500000, -2.791050, -18.340080, 0.218222, -0.590939, 0.776641, -0.035865, -0.629520, 0.776156,
17.500000, 17.500000, 3.969959, 20.864117, -0.291814, -0.681802, 0.670813, -0.035865, -0.629520, 0.776156,
17.500000, -17.500000, 9.832767, -20.707024, 0.204023, -0.752960, 0.625640, -0.035865, -0.629520, 0.776156,
-17.500000, 17.500000, -16.344120, 6.548608, -0.147437, -0.385432, 0.910881, -0.035865, -0.629520, 0.776156,
-17.500000, -17.500000, -15.967962, -18.533813, 0.251910, -0.391482, 0.885033, -0.035865, -0.629520, 0.776156,
-20.000000, 0.000000, -14.648237, -3.846677, 0.019093, -0.412571, 0.910725, -0.035865, -0.629520, 0.776156,
20.000000, 0.000000, 11.449722, 3.116527, -0.063910, -0.771229, 0.633341, -0.035865, -0.629520, 0.776156,
0.000000, 20.000000, -7.639833, 15.984526, -0.272747, -0.520635, 0.809041, -0.035865, -0.629520, 0.776156,
0.000000, -20.000000, -2.941322, -20.264673, 0.244650, -0.588821, 0.770348, -0.035865, -0.629520, 0.776156,
20.000000, 20.000000, 4.905777, 23.154545, -0.313501, -0.693659, 0.648501, -0.035865, -0.629520, 0.776156,
20.000000, -20.000000, 10.646317, -24.782280, 0.243916, -0.762228, 0.599594, -0.035865, -0.629520, 0.776156,
-20.000000, 20.000000, -20.312164, 8.939042, -0.190213, -0.320655, 0.927901, -0.035865, -0.629520, 0.776156,
-20.000000, -20.000000, -18.367348, -20.154463, 0.281528, -0.352615, 0.892415, -0.035865, -0.629520, 0.776156,
#endif
};

const size_t sampleCount = (sizeof(data) / sizeof(data[0])) / kCalibrationSampleColumns;

static inline double degrees(double r) { return r * (180.0 / M_PI); }
static inline double radians(double d) { return d * (M_PI / 180.0); }
template <typename T> static inline T square(T x) { return x * x; }

enum CoeffID {
  kXOffset,
  kYOffset,
  kRollAngle,

  kCoeffCount
};


template <typename T> void anglesToVector(T roll, T pitch, T yaw, T* outVec) {
  // Mathematica: evaluated EulerMatrix[{roll, pitch, yaw}, {3, 1, 2}] . {0, 0, 1}
  // result: {Cos[yaw] Sin[pitch] Sin[roll] + Cos[roll] Sin[yaw], -Cos[roll] Cos[yaw] Sin[pitch] + Sin[roll] Sin[yaw], Cos[pitch] Cos[yaw]}
  outVec[0] = (cos(yaw)*sin(pitch)*sin(roll)) + (cos(roll)*sin(yaw));
  outVec[1] = (-cos(roll)*cos(yaw)*sin(pitch)) + (sin(roll)*sin(yaw));
  outVec[2] = (cos(pitch)*cos(yaw));
}

template <typename T> void vectorToAngles(const T* vec, T& outPitch, T& outYaw) {
  // Note: argument ordering is atan2(y, x)

  // Pitch is rotation around / flattening along the X axis, where the 2d plane is Z, Y
  outPitch = atan2(-vec[1], vec[2]);

  // Yaw is rotation around / flattening along the Y axis, where the 2d plane is Z, X
  //outYaw = atan2(vec[0], vec[2]); // simple, but doesn't round-trip
  outYaw = atan2(vec[0], sqrt((vec[2] * vec[2]) + (vec[1] * vec[1]))); // based on Mathematica's ToSphericalCoordinates[], round-trips with anglesToVector (when roll == 0)
  if (outYaw >= M_PI)
    outYaw -= (2.0 * M_PI);
}


void evaluate(const double* coeffs, double rawPupilPitch, double rawPupilYaw, double& correctedPitch, double& correctedYaw) {
  const double r = 0.5;

  double rollCorrectedPupilPitch;
  double rollCorrectedPupilYaw;

  {
    double roll = coeffs[kRollAngle];
    double v[3];
    anglesToVector(roll, rawPupilPitch, rawPupilYaw, v);
    vectorToAngles(v, rollCorrectedPupilPitch, rollCorrectedPupilYaw);
  }

  {
    const double midGazeX = cos(rollCorrectedPupilYaw);
    const double midGazeY = sin(rollCorrectedPupilYaw);
    double offsetX = double(midGazeX);
    double offsetY = midGazeY + (coeffs[kXOffset] / r);

    correctedYaw = atan2(offsetY, offsetX);
  }

  {
    const double midGazeX = cos(rollCorrectedPupilPitch);
    const double midGazeY = sin(rollCorrectedPupilPitch);
    double offsetX = double(midGazeX);
    double offsetY = midGazeY + (coeffs[kYOffset] / r);
    correctedPitch = atan2(offsetY, offsetX);
  }
}



struct RollResidual {
  RollResidual(double rawPupilPitch, double rawPupilYaw, double targetPupilPitch, double targetPupilYaw) : m_rawPupilPitch(rawPupilPitch), m_rawPupilYaw(rawPupilYaw), m_gtPupilPitch(targetPupilPitch), m_gtPupilYaw(targetPupilYaw) {}

  template <typename T> bool operator()(const T* const coeffs, T* residual) const {
    T roll = coeffs[kRollAngle];

    // Compute roll correction residual/error
    T rolledPupilVec[3];
    anglesToVector(roll, T(m_rawPupilPitch), T(m_rawPupilYaw), rolledPupilVec);

    double targetPupilVec[3];
    anglesToVector(0.0, m_gtPupilPitch, m_gtPupilYaw, targetPupilVec);

#if 1
#if 0
    // Error is the dot product of the normalized 2d/xy projections of the two vectors (flatten along z-axis)
    {
      T tl = sqrt((rolledPupilVec[0] * rolledPupilVec[0]) + (rolledPupilVec[1] * rolledPupilVec[1]));
      rolledPupilVec[0] /= tl;
      rolledPupilVec[1] /= tl;
    }
    {
      double l = sqrt((targetPupilVec[0] * targetPupilVec[0]) + (targetPupilVec[1] * targetPupilVec[1]));
      targetPupilVec[0] /= l;
      targetPupilVec[1] /= l;
    }

    residual[0] = 1.0 - ((rolledPupilVec[0] * targetPupilVec[0]) + (rolledPupilVec[1] * targetPupilVec[1]));
#else

    // Convert rolled vector back to angles
    {
      T rp, ry;
      vectorToAngles(rolledPupilVec, rp, ry);


      double gtL = sqrt(square(m_gtPupilPitch) + square(m_gtPupilYaw));
      T rL = sqrt(square(rp) + square(ry));

      // Dot product between normalized pitch/yaw vectors
      residual[0] = 1.0 - (((rp/rL) * (m_gtPupilPitch/gtL)) + ((ry/rL) * (m_gtPupilYaw/gtL)));
    }
#endif
#else

    // Error is the dot product of the two vectors
    residual[0] = 1.0 - (
      (rolledPupilVec[0] * targetPupilVec[0]) + 
      (rolledPupilVec[1] * targetPupilVec[1]) + 
      (rolledPupilVec[2] * targetPupilVec[2]));
#endif
    return true;
  }

 private:
  const double m_rawPupilPitch;
  const double m_rawPupilYaw;

  const double m_gtPupilPitch;
  const double m_gtPupilYaw;
};


struct OffsetResidual {
  OffsetResidual(double rawPupilPitch, double rawPupilYaw, double targetPupilPitch, double targetPupilYaw) : m_rawPupilPitch(rawPupilPitch), m_rawPupilYaw(rawPupilYaw), m_gtPupilPitch(targetPupilPitch), m_gtPupilYaw(targetPupilYaw) {}

  template <typename T> bool operator()(const T* const coeffs, T* residual) const {
    /*
      r = 0.5;
      yOffset = 0.03;
      eyeAngle = eyeAngleDeg * (Pi/180); (*to radians*)

      midGaze = {r Cos[eyeAngle], r Sin[eyeAngle]};
      offsetEye = Normalize[midGaze + {0, yOffset}] * r;

      // XXX note: Mathematica ordering is ArcTan[x, y], C ordering is atan2(y, x)
      correctedEyeAngle = ArcTan[ offsetEye[[1]], offsetEye[[2]] ];
    */

  /*
  {ArcTan[Cos[pitch] Cos[yaw], -Cos[roll] Cos[yaw] Sin[pitch] + 
     Sin[roll] Sin[yaw]], 
   ArcTan[Cos[pitch] Cos[yaw], 
    Cos[yaw] Sin[pitch] Sin[roll] + Cos[roll] Sin[yaw]]}
  */



    T roll = coeffs[kRollAngle];

    // Compute roll correction residual/error
    T rolledPupilVec[3];
    anglesToVector(roll, T(m_rawPupilPitch), T(m_rawPupilYaw), rolledPupilVec);


    // Apply roll correction and get adjusted pitch/yaw angles
    T rollCorrectedPupilPitch, rollCorrectedPupilYaw;
    vectorToAngles(rolledPupilVec, rollCorrectedPupilPitch, rollCorrectedPupilYaw);

#if 1
#if 0
    // Error is the dot product of the normalized 2d/xy projections of the two vectors (flatten along z-axis)
    {
      T tl = sqrt((rolledPupilVec[0] * rolledPupilVec[0]) + (rolledPupilVec[1] * rolledPupilVec[1]));
      rolledPupilVec[0] /= tl;
      rolledPupilVec[1] /= tl;
    }
    {
      double l = sqrt((targetPupilVec[0] * targetPupilVec[0]) + (targetPupilVec[1] * targetPupilVec[1]));
      targetPupilVec[0] /= l;
      targetPupilVec[1] /= l;
    }

    residual[0] = 1.0 - ((rolledPupilVec[0] * targetPupilVec[0]) + (rolledPupilVec[1] * targetPupilVec[1]));
#else

    // Convert rolled vector back to angles
    {
      double gtL = sqrt(square(m_gtPupilPitch) + square(m_gtPupilYaw));
      T rL = sqrt(square(rollCorrectedPupilPitch) + square(rollCorrectedPupilYaw));

      // Dot product between normalized pitch/yaw vectors
      residual[0] = 1.0 - (((rollCorrectedPupilPitch/rL) * (m_gtPupilPitch/gtL)) + ((rollCorrectedPupilYaw/rL) * (m_gtPupilYaw/gtL)));
    }
#endif
#else

    // Error is the dot product of the two vectors
    residual[0] = 1.0 - (
      (rolledPupilVec[0] * targetPupilVec[0]) + 
      (rolledPupilVec[1] * targetPupilVec[1]) + 
      (rolledPupilVec[2] * targetPupilVec[2]));
#endif

    const double r = 0.5;
    {
      T midGazeX = cos(rollCorrectedPupilYaw);
      T midGazeY = sin(rollCorrectedPupilYaw);
      T offsetX = T(midGazeX);
      T offsetY = midGazeY + (coeffs[kXOffset] / r);

      T offsetL = sqrt((offsetX * offsetX) + (offsetY * offsetY));

      T correctedYaw = atan2(offsetY / offsetL, offsetX / offsetL);
      residual[1] = m_gtPupilYaw - correctedYaw;
    }

    {
      T midGazeX = cos(rollCorrectedPupilPitch);
      T midGazeY = sin(rollCorrectedPupilPitch);
      T offsetX = T(midGazeX);
      T offsetY = midGazeY + (coeffs[kYOffset] / r);
      T offsetL = sqrt((offsetX * offsetX) + (offsetY * offsetY));
      T correctedPitch = atan2(offsetY / offsetL, offsetX / offsetL);
      residual[2] = m_gtPupilPitch - correctedPitch;
    }

    return true;
  }

 private:
  const double m_rawPupilPitch;
  const double m_rawPupilYaw;

  const double m_gtPupilPitch;
  const double m_gtPupilYaw;
};

int main(int argc, char** argv) {
  double coeffs[kCoeffCount];
  memset(coeffs, 0, sizeof(coeffs));

  ceres::Problem offsetProblem;
  for (size_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
    const double* row = data + (sampleIdx * kCalibrationSampleColumns);

    // stored in degrees
    double gtPitch = radians(row[0]);
    double gtYaw = radians(row[1]);

    const double* pupilVec = row + 4;
    const double* centerVec = row + 7;

    // ToPitchYaw[vec_] := {ArcTan[vec[[3]], vec[[2]]], ArcTan[vec[[3]], vec[[1]]]};
    double centerPitch, centerYaw;
    vectorToAngles(centerVec, centerPitch, centerYaw);

    double pupilPitch, pupilYaw;
    vectorToAngles(pupilVec, pupilPitch, pupilYaw);

    pupilPitch -= centerPitch;
    pupilYaw -= centerYaw;

#if 0
    if (sampleIdx == 0) {
      double vvec[3];
      anglesToVector(0.0, centerPitch, centerYaw, vvec);
      printf("anglesToVector Validation: {%.6f, %.6f, %.6f}, {%.6f, %.6f, %.6f}\n",
        vvec[0], vvec[1], vvec[2], centerVec[0], centerVec[1], centerVec[2]);
    }

    {
      double vvec[3];
      double pitch, yaw;
      anglesToVector(0.0, gtPitch, gtYaw, vvec);
      vectorToAngles(vvec, pitch, yaw);
      printf("vectorToAngles Validation: {%.6f, %.6f} -> {%.6f, %.6f, %.6f} -> {%.6f, %.6f}\n",
        degrees(gtPitch), degrees(gtYaw),
        vvec[0], vvec[1], vvec[2],
        degrees(pitch), degrees(yaw));
    }
#endif

#if 1
    printf("Sample [%zu]: GT = {%.3f, %.3f}, raw/center-corrected = {%.3f, %.3f}\n",
      sampleIdx, degrees(gtPitch), degrees(gtYaw), degrees(pupilPitch), degrees(pupilYaw));
#endif

    offsetProblem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<OffsetResidual, /*nResiduals=*/ 3, kCoeffCount>(
          new OffsetResidual(pupilPitch, pupilYaw, gtPitch, gtYaw)
        ),
#if 0
        new ceres::CauchyLoss(0.5),
#else
        nullptr,
#endif
        coeffs);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 1000;
  options.linear_solver_type = ceres::DENSE_QR;
  options.function_tolerance = 1e-10; // default 1e-6
  options.gradient_tolerance = 1e-4 * options.function_tolerance;
  options.minimizer_progress_to_stdout = true;


  {
    ceres::Solver::Summary summary;
    ceres::Solve(options, &offsetProblem, &summary);
    std::cout << summary.BriefReport() << "\n";
  }


  printf("xOffset: %.2f mm\n", coeffs[0] * 1000.0);
  printf("yOffset: %.2f mm\n", coeffs[1] * 1000.0);
  printf("roll: %.2f deg\n", degrees(coeffs[2]));

  std::cout << "const double coeffs[] =  {";
  for (size_t i = 0; i < kCoeffCount; ++i) {
    std::cout << coeffs[i] << ", ";
  }
  std::cout << "};" << std::endl;

  for (size_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
    const double* row = data + (sampleIdx * kCalibrationSampleColumns);

    double gtPitch = radians(row[0]);
    double gtYaw = radians(row[1]);

    const double* pupilVec = row + 4;
    const double* centerVec = row + 7;

    double centerPitch, centerYaw;
    vectorToAngles(centerVec, centerPitch, centerYaw);

    double pupilPitch, pupilYaw;
    vectorToAngles(pupilVec, pupilPitch, pupilYaw);

    pupilPitch -= centerPitch;
    pupilYaw -= centerYaw;


    double rollCorrectedPupilPitch, rollCorrectedPupilYaw;
    // Apply roll correction only and get adjusted pitch/yaw angles
    {
      double rolledPupilVec[3];
      anglesToVector(coeffs[kRollAngle], pupilPitch, pupilYaw, rolledPupilVec);
      vectorToAngles(rolledPupilVec, rollCorrectedPupilPitch, rollCorrectedPupilYaw);
    }

    double correctedPitch, correctedYaw;
    evaluate(coeffs, pupilPitch, pupilYaw, correctedPitch, correctedYaw);

    double dp = gtPitch - correctedPitch;
    double dy = gtYaw - correctedYaw;
    double dist = sqrt((dp * dp) + (dy * dy));

    printf("{%.3f, %.3f} -- rollCorrected={%.3f, %.3f}, pred={%.3f, %.3f} gt={%.3f, %.3f} dist=%.3f\n",
      degrees(pupilPitch), degrees(pupilYaw),
      degrees(rollCorrectedPupilPitch), degrees(rollCorrectedPupilYaw),
      degrees(correctedPitch), degrees(correctedYaw),
      degrees(gtPitch), degrees(gtYaw), degrees(dist));
  }


  return 0;
}

