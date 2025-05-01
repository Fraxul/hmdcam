// clang++ -Wall -I/usr/include/eigen3 solver-test.cpp -o solver-test -lceres -lglog
#include "ceres/ceres.h"

const size_t kCalibrationSampleColumns = 10;
const double data[] = {
-2.500000, 0.000000, -0.623791, -0.043942, 0.026840, -0.577845, 0.815705, 0.027607, -0.586679, 0.809349,
2.500000, 0.000000, 3.046352, -0.307061, 0.022250, -0.628946, 0.777131, 0.027607, -0.586679, 0.809349,
0.000000, 2.500000, 0.266014, 0.202473, 0.031139, -0.590369, 0.806532, 0.027607, -0.586679, 0.809349,
0.000000, -2.500000, 1.084370, -3.284137, -0.029704, -0.601855, 0.798053, 0.027607, -0.586679, 0.809349,
2.500000, 2.500000, 2.388954, 0.266632, 0.032259, -0.619819, 0.784082, 0.027607, -0.586679, 0.809349,
2.500000, -2.500000, 2.693760, -2.848446, -0.022102, -0.624154, 0.780989, 0.027607, -0.586679, 0.809349,
-2.500000, 2.500000, -0.761169, -0.047170, 0.026784, -0.575889, 0.817089, 0.027607, -0.586679, 0.809349,
-2.500000, -2.500000, -0.407745, -3.422661, -0.032121, -0.580826, 0.813394, 0.027607, -0.586679, 0.809349,
-5.000000, 0.000000, -1.751472, -1.663654, -0.001426, -0.561882, 0.827217, 0.027607, -0.586679, 0.809349,
5.000000, 0.000000, 4.318909, -0.542810, 0.018136, -0.646104, 0.763034, 0.027607, -0.586679, 0.809349,
0.000000, 5.000000, -0.406021, 3.154575, 0.082574, -0.579166, 0.811017, 0.027607, -0.586679, 0.809349,
0.000000, -5.000000, 2.088638, -4.768317, -0.055584, -0.615069, 0.786512, 0.027607, -0.586679, 0.809349,
5.000000, 5.000000, 3.578533, 3.041795, 0.080612, -0.634224, 0.768936, 0.027607, -0.586679, 0.809349,
5.000000, -5.000000, 5.028008, -4.120853, -0.044297, -0.654961, 0.754363, 0.027607, -0.586679, 0.809349,
-5.000000, 5.000000, -3.069004, 1.693148, 0.057130, -0.541827, 0.838546, 0.027607, -0.586679, 0.809349,
-5.000000, -5.000000, -1.502213, -5.618207, -0.070387, -0.564073, 0.822719, 0.027607, -0.586679, 0.809349,
-7.500000, 0.000000, -4.056482, -2.029961, -0.007819, -0.528141, 0.849120, 0.027607, -0.586679, 0.809349,
7.500000, 0.000000, 5.951286, -0.477929, 0.019268, -0.667563, 0.744304, 0.027607, -0.586679, 0.809349,
0.000000, 7.500000, -0.996777, 4.406017, 0.104320, -0.569604, 0.815272, 0.027607, -0.586679, 0.809349,
0.000000, -7.500000, 1.769245, -6.998367, -0.094393, -0.608890, 0.787619, 0.027607, -0.586679, 0.809349,
7.500000, 7.500000, 4.074081, 5.529447, 0.123799, -0.637997, 0.760022, 0.027607, -0.586679, 0.809349,
7.500000, -7.500000, 6.302006, -6.778198, -0.090567, -0.669469, 0.737298, 0.027607, -0.586679, 0.809349,
-7.500000, 7.500000, -5.088871, 2.359570, 0.068739, -0.511559, 0.856494, 0.027607, -0.586679, 0.809349,
-7.500000, -7.500000, -2.920525, -7.314501, -0.099885, -0.542163, 0.834316, 0.027607, -0.586679, 0.809349,
-10.000000, 0.000000, -4.707352, -2.878659, -0.022630, -0.518345, 0.854872, 0.027607, -0.586679, 0.809349,
10.000000, 0.000000, 7.445572, -0.339990, 0.021675, -0.686712, 0.726607, 0.027607, -0.586679, 0.809349,
0.000000, 10.000000, -1.043900, 7.095862, 0.150878, -0.565506, 0.810826, 0.027607, -0.586679, 0.809349,
0.000000, -10.000000, 1.892242, -8.684870, -0.123652, -0.608611, 0.783775, 0.027607, -0.586679, 0.809349,
10.000000, 10.000000, 5.573814, 7.193471, 0.152562, -0.655010, 0.740059, 0.027607, -0.586679, 0.809349,
10.000000, -10.000000, 8.694984, -7.234201, -0.098490, -0.699141, 0.708168, 0.027607, -0.586679, 0.809349,
-10.000000, 10.000000, -6.961044, 3.770576, 0.093284, -0.482338, 0.871004, 0.027607, -0.586679, 0.809349,
-10.000000, -10.000000, -5.325769, -9.727326, -0.141685, -0.504081, 0.851955, 0.027607, -0.586679, 0.809349,
-12.500000, 0.000000, -6.813692, -3.169281, -0.027700, -0.486512, 0.873235, 0.027607, -0.586679, 0.809349,
12.500000, 0.000000, 9.019848, -0.009215, 0.027446, -0.706314, 0.707366, 0.027607, -0.586679, 0.809349,
0.000000, 12.500000, -0.864014, 6.941010, 0.148206, -0.568281, 0.809377, 0.027607, -0.586679, 0.809349,
0.000000, -12.500000, 2.121937, -10.648060, -0.157574, -0.608777, 0.777535, 0.027607, -0.586679, 0.809349,
12.500000, 12.500000, 6.577343, 10.186575, 0.203959, -0.661576, 0.721608, 0.027607, -0.586679, 0.809349,
12.500000, -12.500000, 9.668869, -9.339285, -0.134977, -0.708012, 0.693181, 0.027607, -0.586679, 0.809349,
-12.500000, 12.500000, -9.679575, 4.387447, 0.103998, -0.440014, 0.891948, 0.027607, -0.586679, 0.809349,
-12.500000, -12.500000, -6.483852, -12.209788, -0.184429, -0.483285, 0.855817, 0.027607, -0.586679, 0.809349,
};

const size_t sampleCount = (sizeof(data) / sizeof(data[0])) / kCalibrationSampleColumns;

template <typename T> static inline T degrees(T r) { return r * (180.0 / M_PI); }
template <typename T> static inline T radians(T d) { return d * (M_PI / 180.0); }
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
    // Boundary check
    if (((degrees(coeffs[kRollAngle])) > 30.0) || (degrees(coeffs[kRollAngle]) < -30.0))
      return false;
    if ((coeffs[kXOffset] > 0.100) || (coeffs[kXOffset] < -0.100))
      return false;
    if ((coeffs[kYOffset] > 0.100) || (coeffs[kYOffset] < -0.100))
      return false;

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
    printf("Sample [%zu]: GT = {%.3f, %.3f}, sample angles = {%.3f, %.3f}, raw/center-corrected = {%.3f, %.3f}\n",
      sampleIdx, degrees(gtPitch), degrees(gtYaw),
      row[2], row[3],
      degrees(pupilPitch), degrees(pupilYaw));
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

