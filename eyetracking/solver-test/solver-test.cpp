// clang++ -Wall -I/usr/include/eigen3 solver-test.cpp -o solver-test -lceres -lglog
#include "ceres/ceres.h"

const size_t kCalibrationSampleColumns = 10;
const double data[] = {
-2.500000, 0.000000, -3.109352, -0.014764, 0.016022, -0.577826, 0.816003, 0.016280, -0.621234, 0.783456,
2.500000, 0.000000, -0.526764, 0.427288, 0.023736, -0.613913, 0.789016, 0.016280, -0.621234, 0.783456,
0.000000, 2.500000, -2.061260, 1.782217, 0.047368, -0.592066, 0.804496, 0.016280, -0.621234, 0.783456,
0.000000, -2.500000, -0.923656, -2.039393, -0.019313, -0.608491, 0.793326, 0.016280, -0.621234, 0.783456,
2.500000, 2.500000, 0.051331, 2.526862, 0.060346, -0.620885, 0.781576, 0.016280, -0.621234, 0.783456,
2.500000, -2.500000, 0.938347, -1.500963, -0.009916, -0.634034, 0.773242, 0.016280, -0.621234, 0.783456,
-2.500000, 2.500000, -2.973152, 1.375707, 0.040280, -0.579368, 0.814070, 0.016280, -0.621234, 0.783456,
-2.500000, -2.500000, -2.832802, -2.461834, -0.026684, -0.581625, 0.813019, 0.016280, -0.621234, 0.783456,
-5.000000, 0.000000, -4.116608, -1.467647, -0.009335, -0.563440, 0.826104, 0.016280, -0.621234, 0.783456,
5.000000, 0.000000, 2.162128, 0.240937, 0.020484, -0.650299, 0.759402, 0.016280, -0.621234, 0.783456,
0.000000, 5.000000, -1.326317, 3.454965, 0.076506, -0.601246, 0.795393, 0.016280, -0.621234, 0.783456,
0.000000, -5.000000, -0.916916, -4.107139, -0.055374, -0.607764, 0.792185, 0.016280, -0.621234, 0.783456,
5.000000, 5.000000, -0.078365, 4.083427, 0.087438, -0.617869, 0.781404, 0.016280, -0.621234, 0.783456,
5.000000, -5.000000, 2.297577, -3.516659, -0.045082, -0.651566, 0.757251, 0.016280, -0.621234, 0.783456,
-5.000000, 5.000000, -5.447964, 2.738154, 0.064026, -0.543001, 0.837288, 0.016280, -0.621234, 0.783456,
-5.000000, -5.000000, -4.991997, -4.708163, -0.065845, -0.549582, 0.832841, 0.016280, -0.621234, 0.783456,
-7.500000, 0.000000, -6.074451, -0.580707, 0.006145, -0.534901, 0.844892, 0.016280, -0.621234, 0.783456,
7.500000, 0.000000, -6.012535, -0.725669, 0.003615, -0.535820, 0.844324, 0.016280, -0.621234, 0.783456,
0.000000, 7.500000, -3.676464, 4.467957, 0.094121, -0.567265, 0.818139, 0.016280, -0.621234, 0.783456,
0.000000, -7.500000, -0.883312, -6.505158, -0.097103, -0.606285, 0.789297, 0.016280, -0.621234, 0.783456,
7.500000, 7.500000, 2.090771, 5.682563, 0.115203, -0.645165, 0.755308, 0.016280, -0.621234, 0.783456,
7.500000, -7.500000, 4.329224, -5.659284, -0.082399, -0.676385, 0.731925, 0.016280, -0.621234, 0.783456,
-7.500000, 7.500000, -8.607971, 3.421337, 0.075921, -0.495606, 0.865223, 0.016280, -0.621234, 0.783456,
-7.500000, -7.500000, -5.575859, -6.825141, -0.102660, -0.539378, 0.835782, 0.016280, -0.621234, 0.783456,
-10.000000, 0.000000, -6.952415, -1.843933, -0.015902, -0.521836, 0.852898, 0.016280, -0.621234, 0.783456,
10.000000, 0.000000, 4.824932, 0.972084, 0.033240, -0.684643, 0.728121, 0.016280, -0.621234, 0.783456,
0.000000, 10.000000, -3.648628, 6.746398, 0.133626, -0.565080, 0.814143, 0.016280, -0.621234, 0.783456,
0.000000, -10.000000, -1.141613, -7.502610, -0.114414, -0.601605, 0.790557, 0.016280, -0.621234, 0.783456,
10.000000, 10.000000, 2.097649, 8.165370, 0.158126, -0.641408, 0.750727, 0.016280, -0.621234, 0.783456,
10.000000, -10.000000, 5.740009, -7.184738, -0.108901, -0.692426, 0.713223, 0.016280, -0.621234, 0.783456,
-10.000000, 10.000000, -9.700319, 4.485842, 0.094432, -0.478261, 0.873126, 0.016280, -0.621234, 0.783456,
-10.000000, -10.000000, -7.804996, -9.034915, -0.140938, -0.504070, 0.852086, 0.016280, -0.621234, 0.783456,
-12.500000, 0.000000, -9.775454, -2.369960, -0.025081, -0.479106, 0.877399, 0.016280, -0.621234, 0.783456,
12.500000, 0.000000, 6.583267, 1.499212, 0.042434, -0.706416, 0.706524, 0.016280, -0.621234, 0.783456,
0.000000, 12.500000, -4.029171, 9.560923, 0.182128, -0.555279, 0.811477, 0.016280, -0.621234, 0.783456,
0.000000, -12.500000, -0.985428, -9.970089, -0.157077, -0.600204, 0.784271, 0.016280, -0.621234, 0.783456,
12.500000, 12.500000, 4.101078, 10.387280, 0.196290, -0.662616, 0.722779, 0.016280, -0.621234, 0.783456,
12.500000, -12.500000, 7.727535, -9.281029, -0.145189, -0.713393, 0.685558, 0.016280, -0.621234, 0.783456,
-12.500000, 12.500000, -10.521433, 6.383061, 0.127339, -0.463981, 0.876645, 0.016280, -0.621234, 0.783456,
-12.500000, -12.500000, -9.111107, -11.093447, -0.176409, -0.481726, 0.858382, 0.016280, -0.621234, 0.783456,
-15.000000, 0.000000, -11.581476, -1.241658, -0.005391, -0.451352, 0.892330, 0.016280, -0.621234, 0.783456,
15.000000, 0.000000, 8.115593, 1.787453, 0.047459, -0.724892, 0.687226, 0.016280, -0.621234, 0.783456,
0.000000, 15.000000, -5.200699, 10.086059, 0.191132, -0.537635, 0.821229, 0.016280, -0.621234, 0.783456,
0.000000, -15.000000, -1.424469, -11.917897, -0.190554, -0.590622, 0.784127, 0.016280, -0.621234, 0.783456,
15.000000, 15.000000, 4.671680, 13.120361, 0.242822, -0.662626, 0.708494, 0.016280, -0.621234, 0.783456,
15.000000, -15.000000, 10.257282, -10.623431, -0.168328, -0.740199, 0.650977, 0.016280, -0.621234, 0.783456,
-15.000000, 15.000000, -14.904781, 7.193192, 0.141350, -0.394865, 0.907800, 0.016280, -0.621234, 0.783456,
-15.000000, -15.000000, -11.467916, -13.629799, -0.219795, -0.442045, 0.869647, 0.016280, -0.621234, 0.783456,
-17.500000, 0.000000, -12.537743, -3.013188, -0.036302, -0.436115, 0.899158, 0.016280, -0.621234, 0.783456,
17.500000, 0.000000, 10.400921, 2.614764, 0.061877, -0.751125, 0.657253, 0.016280, -0.621234, 0.783456,
0.000000, 17.500000, -5.127411, 13.342731, 0.246585, -0.531856, 0.810139, 0.016280, -0.621234, 0.783456,
0.000000, -17.500000, -0.070724, -13.581646, -0.218975, -0.605293, 0.765291, 0.016280, -0.621234, 0.783456,
17.500000, 17.500000, 6.210880, 15.428321, 0.281690, -0.673996, 0.682920, 0.016280, -0.621234, 0.783456,
17.500000, -17.500000, 12.188488, -11.711959, -0.187024, -0.759108, 0.623520, 0.016280, -0.621234, 0.783456,
-17.500000, 17.500000, -16.588171, 6.655645, 0.132056, -0.368504, 0.920199, 0.016280, -0.621234, 0.783456,
-17.500000, -17.500000, -13.528589, -15.707006, -0.255011, -0.406867, 0.877171, 0.016280, -0.621234, 0.783456,
-20.000000, 0.000000, -14.202217, -1.774202, -0.014685, -0.410040, 0.911949, 0.016280, -0.621234, 0.783456,
20.000000, 0.000000, 12.313496, 2.837461, 0.065756, -0.772450, 0.631662, 0.016280, -0.621234, 0.783456,
0.000000, 20.000000, -7.074276, 13.532335, 0.249791, -0.503600, 0.827038, 0.016280, -0.621234, 0.783456,
0.000000, -20.000000, -0.790009, -15.722760, -0.255277, -0.590228, 0.765810, 0.016280, -0.621234, 0.783456,
20.000000, 20.000000, 7.975151, 18.658234, 0.335304, -0.682108, 0.649846, 0.016280, -0.621234, 0.783456,
20.000000, -20.000000, 14.161652, -12.513061, -0.200741, -0.777974, 0.595366, 0.016280, -0.621234, 0.783456,
-20.000000, 20.000000, -18.752182, 5.778260, 0.116862, -0.334135, 0.935252, 0.016280, -0.621234, 0.783456,
-20.000000, -20.000000, -14.023729, -16.061892, -0.267408, -0.364349, 0.892044, 0.009631, -0.591155, 0.806501,
-22.500000, 0.000000, -15.028622, -1.848214, -0.022625, -0.361732, 0.932007, 0.009631, -0.591155, 0.806501,
22.500000, 0.000000, 15.865978, 3.574980, 0.071964, -0.787112, 0.612597, 0.009631, -0.591155, 0.806501,
0.000000, 22.500000, -4.202557, 16.059128, 0.285871, -0.508349, 0.812317, 0.009631, -0.591155, 0.806501,
0.000000, -22.500000, 2.194603, -16.125864, -0.268484, -0.598810, 0.754548, 0.009631, -0.591155, 0.806501,
22.500000, 22.500000, 11.026592, 20.263996, 0.355365, -0.686586, 0.634284, 0.009631, -0.591155, 0.806501,
-22.500000, 22.500000, -19.780546, 10.823427, 0.191568, -0.295749, 0.935860, 0.003855, -0.606234, 0.795277,
-22.500000, -22.500000, -16.008276, -19.295490, -0.326799, -0.343456, 0.880477, 0.003855, -0.606234, 0.795277,
-25.000000, 0.000000, -15.335083, 0.778672, 0.017444, -0.374274, 0.927154, 0.003855, -0.606234, 0.795277,
25.000000, 0.000000, 18.202236, 3.697839, 0.068341, -0.822399, 0.564791, 0.003855, -0.606234, 0.795277,
0.000000, 25.000000, -6.552738, 17.912930, 0.311237, -0.486116, 0.816592, 0.003855, -0.606234, 0.795277,
0.000000, -25.000000, 0.979446, -18.946045, -0.321029, -0.586941, 0.743264, 0.003855, -0.606234, 0.795277,
25.000000, 25.000000, 10.639397, 23.780870, 0.406764, -0.678433, 0.611777, 0.003855, -0.606234, 0.795277,
25.000000, -25.000000, 17.084274, -16.815065, -0.285591, -0.779259, 0.557847, 0.003855, -0.606234, 0.795277,
-25.000000, 25.000000, -23.347359, 10.602382, 0.187780, -0.237130, 0.953157, 0.003855, -0.606234, 0.795277,
-25.000000, -25.000000, -17.985819, -21.723623, -0.366546, -0.308004, 0.877939, 0.003855, -0.606234, 0.795277,
};

const size_t sampleCount = (sizeof(data) / sizeof(data[0])) / kCalibrationSampleColumns;

template <typename T> static inline T degrees(T r) { return r * (180.0 / M_PI); }
template <typename T> static inline T radians(T d) { return d * (M_PI / 180.0); }
template <typename T> static inline T square(T x) { return x * x; }

enum CoeffID {
  kXOffset,
  kYOffset,
  kRollAngle,
  kPitchScale,
  kYawScale,

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

    correctedYaw = atan2(offsetY, offsetX) * coeffs[kYawScale];
  }

  {
    const double midGazeX = cos(rollCorrectedPupilPitch);
    const double midGazeY = sin(rollCorrectedPupilPitch);
    double offsetX = double(midGazeX);
    double offsetY = midGazeY + (coeffs[kYOffset] / r);
    correctedPitch = atan2(offsetY, offsetX) * coeffs[kPitchScale];
  }
}



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

      T correctedYaw = atan2(offsetY / offsetL, offsetX / offsetL) * coeffs[kYawScale];
      residual[1] = m_gtPupilYaw - correctedYaw;
    }

    {
      T midGazeX = cos(rollCorrectedPupilPitch);
      T midGazeY = sin(rollCorrectedPupilPitch);
      T offsetX = T(midGazeX);
      T offsetY = midGazeY + (coeffs[kYOffset] / r);
      T offsetL = sqrt((offsetX * offsetX) + (offsetY * offsetY));
      T correctedPitch = atan2(offsetY / offsetL, offsetX / offsetL) * coeffs[kPitchScale];
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
#if 1
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


  printf("xOffset: %.2f mm\n", coeffs[kXOffset] * 1000.0);
  printf("yOffset: %.2f mm\n", coeffs[kYOffset] * 1000.0);
  printf("roll: %.2f deg\n", degrees(coeffs[kRollAngle]));
  printf("pitchScale: %.3f\n", coeffs[kPitchScale]);
  printf("yawScale: %.3f\n", coeffs[kYawScale]);

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

