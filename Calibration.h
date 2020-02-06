#pragma once
#include <cstddef>

void updateCameraDistortionMap(size_t cameraIdx, bool useStereoCalibration);
void initCalibration();
void readCalibrationData();
void saveCalibrationData();
bool haveIntrinsicCalibration();
bool haveStereoCalibration();
void doIntrinsicCalibration();
void doStereoCalibration();
void generateCalibrationDerivedData();
