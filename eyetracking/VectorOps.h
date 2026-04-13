#pragma once
#include <stdint.h>
#include <unistd.h>
#include <arm_neon.h>

void convertUnorm8ToSnormFp16(const uint8_t* inU8, void* outFP16, const size_t elementCount);
void convertUnorm8ToDLAInt8(const uint8_t* inU8, void* outDLAInt8, size_t elementCount);
void fp16ThresholdToU8Mask(const _Float16* inFP16, _Float16 thresholdValue, uint8_t* outU8, size_t elementCount);
_Float16 fp16VectorMax(const _Float16* inFP16, size_t elementCount);
void convertGrayToRGBA(const uint8_t* inGray, uint8_t* outRGBA, size_t count);
void areaDownsample6x6(const uint8_t* __restrict inU8, size_t inputRowStride, uint8_t* __restrict outU8, size_t outputWidth, size_t outputHeight, size_t outputRowStride);

