#pragma once
#include <cuda.h>
#define CUDA_CHECK(x) checkCUresult(x, #x, __FILE__, __LINE__)
void checkCUresult(CUresult res, const char* op, const char* file, int line);

