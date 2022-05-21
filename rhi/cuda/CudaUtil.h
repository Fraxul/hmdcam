#pragma once
#include <cuda.h>
#define CUDA_CHECK(x) checkCUresult(x, #x, __FILE__, __LINE__, true)
#define CUDA_CHECK_NONFATAL(x) checkCUresult(x, #x, __FILE__, __LINE__, false)
bool checkCUresult(CUresult res, const char* op, const char* file, int line, bool fatal);

