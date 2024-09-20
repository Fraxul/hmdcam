#pragma once
#include <cuda.h>
#include <driver_types.h> // cudaError_t
#define CUDA_CHECK(x) checkCUresult(x, #x, __FILE__, __LINE__, true)
#define CUDA_CHECK_NONFATAL(x) checkCUresult(x, #x, __FILE__, __LINE__, false)
#define NPP_CHECK(x) do { NppStatus _nppStatus = x; if (_nppStatus < 0) { fprintf(stderr, "%s (%s:%d) returned NppStatus %d\n", #x, __FILE__, __LINE__, _nppStatus); abort(); } } while(0) 
bool checkCUresult(CUresult res, const char* op, const char* file, int line, bool fatal);
bool checkCUresult(cudaError_t res, const char* op, const char* file, int line, bool fatal);

