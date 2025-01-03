#pragma once
#include "nvmedia_core.h"
#include "nvscierror.h"

#define NVMEDIA_CHECK(x) checkNvMediaStatus(x, #x, __FILE__, __LINE__, true)
#define NVSCI_CHECK(x) checkNvSciError(x, #x, __FILE__, __LINE__, true)
bool checkNvMediaStatus(NvMediaStatus res, const char* op, const char* file, int line, bool fatal);
bool checkNvSciError(NvSciError res, const char* op, const char* file, int line, bool fatal);

