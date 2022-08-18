#pragma once
#include <memory>
#ifdef HAVE_VPI2
#include <vpi/Status.h>

#define VPI_CHECK(x) checkVPIStatus(x, #x, __FILE__, __LINE__, true)
bool checkVPIStatus(VPIStatus res, const char* op, const char* file, int line, bool fatal);
#endif
