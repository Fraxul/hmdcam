/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
 
 
#ifndef NVMEDIA_CORE_H
#define NVMEDIA_CORE_H
 
#ifdef __cplusplus
extern "C" {
#endif
 
#include <math.h>
#include <stdint.h>
#include <time.h>
 
#if !defined(NVM_DEPRECATED)
    #if defined(__GNUC__) && (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
        /*
         * deprecated as build time warnings to prompt developers to migrate
         * from older API to new one gradually. Should be removed once API
         * transition is done(ie: no warnings).
         */
 
        #pragma GCC diagnostic warning "-Wdeprecated-declarations"
        #define NVM_DEPRECATED_MSG(fmt) __attribute__((deprecated(fmt)))
    #else
        #define NVM_DEPRECATED
        #define NVM_DEPRECATED_MSG(fmt) NVM_DEPRECATED
    #endif
#else
    #define NVM_DEPRECATED_MSG(fmt) NVM_DEPRECATED
#endif
 
#define NVMEDIA_TRUE  (0 == 0)
 
#define NVMEDIA_FALSE (0 == 1)
 
typedef uint32_t NvMediaBool;
 
typedef struct timespec NvMediaTime;
 
typedef struct {
    uint16_t x0;
    uint16_t y0;
    uint16_t x1;
    uint16_t y1;
} NvMediaRect;
 
typedef enum {
    NVMEDIA_STATUS_OK = 0,
    NVMEDIA_STATUS_BAD_PARAMETER = 1,
    NVMEDIA_STATUS_PENDING = 2,
    NVMEDIA_STATUS_TIMED_OUT = 3,
    NVMEDIA_STATUS_OUT_OF_MEMORY = 4,
    NVMEDIA_STATUS_NOT_INITIALIZED = 5,
    NVMEDIA_STATUS_NOT_SUPPORTED = 6,
    NVMEDIA_STATUS_ERROR = 7,
    NVMEDIA_STATUS_NONE_PENDING = 8,
    NVMEDIA_STATUS_INSUFFICIENT_BUFFERING = 9,
    NVMEDIA_STATUS_INVALID_SIZE = 10,
    NVMEDIA_STATUS_INCOMPATIBLE_VERSION = 11,
    NVMEDIA_STATUS_UNDEFINED_STATE = 13,
    NVMEDIA_STATUS_PFSD_ERROR = 14,
    NVMEDIA_STATUS_INVALID_STATE = 15,
} NvMediaStatus;
 
typedef struct {
    NvMediaStatus status;
    uint64_t endTimestamp;
} NvMediaTaskStatus;
 
typedef struct {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
} NvMediaVersion;
 
/*
 ******* Definitions ******************
 * SOFFence - Start of frame \ref NvSciSyncFence. An NvSciSyncFence
 *            whose expiry indicates that the processing has started.
 * EOFFence - End of frame NvSciSyncFence. An NvSciSyncFence
 *            whose expiry indicates that the processing is done.
 * PREFence - An NvSciSyncFence on which the start of processing is
 *            blocked until the expiry of the fence.
 **************************************
 */
 
typedef enum {
    NVMEDIA_SIGNALER,
    NVMEDIA_WAITER,
    NVMEDIA_SIGNALER_WAITER
} NvMediaNvSciSyncClientType;
 
typedef enum {
    NVMEDIA_PRESYNCOBJ,
    NVMEDIA_EOFSYNCOBJ,
    NVMEDIA_SOFSYNCOBJ,
    NVMEDIA_EOF_PRESYNCOBJ,
    NVMEDIA_SOF_PRESYNCOBJ
 
} NvMediaNvSciSyncObjType;
 
/*
 * \brief Defines all possible access modes.
 * \note This definition is deprecated and will be removed in the next version.
 */
typedef enum {
    NVMEDIA_ACCESS_MODE_READ,
    NVMEDIA_ACCESS_MODE_READ_WRITE,
} NvMediaAccessMode;
 
typedef void NvMediaDevice;
 
#ifdef __cplusplus
}      /* extern "C" */
#endif
 
#endif /* NVMEDIA_CORE_H */

