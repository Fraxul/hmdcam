/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.

// Based on the NvMedia API from DriveOS 6.0.10
//
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/group__x__basic__api__top.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//
 */

/**
 * \file
 * \brief <b> NVIDIA Media Interface: Basic NvMedia Types and Structures </b>
 *
 * Defines basic types used throughout the NvMedia API.
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

/** \brief A true \ref NvMediaBool value. */
#define NVMEDIA_TRUE  (0 == 0)

/** \brief A false \ref NvMediaBool value. */
#define NVMEDIA_FALSE (0 == 1)

/**
 * \brief A boolean value, holding \ref NVMEDIA_TRUE or \ref NVMEDIA_FALSE.
 */
typedef uint32_t NvMediaBool;

/**
 * \brief Holds the media time in timespec format as defined by the
 *  POSIX specification.
 */
typedef struct timespec NvMediaTime;

/**
 * \brief Holds a rectangular region of a surface.
 *
 * The co-ordinates are top-left inclusive, bottom-right exclusive.
 *
 * The NvMedia co-ordinate system has its origin at the top left corner
 * of a surface, with x and y components increasing right and down.
 */
typedef struct {
    /** Left X co-ordinate. Inclusive. */
    uint16_t x0;
    /** Top Y co-ordinate. Inclusive. */
    uint16_t y0;
    /** Right X co-ordinate. Exclusive. */
    uint16_t x1;
    /** Bottom Y co-ordinate. Exclusive. */
    uint16_t y1;
} NvMediaRect;

/**
 * \brief Defines all possible error codes.
 */
typedef enum {
    /** Specifies that the operation completed successfully (with no error). */
    NVMEDIA_STATUS_OK = 0,
    /** Specifies that a bad parameter was passed. */
    NVMEDIA_STATUS_BAD_PARAMETER = 1,
    /** Specifies that the operation has not finished yet. */
    NVMEDIA_STATUS_PENDING = 2,
    /** Specifies that the operation timed out. */
    NVMEDIA_STATUS_TIMED_OUT = 3,
    /** Specifies that the process is out of memory. */
    NVMEDIA_STATUS_OUT_OF_MEMORY = 4,
    /** Specifies that a component requred by the function call
     *  is not initialized. */
    NVMEDIA_STATUS_NOT_INITIALIZED = 5,
    /** Specifies that the requested operation is not supported. */
    NVMEDIA_STATUS_NOT_SUPPORTED = 6,
    /** Specifies a catch-all error, used when no other error code applies. */
    NVMEDIA_STATUS_ERROR = 7,
    /** Specifies that no operation is pending. */
    NVMEDIA_STATUS_NONE_PENDING = 8,
    /** Specifies insufficient buffering. */
    NVMEDIA_STATUS_INSUFFICIENT_BUFFERING = 9,
    /** Specifies that the size of an object passed to a function
     *  was invalid. */
    NVMEDIA_STATUS_INVALID_SIZE = 10,
    /** Specifies that a library's version is incompatible with
     *  the application. */
    NVMEDIA_STATUS_INCOMPATIBLE_VERSION = 11,
    /** Specifies that the operation entered an undefined state. */
    NVMEDIA_STATUS_UNDEFINED_STATE = 13,
    /** Specifies an error from Permanent Fault Software Diagnostic. */
    NVMEDIA_STATUS_PFSD_ERROR = 14,
    /** Specifies that the module is in invalid state to perform
     *  the requested operation. */
    NVMEDIA_STATUS_INVALID_STATE = 15,
} NvMediaStatus;

/**
 * \brief Holds status of latest operation for NvMedia managed data structure.
 */
typedef struct {
    /** Holds the status of the latest operation. */
    NvMediaStatus status;
    /** Holds the end timestamp of the latest operation. */
    uint64_t endTimestamp;
} NvMediaTaskStatus;

/**
 * \brief Provides version information for the NvMedia library.
 */
typedef struct {
    /** Major version number. */
    uint8_t major;
    /** Minor version number. */
    uint8_t minor;
    /** Patch version number. */
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

/**
 * \brief NvMedia NvSciSync Client Type.
 */
typedef enum {
    /** An NvMedia component acts as a signaler. */
    NVMEDIA_SIGNALER,
    /** An NvMedia component acts as a waiter. */
    NVMEDIA_WAITER,
    /** An NvMedia component acts as a signaler and waiter also
     *  for the same \ref NvSciSyncObj. */
    NVMEDIA_SIGNALER_WAITER
} NvMediaNvSciSyncClientType;

/**
 * \brief Defines NvMedia \ref NvSciSyncObj types.
 */
typedef enum {
    /** Specifies an NvSciSyncObj type for which an NvMedia component
     *  acts as a waiter. */
    NVMEDIA_PRESYNCOBJ,
    /** Specifies an NvSciSyncObj type for which an NvMedia component
     *  acts as a signaler, signaling EOFFence. */
    NVMEDIA_EOFSYNCOBJ,
    /** Specifies an NvSciSyncObj type for which an NvMedia component
     *  acts as a signaler, signaling SOFFence. */
    NVMEDIA_SOFSYNCOBJ,
    /** Specifies an NvSciSyncObj type for which an NvMedia component
     *  acts both as a signaler, signaling EOFFence, and as a waiter.
     *  Use this type in use cases where a EOFfence from an NvMedia
     *  component handle in one iteration is used as a PREfence for
     *  the same handle in the next iteration. */
    NVMEDIA_EOF_PRESYNCOBJ,
    /** Specifies an NvSciSyncObj type for which an NvMedia component
     *  acts as a signaler, signaling SOFFence, as a waiter.
     *  Use this type in use cases where a SOFfence from an NvMedia
     *  component handle in one iteration is used as a PREfence for
     *  the same handle in the next iteration. */
    NVMEDIA_SOF_PRESYNCOBJ

} NvMediaNvSciSyncObjType;

/**
 * \brief Defines all possible access modes.
 * \note This definition is deprecated and will be removed in the next version.
 */
typedef enum {
    /** Specifies read-only access mode. */
    NVMEDIA_ACCESS_MODE_READ,
    /** Specifies read/write access mode. */
    NVMEDIA_ACCESS_MODE_READ_WRITE,
} NvMediaAccessMode;

/**
 * \brief Manages NvMediaDevice objects, which are the root of the
 *  NvMedia object system.
 */
typedef void NvMediaDevice;

#ifdef __cplusplus
}      /* extern "C" */
#endif
 
#endif /* NVMEDIA_CORE_H */

