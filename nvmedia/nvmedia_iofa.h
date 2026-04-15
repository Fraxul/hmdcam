/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.

// Based on the NvMedia API from DriveOS 6.0.10
//
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/group__x__image__ofa__api.html
//
//
// Changes:
// - API version number bumped to 1.4.0 to match the version in JetPack 6.1
// - NVMEDIA_IOFA_MAX_PYD_LEVEL changed to 7U to match JetPack (was 5U)
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

 */

/**
 * \defgroup x_image_ofa_api Image Optical Flow Accelerator (IOFA)
 * \ingroup x_nvmedia_image_top
 *
 * The NvMediaIofa object takes an uncompressed bufObj frame pair and turns
 * them into optical flow / stereo disparity estimation data.
 *
 * @{
 */

#ifndef NVMEDIA_IOFA_H
#define NVMEDIA_IOFA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "nvmedia_core.h"
#include "nvscisync.h"
#include "nvscibuf.h"

/** \brief Major version number. */
#define NVMEDIA_IOFA_VERSION_MAJOR            1

/** \brief Minor version number. */
#define NVMEDIA_IOFA_VERSION_MINOR            4

/** \brief Patch version number. */
#define NVMEDIA_IOFA_VERSION_PATCH            0

/** \brief Maximum number of Pyramid level supported in Pyramid OF mode. */
#define NVMEDIA_IOFA_MAX_PYD_LEVEL            7U

/** \brief Maximum number of Region of Interest supported on IOFA. */
#define NVMEDIA_IOFA_MAX_ROI_SUPPORTED        32U

/**
 * \brief Specifies the maximum number of times
 * NvMediaIOFAInsertPreNvSciSyncFence() can be called before each call to
 * NvMediaIOFAProcessFrame().
 */
#define NVMEDIA_IOFA_MAX_PRENVSCISYNCFENCES   16U

/**
 * \brief Defines mode supported by IOFA Driver.
 */
typedef enum
{
    /** IOFA stereo disparity mode. */
    NVMEDIA_IOFA_MODE_STEREO = 0U,
    /** IOFA pyramid optical flow mode. */
    NVMEDIA_IOFA_MODE_PYDOF  = 1U,
    /** OFA epipolar optical flow mode.
     *  Epipolar OF support will be added in next release */
    NVMEDIA_IOFA_MODE_EPIOF  = 2U,
} NvMediaIofaMode;

/**
 * \brief Defines the Output Grid Size.
 *
 * IOFA supports variable flow vector/disparity output density.
 * IOFA provides single output for each input region corrosponding to grid
 * block size.
 * Grid Size controls flow vector/disparity map granularity.
 * Application can set any grid size from the list of Grid sizes supported
 * by IOFA.
 */
typedef enum
{
    /** Grid Size 1x1. */
    NVMEDIA_IOFA_GRIDSIZE_1X1 = 0U,
    /** Grid Size 2x2. */
    NVMEDIA_IOFA_GRIDSIZE_2X2 = 1U,
    /** Grid Size 4x4. */
    NVMEDIA_IOFA_GRIDSIZE_4X4 = 2U,
    /** Grid Size 8x8. */
    NVMEDIA_IOFA_GRIDSIZE_8X8 = 3U,
} NvMediaIofaGridSize;

/**
 * \brief Modes for pyramid SGM
 *
 * Applicable to Pyramid SGM IOFA mode only.
 */
typedef enum
{
    /**
     * All pyramid levels of a input and reference frame will be processed
     * in single NvMediaIOFAProcessFrame call.
     *
     * In this mode, the outSurface of previous pyramid level is directly
     * (without any processing) provided as pydHintSurface to process current
     * pyramid level by IOFA driver.
     * pydHintSurface[lvl] = outSurface[lvl+1]
     */
    NVMEDIA_IOFA_PYD_FRAME_MODE = 0U,
    /**
     * A single pyramid level of a input and reference frame will be processed
     * by NvMediaIOFAProcessFrame API
     * In this mode, the outSurface of previous pyramid level can be
     * processed/filtered by application and then provided as pydHinSurface to
     * process current pyramid level.
     *
     * NvMediaIOFAProcessFrame API accept pydHintSurface only in
     * NVMEDIA_IOFA_PYD_LEVEL_MODE.
     * otherwise pydHintSurface is ignored by IOFA driver.
     * pydHintSurface[lvl] = filter(outSurface[lvl+1],....)
     */
    NVMEDIA_IOFA_PYD_LEVEL_MODE = 1U,
} NvMediaIofaPydMode;

/**
 * \brief Defines IOFA Stereo DISPARITY RANGE.
 */
typedef enum
{
    /** Maximum Stereo Disparity Range of 128 pixels. */
    NVMEDIA_IOFA_DISPARITY_RANGE_128 = 0U,
    /** Maximum Stereo Disparity Range of 256 pixels. */
    NVMEDIA_IOFA_DISPARITY_RANGE_256 = 1U,
} NvMediaIofaDisparityRange;

/**
 * \brief Defines IOFA Flow Epipolar Search Range.
 */
typedef enum
{
    /** Maximum Epipolar Flow Search Range of 128 pixels. */
    NVMEDIA_IOFA_EPI_SEARCH_RANGE_128 = 0U,
    /** Maximum Epipolar Flow Search Range of 256 pixels. */
    NVMEDIA_IOFA_EPI_SEARCH_RANGE_256 = 1U,
} NvMediaIofaEpiSearchRange;

/**
 * \brief Nvmedia Iofa Preset.
 */
typedef enum
{
    /** High Quality Preset. */
    NVMEDIA_IOFA_PRESET_HQ = 0U,
    /** High Performance Preset. */
    NVMEDIA_IOFA_PRESET_HP = 1U,
} NvMediaIofaPreset;

/**
 * \brief NvMedia Iofa task status error codes.
 */
typedef enum
{
    /** task is finished successully */
    NvSciSyncTaskStatusOFA_Success                = 0U,
    /** task status error codes */
    NvSciSyncTaskStatusOFA_Error                  = 1U,
    NvSciSyncTaskStatusOFA_Execution_Start        = 2U,
    NvSciSyncTaskStatusOFA_Error_CRC_Mismatch     = 3U,
    NvSciSyncTaskStatusOFA_Error_Timeout          = 4U,
    NvSciSyncTaskStatusOFA_Error_HW               = 5U,
    NvSciSyncTaskStatusOFA_Error_Input_TaskStatus = 6U,
    NvSciSyncTaskStatusOFA_Error_SW               = 7U,
    /** task status support is not enable */
    NvSciSyncTaskStatusOFA_Invalid                = 0XFFFFU
} NvSciSyncTaskStatusOFA;

/**
 * \brief Structure holds Epipolar information.
 */
typedef struct
{
    float   F_Matrix[3][3];
    float   H_Matrix[3][3];
    int32_t epipole_x;
    int32_t epipole_y;
    uint8_t direction;
} NvMediaIofaEpipolarInfo;

/**
 * \brief Holds Co-ordinates for Region of Interest.
 */
typedef struct
{
    uint16_t startX;
    uint16_t startY;
    uint16_t endX;
    uint16_t endY;
} NvMediaIofaROIRectParams;

/**
 * \brief Structure holds ROI information.
 */
typedef struct
{
    uint32_t                numOfROIs;
    NvMediaIofaROIRectParams rectROIParams[NVMEDIA_IOFA_MAX_ROI_SUPPORTED];
} NvMediaIofaROIParams;

/**
 * \brief Nvmedia Iofa Capability structure.
 */
typedef struct
{
    uint16_t minWidth;
    uint16_t minHeight;
    uint16_t maxWidth;
    uint16_t maxHeight;
} NvMediaIofaCapability;

/**
 * \brief Holds an IOFA object created and returned by NvMediaIOFACreate().
 */
typedef struct NvMediaIofa
{
    struct NvMediaIofaPriv *ofaPriv;
} NvMediaIofa;

/**
 * \brief Holds IOFA Initialization API parameters.
 */
typedef struct
{
    NvMediaIofaMode           ofaMode;
    uint8_t                   ofaPydLevel;
    uint16_t                 width[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t                 height[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvMediaIofaGridSize       gridSize[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t                  outWidth[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint16_t                  outHeight[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvMediaIofaDisparityRange dispRange;
    NvMediaIofaPydMode        pydMode;
    bool                      vprMode;
    NvMediaIofaPreset         preset;
    NvMediaIofaEpiSearchRange epiSearchRange;

} NvMediaIofaInitParams;

/**
 * \brief Holds SGM parameters
 *
 * TBD: Add more details about SGM Params with input range.
 */
typedef struct
{
    uint8_t     penalty1[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t     penalty2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    bool        adaptiveP2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t     alphaLog2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    bool        enableDiag[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t     numPasses[NVMEDIA_IOFA_MAX_PYD_LEVEL];
} NvMediaIofaSGMParams;

/**
 * \brief Holds pointers to NvMedia bufObjs containing input and output
 * surfaces.
 */
typedef struct
{
    NvSciBufObj inputSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj refSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj pydHintSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj outSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj costSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
} NvMediaIofaBufArray;

/**
 * \brief Parameters related to input pyramid hint surface.
 */
typedef struct
{
    bool pydHintMagnitudeScale2x;
    bool pydHintWidth2x;
    bool pydHintHeight2x;
} NvMediaIofaPydHintParams;

/**
 * \brief Holds IOFA Process Frame API parameters.
 */
typedef struct
{
    bool                    rightDispMap;
    uint8_t                 currentPydLevel;
    bool                     noopMode;
    NvMediaIofaPydHintParams pydHintParams;
} NvMediaIofaProcessParams;

/**
 * \brief Retrieves the version information for the NvMedia IOFA library.
 *
 * \pre None
 * \post None
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 *
 * \param[out] version  A pointer to a NvMediaVersion structure filled by
 *                      the IOFA library.
 *
 * \return NvMediaStatus, the completion status of the operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if the \a version pointer is NULL.
 */
NvMediaStatus
NvMediaIOFAGetVersion (
    NvMediaVersion *version
);

/**
 * \brief Creates an NvMediaIofa object that can compute optical flow or
 * stereo disparity using two bufObjs.
 *
 * \pre NvMediaIOFAGetVersion()
 * \post NvMediaIofa object is created
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \return Created NvMediaIofa estimator handle if successful, or NULL
 *         otherwise.
 */
NvMediaIofa *
NvMediaIOFACreate (
    void
);

/**
 * \brief Initializes the parameters for optical flow and stereo estimation.
 *
 * \pre NvMediaIOFAGetVersion()
 * \pre NvMediaIOFACreate()
 * \post NvMediaIofa object is returned with initialized parameters
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] ofaPubl   A pointer to the NvMediaIofa estimator to use.
 *                      Non-NULL - valid pointer address
 * \param[in] initParams  A pointer to a structure that specifies
 *                      initialization parameters. Non-NULL - valid address.
 *                      Ranges specific to each member in the structure can
 *                      be found in NvMediaIofaInitParams.
 * \param[in] maxInputBuffering  Maximum number of NvMediaIOFAProcessFrame()
 *                      operations that can be queued by NvMediaIofa.
 *                      If more than \a maxInputBuffering operations are
 *                      queued, NvMediaIOFAProcessFrame() returns an error
 *                      to indicate insufficient buffering.
 *                      The values between 1 to 8, in increments of 1
 *
 * \return The completion status of the operation:
 * - NVMEDIA_STATUS_OK if the call is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAInit (
    NvMediaIofa                 *ofaPubl,
    const NvMediaIofaInitParams *initParams,
    const uint8_t              maxInputBuffering
);

/**
 * \brief Performs IOFA estimation on a specified frame pair.
 *
 * Estimation is based on the difference between \a refFrame and
 * \a inputframe. The output of Optical Flow processing is motion vectors
 * [X, Y Components], and that of Stereo Disparity processing is a disparity
 * surface [X component].
 *
 * \pre NvMediaIOFAInit()
 * \pre NvMediaIOFARegisterNvSciBufObj()
 * \pre NvMediaIOFARegisterNvSciSyncObj()
 * \pre NvMediaIOFASetNvSciSyncObjforEOF()
 * \pre NvMediaIOFAInsertPreNvSciSyncFence()
 * \post Optical Flow Accelerator estimation task is submitted
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] ofaPubl   A pointer to the NvMediaIOFA estimator to use.
 *                      Non-NULL - valid pointer address
 * \param[in] pSurfArray  A pointer to a structure that specifies input and
 *                      output surface parameters.
 *                      Non-NULL - valid address.
 *                      Ranges specific to each member in the structure can
 *                      be found in NvMediaIofaBufArray.
 * \param[in] pProcessParams  A pointer to a structure that specifies process
 *                      frame parameters. Non-NULL - valid address.
 *                      Ranges specific to each member in the structure can
 *                      be found in NvMediaIofaProcessParams.
 * \param[in] pEpiInfo  A pointer to a structure that specifies Epipolar info
 *                      parameters.
 *                      pEpipolarInfo is required argument only when
 *                      \ref ofaMode in \ref NvMediaIofaInitParams is set to
 *                      \ref NVMEDIA_IOFA_MODE_EPIOF.
 *                      Ranges specific to each member in the structure can
 *                      be found in NvMediaIofaEpipolarInfo.
 *                      Epipolar OF support will be added in next release.
 *                      Ignored right now in driver.
 *                      Set it to NULL till EpiPolar OF support is enabled.
 * \param[in] pROIParams  A pointer to a structure that specifies ROI
 *                      parameters. Note: HW will generate output for ROI
 *                      region only and Single ROI should give best perf for
 *                      pyramid SGM Mode.
 *                      pROIParams are optional argument and can be NULL if
 *                      ROI is not set by App.
 *                      Ranges specific to each member in the structure can
 *                      be found in NvMediaIofaROIParams.
 *
 * \return The completion status of the operation:
 * - NVMEDIA_STATUS_OK if the call is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAProcessFrame (
    const NvMediaIofa              *ofaPubl,
    const NvMediaIofaBufArray      *pSurfArray,
    const NvMediaIofaProcessParams *pProcessParams,
    const NvMediaIofaEpipolarInfo  *pEpiInfo,
    const NvMediaIofaROIParams     *pROIParams
);

/**
 * \brief Destroys the created NvMediaIofa object and frees associated
 * resources.
 *
 * \pre NvMediaIOFAUnregisterNvSciSyncObj()
 * \pre NvMediaIOFAUnregisterNvSciBufObj()
 * \post NvMediaIofa object is destroyed
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \param[in] ofaPubl  Pointer to the NvMediaIofa object to destroy,
 *                     returned by NvMediaIOFACreate().
 *                     Non-NULL - valid pointer address
 *
 * \return The completion status of the operation:
 * - NVMEDIA_STATUS_OK if the call is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFADestroy (
    const NvMediaIofa *ofaPubl
);

/**
 * \brief Registers an NvSciBufObj for use with an NvMediaIofa handle.
 *
 * The NvMediaIofa handle maintains a record of all the bufObjs registered
 * using this API.
 *
 * This is a mandatory API which needs to be called before
 * NvMediaIOFAProcessFrame()
 * All NvMediaIOFANvSciBufRegister() API calls must be made before first
 * NvMediaIOFAProcessFrame() API call. Registration of the buffer is done
 * with the same access permission as that of the NvSciBufObj being
 * registered. NvSciBufObj that need to be registered with a reduced
 * permission (Eg: Input buffer accesses being set to read-only) can be done
 * so by first duplicating the NvSciBufObj using
 * NvSciBufObjDupWithReducePerm() followed by a call the register the
 * duplicated NvSciBufObj.
 *
 * Maximum of 192 NvSciBufObj handles can be registered using
 * NvMediaIOFARegisterNvSciSyncObj() API.
 *
 * \pre NvMediaIOFAInit()
 * \pre NvMediaIOFARegisterNvSciSyncObj()
 * \post NvSciBufObj is registered with NvMediaIofa object
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] ofaPubl  NvMediaIofa handle.
 *                     \em Input range: Non-NULL - valid pointer address
 * \param[in] bufObj   An NvSciBufObj object.
 *                     \em Input range: A valid NvSciBufObj
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if ofa, bufObj or accessMode is invalid.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_ERROR in following cases:
 *   - User registers more than 192 bufObjs.
 *   - User registers same bufObj with more than one accessModes.
 *   - User registers same bufObj multiple times.
 */
NvMediaStatus
NvMediaIOFARegisterNvSciBufObj (
    const NvMediaIofa *ofaPubl,
    NvSciBufObj        bufObj
);

/**
 * \brief Un-registers NvSciBufObj which was previously registered with
 * NvMediaIofa using NvMediaIOFARegisterNvSciBufObj().
 *
 * For all NvSciBufObj handles registered with NvMediaIofa using
 * NvMediaIOFARegisterNvSciBufObj() API, NvMediaIOFAUnregisterNvSciBufObj()
 * must be called before calling NvMediaIOFADestroy() API. For unregistration
 * to succeed, it should be ensured that none of the submitted tasks on the
 * bufObj are pending prior to calling NvMediaIOFAUnregisterNvSciBufObj().
 * In order to ensure this, NvMediaIOFAUnregisterNvSciSyncObj() should be
 * called prior to this API on all registered NvSciSyncObj. Post this
 * NvMediaIOFAUnregisterNvSciBufObj() can be successfully called on a valid
 * NvSciBufObj.
 *
 * For deterministic execution of NvMediaIOFAProcessFrame() API,
 * NvMediaIOFAUnregisterNvSciBufObj() must be called only after last
 * NvMediaIOFAProcessFrame() call.
 *
 * \pre NvMediaIOFAUnregisterNvSciSyncObj() [verify that processing is
 *      complete]
 * \post NvSciBufObj is un-registered from NvMediaIofa object
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \param[in] ofaPubl  NvMediaIofa handle.
 *                     \em Input range: Non-NULL - valid pointer address
 * \param[in] bufObj   An NvSciBufObj object.
 *                     \em Input range: A valid NvSciBufObj
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if ofa or bufObj is invalid
 *   NvMediaIOFARegisterNvSciBufObj() API.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_ERROR in following cases:
 *   - User unregisters an NvSciBufObj which is not previously registered
 *     using NvMediaIOFARegisterNvSciBufObj() API.
 *   - User unregisters an NvSciBufObj multiple times.
 */
NvMediaStatus
NvMediaIOFAUnregisterNvSciBufObj (
    const NvMediaIofa *ofaPubl,
    NvSciBufObj       bufObj
);

/**
 * \brief Get the SGM configuration parameters being used.
 *
 * \pre NvMediaIOFAInit()
 * \post SGM params are returned in structure NvMediaIofaSGMParams
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] ofaPubl     Pointer to the NvMediaIofa object to use.
 *                        Non-NULL - valid pointer address
 * \param[out] pSGMParams A pointer to a structure that specifies SGM
 *                        parameters.
 *
 * \return The completion status of the operation:
 * - NVMEDIA_STATUS_OK if the function call is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAGetSGMConfigParams (
    const NvMediaIofa    *ofaPubl,
    NvMediaIofaSGMParams *pSGMParams
);

/**
 * \brief Set the SGM configuration parameters to be used.
 *
 * \pre NvMediaIOFAInit()
 * \post SGM params are set in IOFA driver and will be used for next frame
 *       processing.
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] ofaPubl     Pointer to the NvMediaIofa object to use.
 *                        Non-NULL - valid pointer address
 * \param[in] pSGMParams  A pointer to a structure that specifies SGM
 *                        parameters.
 *
 * \return The completion status of the operation:
 * - NVMEDIA_STATUS_OK if the function call is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFASetSGMConfigParams (
    const NvMediaIofa          *ofaPubl,
    const NvMediaIofaSGMParams *pSGMParams
);

/**
 * \brief Get IOFA Capability.
 *
 * This function returns ofa hw capabilities.
 *
 * \pre NvMediaIOFACreate
 * \post hw capabilities are returned in structure NvMediaIofaCapability
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * <b>Considerations for Safety</b>:
 * - Operation Mode: Init
 *
 * \param[in] ofaPubl      Pointer to the NvMediaIofa object to use.
 *                         Non-NULL - valid pointer address
 * \param[in] mode         one of the value from NvMediaIofaMode.
 * \param[out] pCapability A pointer to a structure that contains capability
 *                         data
 *
 * \return The completion status of the operation:
 * - NVMEDIA_STATUS_OK if the call is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters are invalid.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_ERROR if there is an internal error in processing.
 */
NvMediaStatus
NvMediaIOFAGetCapability (
    const NvMediaIofa     *ofaPubl,
    const NvMediaIofaMode mode,
    NvMediaIofaCapability *pCapability
);

/**
 * \brief Fills the NvMediaIofa specific NvSciBuf attributes which than then
 * be used to allocate an NvSciBufObj that NvMediaIofa can consume.
 *
 * This function updates the input NvSciBufAttrList with values equivalent
 * to the following public attribute key-values:
 * NvSciBufGeneralAttrKey_PeerHwEngineArray set to
 * - NvSciBufHwEngName: NvSciBufHwEngName_OFA
 * - NvSciBufPlatformName: NvSciBufPlatformName_Orin
 *
 * This function assumes that \a attrlist is a valid NvSciBufAttrList created
 * by the caller by a call to NvSciBufAttrListCreate.
 *
 * \pre NvMediaIOFAGetVersion()
 * \post NvSciBufAttrList populated with NvMediaIofa specific NvSciBuf
 *       attributes. The caller can then set attributes specific to the type
 *       of surface, reconcile attribute lists and allocate an NvSciBufObj.
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[out] attrlist  A pointer to an NvSciBufAttrList structure where
 *                       NvMediaIofa places the NvSciBuf attributes.
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a attrlist is NULL
 */
NvMediaStatus
NvMediaIOFAFillNvSciBufAttrList (
    NvSciBufAttrList attrlist
);

/**
 * \brief Fills the NvMediaIofa specific NvSciSync attributes.
 *
 * This function assumes that \a attrlist is a valid NvSciSyncAttrList.
 *
 * This function updates the input NvSciSyncAttrList with values equivalent
 * to the following public attribute key-values:
 * NvSciSyncAttrKey_RequiredPerm set to
 * - NvSciSyncAccessPerm_WaitOnly for clienttype NVMEDIA_WAITER
 * - NvSciSyncAccessPerm_SignalOnly for clienttype NVMEDIA_SIGNALER
 * - NvSciSyncAccessPerm_WaitSignal for clienttype NVMEDIA_SIGNALER_WAITER
 * NvSciSyncAttrKey_PrimitiveInfo set to
 * - NvSciSyncAttrValPrimitiveType_Syncpoint
 *
 * The application must not set these attributes in the NvSciSyncAttrList
 * passed as an input to this function.
 *
 * \pre NvMediaIOFACreate()
 * \post NvSciSyncAttrList populated with NvMediaIofa specific NvSciSync
 *       attributes
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *       to achieve synchronization between the engines
 *
 * \param[in] ofaPubl     A pointer to the NvMediaIofa object.
 *                        \em Input range: Can be NULL or Non-NULL valid
 *                        pointer address
 * \param[out] attrlist   A pointer to an NvSciSyncAttrList structure where
 *                        NvMedia places NvSciSync attributes.
 * \param[in] clienttype  Indicates whether the NvSciSyncAttrList requested
 *                        for an NvMediaIofa signaler or an NvMediaIofa
 *                        waiter.
 *                        \em Input range: Entries in
 *                        NvMediaNvSciSyncClientType enumeration
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a attrlist is NULL, or any of the
 *   public attributes listed above are already set.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_OUT_OF_MEMORY if there is not enough memory for the
 *   requested operation.
 */
NvMediaStatus
NvMediaIOFAFillNvSciSyncAttrList (
    const NvMediaIofa          *ofaPubl,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * \brief Registers an NvSciSyncObj with NvMediaIofa.
 *
 * Every NvSciSyncObj (even duplicate objects) used by NvMediaIofa must be
 * registered by a call to this function before it is used. Only the exact
 * same registered NvSciSyncObj can be passed to
 * NvMediaIOFASetNvSciSyncObjforEOF(), NvMediaIOFAGetEOFNvSciSyncFence(),
 * or NvMediaIOFAUnregisterNvSciSyncObj().
 *
 * For a given NvMediaIofa handle, one NvSciSyncObj can be registered as one
 * NvMediaNvSciSyncObjType only. For each NvMediaNvSciSyncObjType, a maximum
 * of 16 NvSciSyncObjs can be registered.
 *
 * \pre NvMediaIOFAFillNvSciSyncAttrList()
 * \post NvSciSyncObj registered with NvMediaIofa
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *       to achieve synchronization between the engines
 *
 * \param[in] ofaPubl      A pointer to the NvMediaIofa object.
 *                         \em Input range: Non-NULL - valid pointer address
 * \param[in] syncobjtype  Determines how \a nvscisync is used by
 *                         \a ofaPubl.
 *                         \em Input range: Entries in
 *                         NvMediaNvSciSyncObjType enumeration
 * \param[in] syncObj      The NvSciSyncObj to be registered with
 *                         \a ofaPubl.
 *                         \em Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a ofaPubl is NULL or \a syncobjtype is
 *   not a valid NvMediaNvSciSyncObjType. only NVMEDIA_EOFSYNCOBJ and
 *   NVMEDIA_PRESYNCOBJ supported.
 * - NVMEDIA_STATUS_NOT_SUPPORTED if \a nvscisync is not a compatible
 *   NvSciSiyncObj which NvMediaIofa can support.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObjs are already
 *   registered for the given \a syncobjtype, or if \a nvscisync is already
 *   registered with the same \a ofaPubl handle for a different
 *   \a syncobjtype.
 */
NvMediaStatus
NvMediaIOFARegisterNvSciSyncObj (
    const NvMediaIofa       *ofaPubl,
    NvMediaNvSciSyncObjType syncobjtype,
    NvSciSyncObj            syncObj
);

/**
 * \brief Unregisters an NvSciSyncObj with NvMediaIofa.
 *
 * Every NvSciSyncObj registered with NvMediaIofa by
 * NvMediaIOFARegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIOFAUnregisterNvSciBufObj to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * NvMediaIOFAProcessFrame() operation that uses the NvSciSyncObj has
 * completed. If this function is called while NvSciSyncObj is still in use
 * by any NvMediaIOFAProcessFrame() operation, the API returns
 * NVMEDIA_STATUS_PENDING to indicate the same. NvSciSyncFenceWait() API can
 * be called on the EOF NvSciSyncFence obtained post the last call to
 * NvMediaIOFAProcessFrame() to wait for the associated tasks to complete.
 * The EOF NvSciSyncFence would have been previously obtained via a call to
 * NvMediaIOFAGetEOFNvSciSyncFence().
 *
 * \pre NvMediaIOFAProcessFrame()
 * \pre NvSciSyncFenceWait() [verify that processing is complete]
 * \post NvSciSyncObj un-registered with NvMediaIofa
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *       to achieve synchronization between the engines
 *
 * \param[in] ofaPubl    A pointer to the NvMediaIofa object.
 *                       \em Input range: Non-NULL - valid pointer address
 * \param[in] syncObj    An NvSciSyncObj to be unregistered with \a ofaPubl.
 *                       \em Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if ofaPubl is NULL, or \a nvscisync is not
 *   registered with \a ofaPubl.
 * - NVMEDIA_STATUS_PENDING if the NvSciSyncObj is still in use, i.e., the
 *   submitted task is still in progress. In this case, the application can
 *   choose to wait for operations to complete on the output surface using
 *   NvSciSyncFenceWait() or re-try the NvMediaIOFAUnregisterNvSciBufObj()
 *   API call, until the status returned is not NVMEDIA_STATUS_PENDING.
 * - NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect system
 *   state.
 * - NVMEDIA_STATUS_ERROR if \a ofaPubl was destroyed before this function
 *   was called.
 */
NvMediaStatus
NvMediaIOFAUnregisterNvSciSyncObj (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      syncObj
);

/**
 * \brief Specifies the NvSciSyncObj to be used for an EOF NvSciSyncFence.
 *
 * To use NvMediaIOFAGetEOFNvSciSyncFence(), the application must call
 * NvMediaIOFASetNvSciSyncObjforEOF() before it calls
 * NvMediaIOFAProcessFrame().
 *
 * NvMediaIOFASetNvSciSyncObjforEOF() currently may be called only once
 * before each call to NvMediaIOFAProcessFrame(). The application may choose
 * to call this function only once before the first call to
 * NvMediaIOFAProcessFrame().
 *
 * \pre NvMediaIOFARegisterNvSciSyncObj()
 * \post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *       to achieve synchronization between the engines
 *
 * \param[in] ofaPubl       A pointer to the NvMediaIofa object.
 *                          \em Input range: Non-NULL - valid pointer address
 * \param[in] nvscisyncEOF  A registered NvSciSyncObj which is to be
 *                          associated with EOF NvSciSyncFence.
 *                          \em Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a ofaPubl is NULL, or if
 *   \a nvscisyncEOF is not registered with \a ofaPubl as either type
 *   NVMEDIA_EOFSYNCOBJ or NVMEDIA_EOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIOFASetNvSciSyncObjforEOF (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      nvscisyncEOF
);

/**
 * \brief Sets an NvSciSyncFence as a prefence for an
 * NvMediaIOFAProcessFrame() NvSciSyncFence operation.
 *
 * You must call NvMediaIOFAInsertPreNvSciSyncFence() before you call
 * NvMediaIOFAProcessFrame(). The NvMediaIOFAProcessFrame() operation is
 * started only after the expiry of the \a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIOFAInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIOFAProcessFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * \endcode
 * the NvMediaIOFAProcessFrame() operation is assured to start only after
 * the expiry of \a prenvscisyncfence.
 *
 * You can set a maximum of NVMEDIA_IOFA_MAX_PRENVSCISYNCFENCES prefences by
 * calling NvMediaIOFAInsertPreNvSciSyncFence() before
 * NvMediaIOFAProcessFrame(). After the call to NvMediaIOFAProcessFrame(),
 * all NvSciSyncFences previously inserted by
 * NvMediaIOFAInsertPreNvSciSyncFence() are removed, and they are not reused
 * for the subsequent NvMediaIOFAProcessFrame() calls.
 *
 * \pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * \post Pre-NvSciSync fence is set
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *       to achieve synchronization between the engines
 *
 * \param[in] ofaPubl            A pointer to the NvMediaIofa object.
 *                               \em Input range: Non-NULL - valid pointer
 *                               address
 * \param[in] prenvscisyncfence  A pointer to NvSciSyncFence.
 *                               \em Input range: Non-NULL - valid pointer
 *                               address
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a ofaPubl is not a valid NvMediaIofa
 *   handle, or \a prenvscisyncfence is NULL, or if \a prenvscisyncfence was
 *   not generated with an NvSciSyncObj that was registered with \a ofaPubl
 *   as either NVMEDIA_PRESYNCOBJ or NVMEDIA_EOF_PRESYNCOBJ type.
 * - NVMEDIA_STATUS_NOT_SUPPORTED if NvMediaIOFAInsertPreNvSciSyncFence()
 *   has already been called at least NVMEDIA_IOFA_MAX_PRENVSCISYNCFENCES
 *   times with the same \a ofaPubl handle before an
 *   NvMediaIOFAProcessFrame() call.
 */
NvMediaStatus
NvMediaIOFAInsertPreNvSciSyncFence (
    const NvMediaIofa    *ofaPubl,
    const NvSciSyncFence *prenvscisyncfence
);

/**
 * \brief Gets EOF NvSciSyncFence for an NvMediaIOFAProcessFrame() operation.
 *
 * The EOF NvSciSyncFence associated with an NvMediaIOFAProcessFrame()
 * operation is an NvSciSyncFence. Its expiry indicates that the
 * corresponding NvMediaIOFAProcessFrame() operation has finished.
 *
 * This function returns the EOF NvSciSyncFence associated with the last
 * NvMediaIOFAProcessFrame() call. NvMediaIOFAGetEOFNvSciSyncFence() must be
 * called after an NvMediaIOFAProcessFrame() call.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIOFAProcessFrame(handle, srcsurf, srcrect, picparams, instanceid);
 * nvmstatus = NvMediaIOFAGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * \endcode
 * expiry of \a eofnvscisyncfence indicates that the preceding
 * NvMediaIOFAProcessFrame() operation has finished.
 *
 * \pre NvMediaIOFASetNvSciSyncObjforEOF()
 * \pre NvMediaIOFAProcessFrame()
 * \post EOF NvSciSync fence for a submitted task is obtained
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIofa object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *       to achieve synchronization between the engines
 *
 * \param[in] ofaPubl           A pointer to the NvMediaIofa object.
 *                              \em Input range: Non-NULL - valid pointer
 *                              address
 * \param[in] eofnvscisyncobj   An EOF NvSciSyncObj associated with the
 *                              NvSciSyncFence which is being requested.
 *                              \em Input range: A valid NvSciSyncObj
 * \param[out] eofnvscisyncfence  A pointer to the EOF NvSciSyncFence.
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a ofaPubl is not a valid NvMediaIofa
 *   handle, \a eofnvscisyncfence is NULL, or \a eofnvscisyncobj is not
 *   registered with \a ofaPubl as type NVMEDIA_EOFSYNCOBJ or
 *   NVMEDIA_EOF_PRESYNCOBJ.
 * - NVMEDIA_STATUS_ERROR if the function was called before
 *   NvMediaIOFAProcessFrame() was called.
 */
NvMediaStatus
NvMediaIOFAGetEOFNvSciSyncFence (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/*
 * \defgroup 6x_history_nvmedia_iofa History
 * Provides change history for the NvMedia IOFA API.
 *
 * \section 6x_history_nvmedia_iofa Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 *
 * <b> Version 1.1 </b> April 10, 2023
 * - Removed API NvMediaIOFAGetProfileData
 */

/** @} */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif // NVMEDIA_IOFA_H
