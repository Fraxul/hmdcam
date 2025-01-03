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
 
#define NVMEDIA_IOFA_VERSION_MAJOR            1
 
#define NVMEDIA_IOFA_VERSION_MINOR            4
 
#define NVMEDIA_IOFA_VERSION_PATCH            0
 
#define NVMEDIA_IOFA_MAX_PYD_LEVEL            7U
 
#define NVMEDIA_IOFA_MAX_ROI_SUPPORTED        32U
 
#define NVMEDIA_IOFA_MAX_PRENVSCISYNCFENCES   16U
 
typedef enum
{
    NVMEDIA_IOFA_MODE_STEREO = 0U,
    NVMEDIA_IOFA_MODE_PYDOF  = 1U,
    NVMEDIA_IOFA_MODE_EPIOF  = 2U,
} NvMediaIofaMode;
 
typedef enum
{
    NVMEDIA_IOFA_GRIDSIZE_1X1 = 0U,
    NVMEDIA_IOFA_GRIDSIZE_2X2 = 1U,
    NVMEDIA_IOFA_GRIDSIZE_4X4 = 2U,
    NVMEDIA_IOFA_GRIDSIZE_8X8 = 3U,
} NvMediaIofaGridSize;
 
typedef enum
{
    NVMEDIA_IOFA_PYD_FRAME_MODE = 0U,
    NVMEDIA_IOFA_PYD_LEVEL_MODE = 1U,
} NvMediaIofaPydMode;
 
typedef enum
{
    NVMEDIA_IOFA_DISPARITY_RANGE_128 = 0U,
    NVMEDIA_IOFA_DISPARITY_RANGE_256 = 1U,
} NvMediaIofaDisparityRange;
 
typedef enum
{
    NVMEDIA_IOFA_EPI_SEARCH_RANGE_128 = 0U,
    NVMEDIA_IOFA_EPI_SEARCH_RANGE_256 = 1U,
} NvMediaIofaEpiSearchRange;
 
typedef enum
{
    NVMEDIA_IOFA_PRESET_HQ = 0U,
    NVMEDIA_IOFA_PRESET_HP = 1U,
} NvMediaIofaPreset;
 
typedef enum
{
    NvSciSyncTaskStatusOFA_Success                = 0U,
    NvSciSyncTaskStatusOFA_Error                  = 1U,
    NvSciSyncTaskStatusOFA_Execution_Start        = 2U,
    NvSciSyncTaskStatusOFA_Error_CRC_Mismatch     = 3U,
    NvSciSyncTaskStatusOFA_Error_Timeout          = 4U,
    NvSciSyncTaskStatusOFA_Error_HW               = 5U,
    NvSciSyncTaskStatusOFA_Error_Input_TaskStatus = 6U,
    NvSciSyncTaskStatusOFA_Error_SW               = 7U,
    NvSciSyncTaskStatusOFA_Invalid                = 0XFFFFU
} NvSciSyncTaskStatusOFA;
typedef struct
{
    float   F_Matrix[3][3];
    float   H_Matrix[3][3];
    int32_t epipole_x;
    int32_t epipole_y;
    uint8_t direction;
} NvMediaIofaEpipolarInfo;
 
typedef struct
{
    uint16_t startX;
    uint16_t startY;
    uint16_t endX;
    uint16_t endY;
} NvMediaIofaROIRectParams;
 
typedef struct
{
    uint32_t                numOfROIs;
    NvMediaIofaROIRectParams rectROIParams[NVMEDIA_IOFA_MAX_ROI_SUPPORTED];
} NvMediaIofaROIParams;
 
typedef struct
{
    uint16_t minWidth;
    uint16_t minHeight;
    uint16_t maxWidth;
    uint16_t maxHeight;
} NvMediaIofaCapability;
 
typedef struct NvMediaIofa
{
    struct NvMediaIofaPriv *ofaPriv;
} NvMediaIofa;
 
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
 
typedef struct
{
    uint8_t     penalty1[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t     penalty2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    bool        adaptiveP2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t     alphaLog2[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    bool        enableDiag[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    uint8_t     numPasses[NVMEDIA_IOFA_MAX_PYD_LEVEL];
} NvMediaIofaSGMParams;
 
typedef struct
{
    NvSciBufObj inputSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj refSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj pydHintSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj outSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
    NvSciBufObj costSurface[NVMEDIA_IOFA_MAX_PYD_LEVEL];
} NvMediaIofaBufArray;
 
typedef struct
{
    bool pydHintMagnitudeScale2x;
    bool pydHintWidth2x;
    bool pydHintHeight2x;
} NvMediaIofaPydHintParams;
 
typedef struct
{
    bool                    rightDispMap;
    uint8_t                 currentPydLevel;
    bool                     noopMode;
    NvMediaIofaPydHintParams pydHintParams;
} NvMediaIofaProcessParams;
 
NvMediaStatus
NvMediaIOFAGetVersion (
    NvMediaVersion *version
);
 
NvMediaIofa *
NvMediaIOFACreate (
    void
);
 
NvMediaStatus
NvMediaIOFAInit (
    NvMediaIofa                 *ofaPubl,
    const NvMediaIofaInitParams *initParams,
    const uint8_t              maxInputBuffering
);
 
NvMediaStatus
NvMediaIOFAProcessFrame (
    const NvMediaIofa              *ofaPubl,
    const NvMediaIofaBufArray      *pSurfArray,
    const NvMediaIofaProcessParams *pProcessParams,
    const NvMediaIofaEpipolarInfo  *pEpiInfo,
    const NvMediaIofaROIParams     *pROIParams
);
 
NvMediaStatus
NvMediaIOFADestroy (
    const NvMediaIofa *ofaPubl
);
 
NvMediaStatus
NvMediaIOFARegisterNvSciBufObj (
    const NvMediaIofa *ofaPubl,
    NvSciBufObj        bufObj
);
 
NvMediaStatus
NvMediaIOFAUnregisterNvSciBufObj (
    const NvMediaIofa *ofaPubl,
    NvSciBufObj       bufObj
);
 
NvMediaStatus
NvMediaIOFAGetSGMConfigParams (
    const NvMediaIofa    *ofaPubl,
    NvMediaIofaSGMParams *pSGMParams
);
 
NvMediaStatus
NvMediaIOFASetSGMConfigParams (
    const NvMediaIofa          *ofaPubl,
    const NvMediaIofaSGMParams *pSGMParams
);
 
NvMediaStatus
NvMediaIOFAGetCapability (
    const NvMediaIofa     *ofaPubl,
    const NvMediaIofaMode mode,
    NvMediaIofaCapability *pCapability
);
 
NvMediaStatus
NvMediaIOFAFillNvSciBufAttrList (
    NvSciBufAttrList attrlist
);
 
NvMediaStatus
NvMediaIOFAFillNvSciSyncAttrList (
    const NvMediaIofa          *ofaPubl,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);
 
NvMediaStatus
NvMediaIOFARegisterNvSciSyncObj (
    const NvMediaIofa       *ofaPubl,
    NvMediaNvSciSyncObjType syncobjtype,
    NvSciSyncObj            syncObj
);
 
NvMediaStatus
NvMediaIOFAUnregisterNvSciSyncObj (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      syncObj
);
 
NvMediaStatus
NvMediaIOFASetNvSciSyncObjforEOF (
    const NvMediaIofa *ofaPubl,
    NvSciSyncObj      nvscisyncEOF
);
 
NvMediaStatus
NvMediaIOFAInsertPreNvSciSyncFence (
    const NvMediaIofa    *ofaPubl,
    const NvSciSyncFence *prenvscisyncfence
);
 
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
 
#ifdef __cplusplus
}     /* extern "C" */
#endif
 
#endif // NVMEDIA_IOFA_H
