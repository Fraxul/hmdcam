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
 */

// Based on the NvMedia API from DriveOS 6.0.10
//
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/group__x__image__jpeg__decode__api.html
//

#ifndef NVMEDIA_IJPD_H
#define NVMEDIA_IJPD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_core.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvmedia_common_encode_decode.h"

#define NVMEDIA_IJPD_VERSION_MAJOR   1

#define NVMEDIA_IJPD_VERSION_MINOR   0

#define NVMEDIA_IJPD_VERSION_PATCH   0

#define NVMEDIA_IJPD_MAX_PRENVSCISYNCFENCES        (16U)

#define NVMEDIA_JPEG_DEC_ATTRIBUTE_ALPHA_VALUE     (1 << 0)

#define NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD  (1 << 1)

#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_0          0

#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_90         1

#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_180        2

#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_270        3

#define NVMEDIA_IJPD_RENDER_FLAG_FLIP_HORIZONTAL   (1 << 2)

#define NVMEDIA_IJPD_RENDER_FLAG_FLIP_VERTICAL     (1 << 3)

#define NVMEDIA_MAX_JPEG_APP_MARKERS               16

typedef enum {
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601,
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709,
    NVMEDIA_IJPD_COLOR_STANDARD_SMPTE_240M,
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601_ER,
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709_ER
} NvMediaIJPDColorStandard;

typedef struct {
    NvMediaIJPDColorStandard colorStandard;

    uint32_t alphaValue;
} NVMEDIAJPEGDecAttributes;

typedef struct {
  uint16_t marker;
  uint16_t len;
  void    *pMarker;
} NvMediaJPEGAppMarkerInfo;

typedef struct {
  uint16_t width;
  uint16_t height;
  uint8_t  partialAccel;
  uint8_t  num_app_markers;
  NvMediaJPEGAppMarkerInfo appMarkerInfo[NVMEDIA_MAX_JPEG_APP_MARKERS];
} NVMEDIAJPEGDecInfo;

typedef struct NvMediaIJPD NvMediaIJPD;

NvMediaStatus
NvMediaIJPDGetVersion(
    NvMediaVersion *version
);

NvMediaIJPD *
NvMediaIJPDCreate(
    uint16_t maxWidth,
    uint16_t maxHeight,
    uint32_t maxBitstreamBytes,
    bool supportPartialAccel,
    NvMediaJPEGInstanceId instanceId
);

void NvMediaIJPDDestroy(NvMediaIJPD *decoder);

NvMediaStatus
NvMediaIJPDResize (
   NvMediaIJPD *decoder,
   uint16_t maxWidth,
   uint16_t maxHeight,
   uint32_t maxBitstreamBytes
);

NvMediaStatus
NvMediaIJPDSetAttributes(
   const NvMediaIJPD *decoder,
   uint32_t attributeMask,
   const void *attributes
);

NvMediaStatus
NvMediaIJPDGetInfo (
   NVMEDIAJPEGDecInfo *info,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams
);

NvMediaStatus
NvMediaIJPDRender(
   const NvMediaIJPD *decoder,
   NvSciBufObj target,
   const NvMediaRect *srcRect,
   const NvMediaRect *dstRect,
   uint8_t downscaleLog2,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams,
   uint32_t flags,
   NvMediaJPEGInstanceId instanceId
);

NvMediaStatus
NvMediaIJPDRenderYUV(
   const NvMediaIJPD *decoder,
   NvSciBufObj target,
   uint8_t downscaleLog2,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams,
   uint32_t flags,
   NvMediaJPEGInstanceId instanceId
);

NvMediaStatus
NvMediaIJPDRegisterNvSciBufObj(
    const NvMediaIJPD   *decoder,
    NvSciBufObj         bufObj
);

// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIJPDUnregisterNvSciBufObj(
    const NvMediaIJPD    *decoder,
    NvSciBufObj          bufObj
);

NvMediaStatus
NvMediaIJPDFillNvSciBufAttrList(
    NvMediaJPEGInstanceId     instanceId,
    NvSciBufAttrList          attrlist
);

NvMediaStatus
NvMediaIJPDFillNvSciSyncAttrList(
    const NvMediaIJPD           *decoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

NvMediaStatus
NvMediaIJPDRegisterNvSciSyncObj(
    const NvMediaIJPD           *decoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

NvMediaStatus
NvMediaIJPDUnregisterNvSciSyncObj(
    const NvMediaIJPD  *decoder,
    NvSciSyncObj      nvscisync
);

NvMediaStatus
NvMediaIJPDSetNvSciSyncObjforEOF(
    const NvMediaIJPD      *decoder,
    NvSciSyncObj          nvscisyncEOF
);

NvMediaStatus
NvMediaIJPDInsertPreNvSciSyncFence(
    const NvMediaIJPD         *decoder,
    const NvSciSyncFence     *prenvscisyncfence
);

NvMediaStatus
NvMediaIJPDGetEOFNvSciSyncFence(
    const NvMediaIJPD        *decoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);


/*
 * @defgroup 6x_history_nvmedia_ijpd History
 * Provides change history for the NvMedia Image Jpeg Decode API.
 *
 * \section 6x_history_nvmedia_ijpd Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IJPD_H */

