/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__common__encode__decode_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: Common Types for Encode and Decode
 *
 * This file contains common types and definitions for decode and encode operations.
 */

#ifndef NVMEDIA_COMMON_ENCODE_DECODE_H
#define NVMEDIA_COMMON_ENCODE_DECODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * \brief Specifies NVJPG HW instance ID.
 */
typedef enum  {
    NVMEDIA_JPEG_INSTANCE_0 = 0,
    NVMEDIA_JPEG_INSTANCE_1,
    NVMEDIA_JPEG_INSTANCE_AUTO
} NvMediaJPEGInstanceId;

/**
 * \brief Video codec type.
 */
typedef enum {
    NVMEDIA_VIDEO_CODEC_H264,
    NVMEDIA_VIDEO_CODEC_VC1,
    NVMEDIA_VIDEO_CODEC_VC1_ADVANCED,
    NVMEDIA_VIDEO_CODEC_MPEG1,
    NVMEDIA_VIDEO_CODEC_MPEG2,
    NVMEDIA_VIDEO_CODEC_MPEG4,
    NVMEDIA_VIDEO_CODEC_MJPEG,
    NVMEDIA_VIDEO_CODEC_VP8,
    NVMEDIA_VIDEO_CODEC_HEVC,
    NVMEDIA_VIDEO_CODEC_VP9,
    NVMEDIA_VIDEO_CODEC_H264_MVC,
    NVMEDIA_VIDEO_CODEC_HEVC_MV,
    NVMEDIA_VIDEO_CODEC_AV1,
    NVMEDIA_VIDEO_CODEC_END
} NvMediaVideoCodec;

/**
 * \brief Holds an application data buffer containing compressed video data.
 */
typedef struct {
    uint8_t *bitstream;         /**< A pointer to the bitstream data bytes. */
    uint32_t bitstreamBytes;    /**< The number of data bytes. */
    uint32_t bitstreamSize;     /**< Size of bitstream array. */
} NvMediaBitstreamBuffer;

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_COMMON_ENCODE_DECODE_H */
