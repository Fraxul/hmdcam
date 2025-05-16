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


#ifndef NVMEDIA_COMMON_ENCODE_DECODE_H
#define NVMEDIA_COMMON_ENCODE_DECODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef enum  {
    NVMEDIA_JPEG_INSTANCE_0 = 0,
    NVMEDIA_JPEG_INSTANCE_1,
    NVMEDIA_JPEG_INSTANCE_AUTO
} NvMediaJPEGInstanceId;

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

typedef struct {
    uint8_t *bitstream;
    uint32_t bitstreamBytes;
    uint32_t bitstreamSize;
} NvMediaBitstreamBuffer;

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_COMMON_ENCODE_DECODE_H */
