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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__iep__output__extradata_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: NvMedia Image Encode Processing Output ExtraData
 *
 * This file contains the output extradata definition for "Image Encode Processing API".
 */

#ifndef NVMEDIA_IEP_OUTPUT_EXTRA_DATA_H
#define NVMEDIA_IEP_OUTPUT_EXTRA_DATA_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/** \brief Maximum number of reference pictures including current frame. */
#define NVMEDIA_ENCODE_MAX_RPS_SIZE 17U

/** \brief Token that indicates the start of NvMediaEncodeMVBufferHeader. */
#define MV_BUFFER_HEADER 0xFFFEFDFCU

/**
 * \brief Enumeration of possible frame types - common to H264, H265.
 */
typedef enum {
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_P=0,
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_B,
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_I,
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_IDR,
    NVMEDIA_ENCODE_H26X_FRAME_TYPE_END
} NvMediaEncodeH26xFrameType;

/**
 * \brief Holds a encoder frames property.
 */
typedef struct {
    uint32_t ulFrameId;             /**< Unique Frame ID. */
    bool bIdrFrame;                 /**< Is the frame an IDR frame. */
    bool bLTRefFrame;               /**< Is the frame a Long Term Ref Frame. */
    uint32_t  ulPictureOrderCnt;    /**< Picture order count of the frame. */
    uint32_t  ulFrameNum;           /**< Frame Number of the frame. */
    uint32_t  ulLTRFrameIdx;        /**< LongTermFrameIdx of the picture. */
} NvMediaEncodeFrameFullProp;

/**
 * \brief Holds the statistics from encode profiling.
 */
typedef struct {
    uint32_t  ulCycleCount;         /**< Hardware Cycle Count. */
    uint32_t  ulPresetTime;         /**< Time taken for setting the preset in feedframe. */
    uint32_t  ulFlushTime;          /**< Hardware Flush Time. */
    uint32_t  ulEncodeTime;         /**< Hardware Encode Time. */
    uint32_t  ulFetchTime;          /**< Time taken to fetch the encoded bitstream once a frame is passed for encoded. */
} NvMediaFrameStats;

/**
 * \brief Holds a codec-specific extradata output.
 */
typedef union {
    struct {
        NvMediaEncodeH26xFrameType eFrameType;  /**< Frame type of the encoded frame. */
        bool bRefPic;                           /**< Is this a reference frame. */
        bool bIntraRefresh;                     /**< Is this an intra refresh frame. */
        uint32_t uIntraMBCount;                 /**< Count of the number of intra MBs. */
        uint32_t uInterMBCount;                 /**< Count of the number of inter MBs. */
    } h264Extradata;
    struct {
        NvMediaEncodeH26xFrameType eFrameType;  /**< Frame type of the encoded frame. */
        bool bRefPic;                           /**< Is this a reference frame. */
        bool bIntraRefresh;                     /**< Is this an intra refresh frame. */
        uint32_t uIntraCU32x32Count;            /**< Count of the number of intra 32x32 CUs. */
        uint32_t uInterCU32x32Count;            /**< Count of the number of inter 32x32 CUs. */
        uint32_t uIntraCU16x16Count;            /**< Count of the number of intra 16x16 CUs. */
        uint32_t uInterCU16x16Count;            /**< Count of the number of inter 16x16 CUs. */
        uint32_t uIntraCU8x8Count;              /**< Count of the number of intra 8x8 CUs. */
        uint32_t uInterCU8x8Count;              /**< Count of the number of inter 8x8 CUs. */
    } h265Extradata;
} NvMediaEncodeCodecExData;

/**
 * \brief Header format that defines motion vector output.
 */
typedef struct
{
    uint32_t MagicNum;              /**< Used to verify the integrity of the header. */
    uint32_t buffersize;            /**< Size of motion vector output (excluding header size), i.e., the size of MV data in the bitstream post the header. */
    uint16_t blocksize;             /**< Macro Block size. */
    uint16_t width_in_blocksize;    /**< Input frame width in terms of _blocksize_. */
    uint16_t height_in_blocksize;   /**< Input frame height in terms of _blocksize_. */
    uint16_t reserved;              /**< Reserved. */
} NvMediaEncodeMVBufferHeader;

/**
 * \brief Motion Vector format - motion vectors for each of the macro blocks are dumped in this format contiguously in memory, beyond the NvMediaEncodeMVBufferHeader in the bitstream.
 */
typedef struct {
    int32_t mv_x;                   /**< X component of the motion vector pertaining to 1 macro block. */
    int32_t mv_y;                   /**< Y component of the motion vector pertaining to 1 macro block. */
} NvMediaEncodeMVData;

/**
 * \brief Holds the encoder output extradata configuration.
 */
typedef struct {
    uint32_t ulExtraDataSize;       /**< Size of this extradata structure. */
    bool bkeyFrame;                 /**< Format of input H264 data. */
    bool bEndOfFrame;               /**< Slice end or frame end in the packet for application to handle packets When slice encode is completed but not complete frame then it will set bEndOfFrame to false. */
    uint32_t ulHdrSize;             /**< Size of SPS/PPS header if it passed with output buffer. */
    int16_t  AvgQP;                 /**< Average QP index of the encoded frame. */
    bool bIsGoldenOrAlternateFrame; /**< Flag for vp8 reference frame information. */
    bool bValidReconCRC;            /**< Whether Recon CRC for Recon frame is present. */
    uint32_t ulReconCRC_Y;          /**< Recon CRC for Y component when ReconCRC generation is enabled. */
    uint32_t ulReconCRC_U;          /**< Recon CRC for U component when ReconCRC generation is enabled. */
    uint32_t ulReconCRC_V;          /**< Recon CRC for V component when ReconCRC generation is enabled. */
    uint32_t ulFrameMinQP;          /**< Rate Control Feedback. Minimum QP used for this frame */
    uint32_t ulFrameMaxQP;          /**< Maximum QP used for this frame. */
    bool bRPSFeedback;              /**< RPS Feedback. Reference Picture Set data output enabled. */
    uint32_t  ulCurrentRefFrameId;  /**< frame id of reference frame to be used for motion search, ignored for IDR. */
    uint32_t  ulActiveRefFrames;    /**< Number of valid entries in RPS. */
    NvMediaEncodeFrameFullProp RPSList[NVMEDIA_ENCODE_MAX_RPS_SIZE]; /**< RPS List including most recent frame if it is reference frame. */
    bool bMVbufferdump;             /**< Set if bitstream buffer contains MV Buffer dump. */
    uint32_t MVBufferDumpSize;      /**< Size of the MV buffer, including NvMediaEncodeMVBufferHeader and NvMediaEncodeMVData for each macroblock, as per the format defined in NvMediaEncodeMVBufferHeader. */
    uint32_t MVBufferDumpStartOffset; /**< Encoded motion vector buffer dump start offset in the bitstream. */
    NvMediaFrameStats FrameStats;   /**< Encoder Profiling stats. */
    uint32_t ulHrdBitrate;          /**< hrdBitrate to be used to calculate RC stats. */
    uint32_t ulVbvBufSize;          /**< vbvBufSize to be used to compute RC stats. */
    NvMediaVideoCodec codec;        /**< Codec Type. */
    NvMediaEncodeCodecExData codecExData; /**< Codec specific extradata. */
} NvMediaEncodeOutputExtradata;

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IEP_OUTPUT_EXTRA_DATA_H */

