/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__common__encode_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: Common Types for Encoding
 *
 * This file contains common types and definitions for encode operations.
 */

#ifndef NVMEDIA_COMMON_ENCODE_H
#define NVMEDIA_COMMON_ENCODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "nvmedia_core.h"
#include "nvmedia_common_encode_decode.h"

/** \brief Infinite time-out for NvMediaIEPBitsAvailable. */
#define NVMEDIA_ENCODE_TIMEOUT_INFINITE         0xFFFFFFFFU

/** \brief Infinite GOP length so that keyframes are not inserted automatically. */
#define NVMEDIA_ENCODE_INFINITE_GOPLENGTH       0xFFFFFFFFU

/** \brief Maximum number of Personal Identifiable Information (PII) regions. */
#define NVMEDIA_ENCODE_MAX_PII_REGIONS 32U

/**
 * \brief Specifies the encoder instance ID.
 */
typedef enum {
    NVMEDIA_ENCODER_INSTANCE_0 = 0,
    NVMEDIA_ENCODER_INSTANCE_1,
    NVMEDIA_ENCODER_INSTANCE_AUTO,
} NvMediaEncoderInstanceId;

/**
 * \brief Holds quantization parameters (QP) value for frames.
 */
typedef struct {
    int16_t qpInterP;       /**< QP value for P frames. */
    int16_t qpInterB;       /**< QP value for B frames. */
    int16_t qpIntra;        /**< QP value for Intra frames. */

    int16_t reserved[3];
} NvMediaEncodeQP;

/**
 * \brief Holds Personal Identifiable Information (PII) regions.
 *
 * \note This feature is not supported in the QNX Safety build.
 */
typedef struct {
    NvMediaRect  piiRect;   /**< PII region rectangle. */
} NvMediaEncPIIParams;

/**
 * \brief Rate Control Modes.
 */
typedef enum
{
    NVMEDIA_ENCODE_PARAMS_RC_CBR          = 0,
    NVMEDIA_ENCODE_PARAMS_RC_CONSTQP      = 1,
    NVMEDIA_ENCODE_PARAMS_RC_VBR          = 2,
    NVMEDIA_ENCODE_PARAMS_RC_VBR_MINQP    = 3,
    NVMEDIA_ENCODE_PARAMS_RC_CBR_MINQP    = 4
} NvMediaEncodeParamsRCMode;

/**
 * \brief Holds rate control configuration parameters.
 */
typedef struct {
    /** Holds the rate control mode. */
    NvMediaEncodeParamsRCMode rateControlMode;
    /** Specified number of B frames between two reference frames. */
    uint32_t numBFrames;
    union {
        struct {
            uint32_t averageBitRate;        /**< Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t vbvBufferSize;         /**< Holds the VBV(HRD) buffer size, in bits. */
            uint32_t vbvInitialDelay;       /**< Holds the VBV(HRD) initial delay in bits. */
        } cbr;
        struct {
            /** Holds the initial QP to be used for encoding, these values would be used for all frames in Constant QP mode. */
            NvMediaEncodeQP constQP;
        } const_qp;
        struct {
            uint32_t averageBitRate;        /**< Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t maxBitRate;            /**< Holds the maximum bitrate for the encoded output. */
            uint32_t vbvBufferSize;         /**< Holds the VBV(HRD) buffer size, in bits. */
            uint32_t vbvInitialDelay;       /**< Holds the VBV(HRD) initial delay in bits. */
        } vbr;
        struct {
            uint32_t averageBitRate;        /**< Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t maxBitRate;            /**< Holds the maximum bitrate for the encoded output. */
            uint32_t vbvBufferSize;         /**< Holds the VBV(HRD) buffer size, in bits. */
            uint32_t vbvInitialDelay;       /**< Holds the VBV(HRD) initial delay in bits. */
            NvMediaEncodeQP minQP;          /**< Holds the minimum QP used for rate control. */
        } vbr_minqp;
        struct {
            uint32_t averageBitRate;        /**< Holds the average bitrate (in bits/sec) used for encoding. */
            uint32_t vbvBufferSize;         /**< Holds the VBV(HRD) buffer size, in bits. */
            uint32_t vbvInitialDelay;       /**< Holds the VBV(HRD) initial delay in bits. */
            NvMediaEncodeQP minQP;          /**< Holds the minimum QP used for rate control. */
        } cbr_minqp;
    } params;
    /** Use constant QP at frame level or MB row level. */
    bool bConstFrameQP;
    /** Holds the max QP for encoding session when external picture RC hint is used. */
    int8_t maxSessionQP;
    int8_t reserved[3];
 } NvMediaEncodeRCParams;

/**
 * \brief Blocking type.
 */
typedef enum {
    NVMEDIA_ENCODE_BLOCKING_TYPE_NEVER,
    NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING
} NvMediaBlockingType;

/**
 * \brief Input picture type.
 */
typedef enum {
    NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT      = 0,
    NVMEDIA_ENCODE_PIC_TYPE_P               = 1,
    NVMEDIA_ENCODE_PIC_TYPE_B               = 2,
    NVMEDIA_ENCODE_PIC_TYPE_I               = 3,
    NVMEDIA_ENCODE_PIC_TYPE_IDR             = 4,
    NVMEDIA_ENCODE_PIC_TYPE_P_INTRA_REFRESH = 5
} NvMediaEncodePicType;

/**
 * \brief Defines encoding profiles.
 */
typedef enum {
    NVMEDIA_ENCODE_PROFILE_AUTOSELECT  = 0,

    NVMEDIA_ENCODE_PROFILE_BASELINE    = 66,
    NVMEDIA_ENCODE_PROFILE_MAIN        = 77,
    NVMEDIA_ENCODE_PROFILE_EXTENDED    = 88,
    NVMEDIA_ENCODE_PROFILE_HIGH        = 100,
    NVMEDIA_ENCODE_PROFILE_HIGH10      = 110,
    NVMEDIA_ENCODE_PROFILE_HIGH422     = 122,
    NVMEDIA_ENCODE_PROFILE_HIGH444     = 244,
    NVMEDIA_ENCODE_PROFILE_CAVLC444_INTRA= 44,
} NvMediaEncodeProfile;
 //check eprf
 //professional profile not part of above enum for h265?

/**
 * \brief Defines extended encoding profiles.
 */
typedef enum {
    NVMEDIA_ENCODE_EXT_PROFILE_AUTOSELECT  = 0,

    NVMEDIA_ENCODE_EXT_PROFILE_BASELINE,
    NVMEDIA_ENCODE_EXT_PROFILE_CONSTRAINED_BASELINE,
    NVMEDIA_ENCODE_EXT_PROFILE_MAIN,
    NVMEDIA_ENCODE_EXT_PROFILE_EXTENDED,
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH,
    NVMEDIA_ENCODE_EXT_PROFILE_PROGRESSIVE_HIGH,
    NVMEDIA_ENCODE_EXT_PROFILE_CONSTRAINED_HIGH,
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH10,
    NVMEDIA_ENCODE_EXT_PROFILE_PROGRESSIVE_HIGH10,
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH422,
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH444_PREDICTIVE,
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH10_INTRA,
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH422_INTRA,
    NVMEDIA_ENCODE_EXT_PROFILE_HIGH444_INTRA,
    NVMEDIA_ENCODE_EXT_PROFILE_CAVLC444_INTRA,
} NvMediaEncodeExtProfile;

/**
 * \brief Defines encoding levels for H264 encoder.
 */
typedef enum {
    NVMEDIA_ENCODE_LEVEL_AUTOSELECT         = 0,

    NVMEDIA_ENCODE_LEVEL_H264_1             = 10,
    NVMEDIA_ENCODE_LEVEL_H264_1b            = 9,
    NVMEDIA_ENCODE_LEVEL_H264_11            = 11,
    NVMEDIA_ENCODE_LEVEL_H264_12            = 12,
    NVMEDIA_ENCODE_LEVEL_H264_13            = 13,
    NVMEDIA_ENCODE_LEVEL_H264_2             = 20,
    NVMEDIA_ENCODE_LEVEL_H264_21            = 21,
    NVMEDIA_ENCODE_LEVEL_H264_22            = 22,
    NVMEDIA_ENCODE_LEVEL_H264_3             = 30,
    NVMEDIA_ENCODE_LEVEL_H264_31            = 31,
    NVMEDIA_ENCODE_LEVEL_H264_32            = 32,
    NVMEDIA_ENCODE_LEVEL_H264_4             = 40,
    NVMEDIA_ENCODE_LEVEL_H264_41            = 41,
    NVMEDIA_ENCODE_LEVEL_H264_42            = 42,
    NVMEDIA_ENCODE_LEVEL_H264_5             = 50,
    NVMEDIA_ENCODE_LEVEL_H264_51            = 51,
    NVMEDIA_ENCODE_LEVEL_H264_52            = 52,
    NVMEDIA_ENCODE_LEVEL_H264_END           = 255
} NvMediaEncodeLevel;

/**
 * \brief Defines H265 encoding profiles.
 */
typedef enum {
    NVMEDIA_ENCODE_H265_PROFILE_AUTOSELECT  = 0,

    NVMEDIA_ENCODE_H265_PROFILE_MAIN      = 1,
    NVMEDIA_ENCODE_H265_PROFILE_MAIN10    = 2,
    NVMEDIA_ENCODE_H265_PROFILE_MAIN_STILLPICTURE  = 3,
    NVMEDIA_ENCODE_H265_PROFILES_FORMAT_RANGE_EXTENSIONS = 4,
    NVMEDIA_ENCODE_H265_PROFILES_HIGH_THROUGHPUT = 5,
    NVMEDIA_ENCODE_H265_PROFILES_SCREEN_CONTENT_CODING_EXTENSIONS = 9,
    NVMEDIA_ENCODE_H265_PROFILES_HIGH_THROUGHPUT_SCREEN_CONTENT_CODING_EXTENSIONS = 11,
} NvMediaEncodeH265Profile;

/**
 * \brief Defines encoding levels for H265 encoder.
 */
typedef enum {
    NVMEDIA_ENCODE_LEVEL_H265_AUTOSELECT    = 0,

    NVMEDIA_ENCODE_LEVEL_H265_1             = 30,
    NVMEDIA_ENCODE_LEVEL_H265_2             = 60,
    NVMEDIA_ENCODE_LEVEL_H265_21            = 63,
    NVMEDIA_ENCODE_LEVEL_H265_3             = 90,
    NVMEDIA_ENCODE_LEVEL_H265_31            = 93,
    NVMEDIA_ENCODE_LEVEL_H265_4             = 120,
    NVMEDIA_ENCODE_LEVEL_H265_41            = 123,
    NVMEDIA_ENCODE_LEVEL_H265_5             = 150,
    NVMEDIA_ENCODE_LEVEL_H265_51            = 153,
    NVMEDIA_ENCODE_LEVEL_H265_52            = 156,
    NVMEDIA_ENCODE_LEVEL_H265_6             = 180,
    NVMEDIA_ENCODE_LEVEL_H265_61            = 183,
    NVMEDIA_ENCODE_LEVEL_H265_62            = 186,
    NVMEDIA_ENCODE_LEVEL_H265_END           = 255

} NvMediaEncodeLevelH265;

/**
 * \brief Defines encoding Picture encode flags.
 */
typedef enum {
    NVMEDIA_ENCODE_PIC_FLAG_OUTPUT_SPSPPS      = (1 << 0),
    NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE = (1 << 1),
    NVMEDIA_ENCODE_PIC_FLAG_CONSTRAINED_FRAME  = (1 << 2)
} NvMediaEncodePicFlags;

/**
 * \brief Defines encode preset level settings.
 */
typedef enum {
    NVMEDIA_ENC_PRESET_HQ                      = 0x0,
    NVMEDIA_ENC_PRESET_HP                      = 0x10,
    NVMEDIA_ENC_PRESET_UHP                     = 0x20,
    NVMEDIA_ENC_PRESET_DEFAULT                 = 0x7FFFFFFF
} NvMediaEncPreset;


/**
 * \brief Defines H.264 entropy coding modes.
 */
typedef enum {
    NVMEDIA_ENCODE_H264_ENTROPY_CODING_MODE_CAVLC = 0,
    NVMEDIA_ENCODE_H264_ENTROPY_CODING_MODE_CABAC = 1
} NvMediaEncodeH264EntropyCodingMode;

/**
 * \brief Defines H.264 specific Bdirect modes.
 */
typedef enum {
    NVMEDIA_ENCODE_H264_BDIRECT_MODE_SPATIAL  = 0,
    NVMEDIA_ENCODE_H264_BDIRECT_MODE_DISABLE  = 1,
    NVMEDIA_ENCODE_H264_BDIRECT_MODE_TEMPORAL = 2
} NvMediaEncodeH264BDirectMode;

/**
 * \brief Defines H.264 specific Adaptive Transform modes.
 */
typedef enum {
    NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_AUTOSELECT = 0,
    NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_DISABLE    = 1,
    NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_ENABLE     = 2
} NvMediaEncodeH264AdaptiveTransformMode;

/**
 * \brief Defines motion prediction exclusion flags for H.264.
 */
typedef enum {
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_VERTICAL_PREDICTION             = (1 << 0),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_HORIZONTAL_PREDICTION           = (1 << 1),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_DC_PREDICTION                   = (1 << 2),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_DIAGONAL_DOWN_LEFT_PREDICTION   = (1 << 3),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_DIAGONAL_DOWN_RIGHT_PREDICTION  = (1 << 4),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_VERTICAL_RIGHT_PREDICTION       = (1 << 5),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_HORIZONTAL_DOWN_PREDICTION      = (1 << 6),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_VERTICAL_LEFT_PREDICTION        = (1 << 7),
    NVMEDIA_ENCODE_DISABLE_INTRA_4x4_HORIZONTAL_UP_PREDICTION        = (1 << 8),

    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_VERTICAL_PREDICTION             = (1 << 9),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_HORIZONTAL_PREDICTION           = (1 << 10),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_DC_PREDICTION                   = (1 << 11),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_DIAGONAL_DOWN_LEFT_PREDICTION   = (1 << 12),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_DIAGONAL_DOWN_RIGHT_PREDICTION  = (1 << 13),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_VERTICAL_RIGHT_PREDICTION       = (1 << 14),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_HORIZONTAL_DOWN_PREDICTION      = (1 << 15),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_VERTICAL_LEFT_PREDICTION        = (1 << 16),
    NVMEDIA_ENCODE_DISABLE_INTRA_8x8_HORIZONTAL_UP_PREDICTION        = (1 << 17),

    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_VERTICAL_PREDICTION           = (1 << 18),
    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_HORIZONTAL_PREDICTION         = (1 << 19),
    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_DC_PREDICTION                 = (1 << 20),
    NVMEDIA_ENCODE_DISABLE_INTRA_16x16_PLANE_PREDICTION              = (1 << 21),

    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_VERTICAL_PREDICTION          = (1 << 22),
    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_HORIZONTAL_PREDICTION        = (1 << 23),
    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_DC_PREDICTION                = (1 << 24),
    NVMEDIA_ENCODE_DISABLE_INTRA_CHROMA_PLANE_PREDICTION             = (1 << 25),

    NVMEDIA_ENCODE_DISABLE_INTER_L0_16x16_PREDICTION                 = (1 << 26),
    NVMEDIA_ENCODE_DISABLE_INTER_L0_16x8_PREDICTION                  = (1 << 27),
    NVMEDIA_ENCODE_DISABLE_INTER_L0_8x16_PREDICTION                  = (1 << 28),
    NVMEDIA_ENCODE_DISABLE_INTER_L0_8x8_PREDICTION                   = (1 << 29)
} NvMediaEncodeH264MotionPredictionExclusionFlags;

/**
 * \brief Defines motion search mode control flags for H.264.
 */
typedef enum {
    NVMEDIA_ENCODE_ENABLE_IP_SEARCH_INTRA_4x4                = (1 << 0),
    NVMEDIA_ENCODE_ENABLE_IP_SEARCH_INTRA_8x8                = (1 << 1),
    NVMEDIA_ENCODE_ENABLE_IP_SEARCH_INTRA_16x16              = (1 << 2),
    NVMEDIA_ENCODE_ENABLE_SELF_TEMPORAL_REFINE               = (1 << 3),
    NVMEDIA_ENCODE_ENABLE_SELF_SPATIAL_REFINE                = (1 << 4),
    NVMEDIA_ENCODE_ENABLE_COLOC_REFINE                       = (1 << 5),
    NVMEDIA_ENCODE_ENABLE_EXTERNAL_REFINE                    = (1 << 6),
    NVMEDIA_ENCODE_ENABLE_CONST_MV_REFINE                    = (1 << 7),
    NVMEDIA_ENCODE_MOTION_SEARCH_CONTROL_FLAG_VALID          = (1 << 31)
} NvMediaEncodeH264MotionSearchControlFlags;

/**
 * \brief Specifies the frequency of the writing of Sequence and Picture parameters for H.264.
 */
typedef enum {
    NVMEDIA_ENCODE_SPSPPS_REPEAT_DISABLED          = 0,
    NVMEDIA_ENCODE_SPSPPS_REPEAT_INTRA_FRAMES      = 1,
    NVMEDIA_ENCODE_SPSPPS_REPEAT_IDR_FRAMES        = 2
} NvMediaEncodeH264SPSPPSRepeatMode;

/**
 * \brief Specifies the encoder get attribute type.
 */
typedef enum {
    NvMediaEncAttr_GetSPS = 1,
    NvMediaEncAttr_GetPPS,
    NvMediaEncAttr_GetVPS
} NvMediaEncAttrType;

/**
 * \brief Define H.264 pic_order_cnt_type.
 */
typedef enum {
    NVMEDIA_ENCODE_H264_POC_TYPE_AUTOSELECT     = 0,
    NVMEDIA_ENCODE_H264_POC_TYPE_0              = 1,
    NVMEDIA_ENCODE_H264_POC_TYPE_2              = 2
} NvMediaEncodeH264POCType;

/** \brief Maximum encoded header info size. */
#define MAX_NON_SLICE_DATA_SIZE 2048U

/**
 * \brief This is used to get header info (SPS/PPS/VPS) using GetAttribute call.
 */
typedef struct {
    uint32_t ulNalSize;                     /**< Nal size for header. */
    uint8_t data[MAX_NON_SLICE_DATA_SIZE];  /**< Header data passed on this buffer. */
} NvMediaNalData;

/**
 * \brief Holds H264 video usability information parameters.
 */
typedef struct {
    /** If set to true, it specifies that the aspectRatioIdc is present. */
    bool aspectRatioInfoPresentFlag;
    /** Holds the aspect ratio IDC (as defined in Annex E of the ITU-T Specification). */
    uint8_t aspectRatioIdc;
    /** If aspectRatioIdc is Extended SAR then it indicates horizontal size of the sample aspect ratio (in arbitrary units). */
    uint16_t aspectSARWidth;
    /** If aspectRatioIdc is Extended SAR then it indicates vertical size of the sample aspect ratio (in the same arbitrary units as aspectSARWidth). */
    uint16_t aspectSARHeight;
    /** If set to true, it specifies that the overscanInfo is present. */
    bool overscanInfoPresentFlag;
    /** Holds the overscan info (as defined in Annex E of the ITU-T Specification). */
    bool overscanAppropriateFlag;
    /** If set to true, it specifies that the videoFormat, videoFullRangeFlag and colourDescriptionPresentFlag are present. */
    bool videoSignalTypePresentFlag;
    /** Holds the source video format (as defined in Annex E of the ITU-T Specification). */
    uint8_t videoFormat;
    /** Holds the output range of the luma and chroma samples (as defined in Annex E of the ITU-T Specification). */
    bool videoFullRangeFlag;
    /** If set to true, it specifies that the colourPrimaries, transferCharacteristics and colourMatrix are present. */
    bool colourDescriptionPresentFlag;
    /** Holds color primaries for converting to RGB (as defined in Annex E of the ITU-T Specification). */
    uint8_t colourPrimaries;
    /** Holds the opto-electronic transfer characteristics to use (as defined in Annex E of the ITU-T Specification). */
    uint8_t transferCharacteristics;
    /** Holds the matrix coefficients used in deriving the luma and chroma from the RGB primaries (as defined in Annex E of the ITU-T Specification). */
    uint8_t colourMatrix;
    /** Holds that num_units_in_tick, time_scale and fixed_frame_rate_flag are present in the bitstream. */
    bool timingInfoPresentFlag;
    /** Holds the bitstream restriction info (as defined in Annex E of the ITU-T Specification). */
    bool bitstreamRestrictionFlag;
} NvMediaEncodeConfigH264VUIParams;

/**
 * \brief Holds an external motion vector hint with counts per block type.
 */
typedef struct {
    uint32_t   numCandsPerBlk16x16;     /**< Holds the number of candidates per 16x16 block. */
    uint32_t   numCandsPerBlk16x8;      /**< Holds the number of candidates per 16x8 block. */
    uint32_t   numCandsPerBlk8x16;      /**< Holds the number of candidates per 8x16 block. */
    uint32_t   numCandsPerBlk8x8;       /**< Holds the number of candidates per 8x8 block. */
} NvMediaEncodeExternalMeHintCountsPerBlocktype;

/**
 * \brief Holds an External Motion Vector hint.
 */
typedef struct {
    int32_t mvx        : 12;    /**< Holds the x component of integer pixel MV (relative to current MB) S12.0. */
    int32_t mvy        : 10;    /**< Holds the y component of integer pixel MV (relative to current MB) S10.0. */
    uint32_t refidx     : 5;    /**< Holds the reference index (31=invalid). */
    uint32_t dir        : 1;    /**< Holds the direction of motion estimation. */
    uint32_t partType   : 2;    /**< Holds the block partition type. */
    uint32_t lastofPart : 1;    /**< Set to true for the last MV of (sub) partition. */
    uint32_t lastOfMB   : 1;    /**< Set to true for the last MV of macroblock. */
} NvMediaEncodeExternalMEHint;

/**
 * \brief Defines H264 encoder configuration features.
 */
typedef enum {
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_OUTPUT_AUD               = (1 << 0),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_INTRA_REFRESH            = (1 << 1),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_DYNAMIC_SLICE_MODE       = (1 << 2),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_CONSTRANED_ENCODING      = (1 << 3),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_LOSSLESS_COMPRESSION     = (1 << 4),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_SLICE_LEVEL_OUTPUT       = (1 << 5),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_RTP_MODE_OUTPUT          = (1 << 6),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_EXT_PIC_RC_HINT          = (1 << 7),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_DYNAMIC_RPS              = (1 << 8),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP           = (1 << 9),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_PROFILING                = (1 << 10),
    NVMEDIA_ENCODE_CONFIG_H264_INIT_QP                         = (1 << 11),
    NVMEDIA_ENCODE_CONFIG_H264_QP_MAX                          = (1 << 12),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_FOUR_BYTE_START_CODE     = (1 << 13),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_ULTRA_FAST_ENCODE        = (1 << 14),
    NVMEDIA_ENCODE_CONFIG_H264_ENABLE_MV_BUFFER_DUMP_V2         = (1 << 15),
} NvMediaEncodeH264Features;

/**
 * \brief Holds an H264 encoder configuration.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags. */
    uint32_t features;
    /** Holds the number of pictures in one GOP. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the frequency of the writing of Sequence and Picture parameters. */
    NvMediaEncodeH264SPSPPSRepeatMode repeatSPSPPS;
    /** Holds the IDR interval. */
    uint32_t idrPeriod;
    /** Holds a number that is 1 less than the desired number of slices per frame. */
    uint16_t numSliceCountMinus1;
    /** Holds the deblocking filter mode. */
    uint8_t disableDeblockingFilterIDC;
    /** Holds the Adaptive Transform Mode. */
    NvMediaEncodeH264AdaptiveTransformMode adaptiveTransformMode;
    /** Holds the BDirect mode. */
    NvMediaEncodeH264BDirectMode bdirectMode;
    /** Holds the entropy coding mode. */
    NvMediaEncodeH264EntropyCodingMode entropyCodingMode;
    /** Holds the interval between frames that triggers a new intra refresh cycle. */
    uint32_t intraRefreshPeriod;
    /** Holds the number of frames over which intra refresh happens. */
    uint32_t intraRefreshCnt;
    /** Holds the maximum slice size in bytes for dynamic slice mode. */
    uint32_t maxSliceSizeInBytes;
    /** Holds the number of macroblocks per slice. */
    uint32_t numMacroblocksPerSlice;
    /** Holds the H.264 video usability information pamameters. */
    NvMediaEncodeConfigH264VUIParams *h264VUIParameters;
    /** Holds bitwise OR`ed exclusion flags. */
    uint32_t motionPredictionExclusionFlags;
    /** Holds pic_ordec_cnt_type. */
    NvMediaEncodeH264POCType pocType;
    /** Holds the initial QP parameters. */
    NvMediaEncodeQP initQP;
    /** Holds the maximum QP parameters. */
    NvMediaEncodeQP maxQP;
    /** Enable/disable weighted prediction. */
    uint8_t enableWeightedPrediction;
    /** Holds the encode quality pre-set. */
    NvMediaEncPreset encPreset;
} NvMediaEncodeConfigH264;

/**
 * \brief H.264 specific User SEI message.
 */
typedef struct {
    uint32_t payloadSize;       /**< SEI payload size in bytes. */
    uint32_t payloadType;       /**< SEI payload types and syntax can be found in Annex D of the H.264 Specification. */
    uint8_t *payload;           /**< Pointer to user data. */
} NvMediaEncodeH264SEIPayload;

/**
 * \brief Holds H264-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width. */
    uint16_t encodeWidth;
    uint16_t reserved1;
    /** Holds the encode height. */
    uint16_t encodeHeight;
    uint16_t reserved2;
    /** Set this to true for limited-RGB (16-235) input. */
    bool enableLimitedRGB;
    /** Holds the numerator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateDen;
    /** Holds the encoding profile. */
    uint8_t profile;
    /** Enables extended encoding profiles. */
    bool enableExtProfile;
    /** Holds the extended encoding profile. */
    NvMediaEncodeExtProfile extProfile;
    /** Holds the encoding level. */
    uint8_t level;
    /** Holds the maximum number of reference frames used for encoding. */
    uint8_t maxNumRefFrames;
    /** Set to true to enable external ME hints. */
    bool enableExternalMEHints;
    /** If Client wants to pass external motion vectors in NvMediaEncodePicParamsH264 meExternalHints buffer
     *  it must specify the maximum number of hint candidates per block per direction for the encode session. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Enable support for recon CRC generation. */
    bool enableReconCRC;
    /** If client want to do MVC encoding then this flag need to be set. */
    bool enableMVC;
    /** Enable region of interest encoding. */
    bool enableROIEncode;
    /** Use slice encode to reduce latency in getting encoded buffers. */
    bool enableSliceEncode;
    /** Enables B frames to be used as reference frames. */
    uint8_t useBFramesAsRef;
    uint8_t reserved3[3];
    /** Enable 2 pass RC support. */
    bool enableTwoPassRC;
    /** Enable 2 pass RC with quarter resolution first pass. */
    bool enableSourceHalfScaled;
    /** Number of views used for MVC. */
    uint32_t mvcNumViews             : 4;
    /** Enable external picture rate control. */
    uint32_t enableExternalPictureRC : 1;
    /** Encode all frames as I frames. */
    uint32_t enableAllIFrames : 1;
    /** Enables feature to allocate memory as per optimizations. */
    uint32_t enableMemoryOptimization : 1;
    /** Enables video anonymization of PII regions. */
    uint32_t enableAnonEncode : 1;
    /** Add padding. */
    uint32_t reserved                : 24;
} NvMediaEncodeInitializeParamsH264;

/**
 * \brief H264 specific encoder picture params. Sent on a per frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags. */
    uint32_t encodePicFlags;
    /** Specifies the number of B-frames that follow the current frame. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward if the
     *  NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the encodePicFlags. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of elements allocated in seiPayloadArray array. */
    uint32_t seiPayloadArrayCnt;
    /** Array of SEI payloads which will be inserted for this frame. */
    NvMediaEncodeH264SEIPayload *seiPayloadArray;
    /** Holds the number of hint candidates per block per direction for the current frame. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame. */
    union
    {
        NvMediaEncodeExternalMEHint *meExternalHints;
        uint8_t *meHints;
    };
    /** Holds the numerator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateDen;
    /** Holds the viewId of current picture. */
    uint32_t viewId;
    /** Holds the PII regions information.
     *  \note This feature is not supported in the QNX Safety build. */
    NvMediaEncPIIParams PIIparams[NVMEDIA_ENCODE_MAX_PII_REGIONS];
    /** Holds the number of PII regions for current pic.
     *  \note This feature is not supported in the QNX Safety build. */
    uint32_t numPIIRegions;
} NvMediaEncodePicParamsH264;

/**
 * \brief Holds the H.265 video usability information parameters.
 */
typedef struct {
    /** If set to true, specifies the aspectRatioIdc is present. */
    bool aspectRatioInfoPresentFlag;
    /** Holds the aspect ratio IDC (as defined in Annex E of the ITU-T specification). */
    uint8_t aspectRatioIdc;
    /** If aspectRatioIdc is Extended SAR it indicates horizontal size of the sample aspect ratio (in arbitrary units). */
    uint16_t aspectSARWidth;
    /** If aspectRatioIdc is Extended SAR it indicates vertical size of the sample aspect ratio (in the same arbitrary units as aspectSARWidth). */
    uint16_t aspectSARHeight;
    /** If set to true, it specifies that the overscanInfo is present. */
    bool overscanInfoPresentFlag;
    /** Holds the overscan info (as defined in Annex E of the ITU-T Specification). */
    bool overscanAppropriateFlag;
    /** If set to true, it specifies that the videoFormat, videoFullRangeFlag, and colourDescriptionPresentFlag are present. */
    bool videoSignalTypePresentFlag;
    /** Holds the source video format (as defined in Annex E of the ITU-T Specification). */
    uint8_t videoFormat;
    /** Holds the output range of the luma and chroma samples (as defined in Annex E of the ITU-T Specification). */
    bool videoFullRangeFlag;
    /** If set to true, it specifies that the colourPrimaries, transferCharacteristics, and colourMatrix are present. */
    bool colourDescriptionPresentFlag;
    /** Holds color primaries for converting to RGB (as defined in Annex E of the ITU-T Specification). */
    uint8_t colourPrimaries;
    /** Holds the opto-electronic transfer characteristics to use (as defined in Annex E of the ITU-T Specification). */
    uint8_t transferCharacteristics;
    /** Holds the matrix coefficients used in deriving the luma and chroma from the RGB primaries (as defined in Annex E of the ITU-T Specification). */
    uint8_t matrixCoeffs;
    /** Holds that num_units_in_tick, time_scale and fixed_frame_rate_flag are present in the bitstream (as defined in Annex E of the ITU-T Specification). */
    bool vuiTimingInfoPresentFlag;
    /** Specified the bitstream restriction info (as defined in Annex E of the ITU-T Specification). */
    bool bitstreamRestrictionFlag;
} NvMediaEncodeConfigH265VUIParams;

/**
 * \brief Defines H265 encoder configuration features.
 */
typedef enum {
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_OUTPUT_AUD               = (1 << 0),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_INTRA_REFRESH            = (1 << 1),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_DYNAMIC_SLICE_MODE       = (1 << 2),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_CONSTRANED_ENCODING      = (1 << 3),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_LOSSLESS_COMPRESSION     = (1 << 4),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_SLICE_LEVEL_OUTPUT       = (1 << 5),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_RTP_MODE_OUTPUT          = (1 << 6),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_EXT_PIC_RC_HINT          = (1 << 7),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_DYNAMIC_RPS              = (1 << 8),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP           = (1 << 9),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_PROFILING                = (1 << 10),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_ULTRA_FAST_ENCODE        = (1 << 11),
    NVMEDIA_ENCODE_CONFIG_H265_INIT_QP                         = (1 << 12),
    NVMEDIA_ENCODE_CONFIG_H265_QP_MAX                        = (1 << 13),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_FOUR_BYTE_START_CODE     = (1 << 14),
    NVMEDIA_ENCODE_CONFIG_H265_ENABLE_MV_BUFFER_DUMP_V2         = (1 << 15),
} NvMediaEncodeH265Features;

/**
 * \brief Holds the H265 encoder configuration parameters.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags. */
    uint32_t features;
    /** Holds the number of pictures in one GOP. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the frequency of the writing of Sequence and Picture parameters. */
    NvMediaEncodeH264SPSPPSRepeatMode repeatSPSPPS;
    /** Holds the IDR interval. */
    uint32_t idrPeriod;
    /** Holds a number that is 1 less than the desired number of slices per frame. */
    uint16_t numSliceCountMinus1;
    /** Holds disable the deblocking filter. */
    uint8_t disableDeblockingFilter;
    /** Holds enable weighted prediction. */
    uint8_t enableWeightedPrediction;
    /** Holds the interval between frames that trigger a new intra refresh cycle. */
    uint32_t intraRefreshPeriod;
    /** Holds the number of frames over which intra refresh will happen. */
    uint32_t intraRefreshCnt;
    /** Holds the maximum slice size in bytes for dynamic slice mode. */
    uint32_t maxSliceSizeInBytes;
    /** Number of CTU per slice. */
    uint32_t numCTUsPerSlice;
    /** Holds the H265 video usability info pamameters. */
    NvMediaEncodeConfigH265VUIParams *h265VUIParameters;
    /** Holds Initial QP parameters. */
    NvMediaEncodeQP initQP;
    /** Holds maximum QP parameters. */
    NvMediaEncodeQP maxQP;
    /** Holds the encode quality pre-set. */
    NvMediaEncPreset encPreset;
} NvMediaEncodeConfigH265;

/**
 * \brief Holds H265-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width. */
    uint16_t encodeWidth;
    uint16_t reserved1;
    /** Holds the encode height. */
    uint16_t encodeHeight;
    uint16_t reserved2;
    /** Set this to true for limited-RGB (16-235) input. */
    bool enableLimitedRGB;
    /** Set this to true for slice level encode. */
    bool enableSliceLevelEncode;
    /** Holds the numerator for frame rate used for encoding in frames per second. */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second. */
    uint32_t frameRateDen;
    /** Holds the encoding profile to be set based on NvMediaEncodeH265Profile. */
    uint8_t profile;
    /** Holds the encoding level. */
    uint8_t level;
    /** Holds the level tier information. */
    uint8_t levelTier;
    /** Holds the maximum number of reference frames used for encoding. */
    uint8_t maxNumRefFrames;
    /** Set to true to enable external ME hints. */
    bool enableExternalMEHints;
    /** Specifies maximum hint candidates per block per direction for encoding session. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Enable support for recon CRC generation. */
    bool enableReconCRC;
    /** If client want to do MVC encoding then this flag need to be set. */
    bool enableMVC;
    /** Enable region of interest encoding. */
    bool enableROIEncode;
    /** Use slice encode to reduce latency in getting encoded buffers. */
    bool enableSliceEncode;
    /** Enables B frames to be used as reference frames. */
    uint32_t useBFramesAsRef;
    /** Enable 2 pass RC support. */
    bool enableTwoPassRC;
    /** Enable 2 pass RC with quarter resolution first pass. */
    bool enableSourceHalfScaled;
    /** Number of views used for MV-Hevc. */
    uint32_t mvNumViews              : 4;
    /** Enable external picture rate control. */
    uint32_t enableExternalPictureRC : 1;
    /** Encode all frames as I frames. */
    uint32_t enableAllIFrames : 1;
    /** Add padding. */
    uint32_t reserved                : 26;
    /** Use ampDisable to enable or disable assymetric partition types. */
    bool ampDisable;
} NvMediaEncodeInitializeParamsH265;

/**
 * \brief Holds an H265-specific User SEI message.
 */
typedef struct {
    uint32_t payloadSize;       /**< SEI payload size in bytes. */
    uint32_t payloadType;       /**< SEI payload types and syntax can be found in Annex D of the H265 Specification. */
    uint32_t nalUnitType;       /**< SEI nal_unit_type. */
    uint8_t *payload;           /**< Pointer to user data. */
} NvMediaEncodeH265SEIPayload;

/**
 * \brief Holds H265-specific encoder picture parameters. Sent on a per frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags. */
    uint32_t encodePicFlags;
    /** Specifies the number of B-frames that follow the current frame. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward if the
     *  NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the encodePicFlags. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of elements allocated in seiPayloadArray array. */
    uint32_t seiPayloadArrayCnt;
    /** Array of SEI payloads which will be inserted for this frame. */
    NvMediaEncodeH265SEIPayload *seiPayloadArray;
    /** Holds the number of hint candidates per block per direction for the current frame. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame. */
    NvMediaEncodeExternalMEHint *meExternalHints;
    /** Holds the numerator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateDen;
    /** Holds the viewId of current picture. */
    uint32_t viewId;
} NvMediaEncodePicParamsH265;

/**
 * \brief Defines VP9 encoder configuration features.
 */
typedef enum {
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_LOOP_FILTER_PARAMS               = (1 << 0),
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS              = (1 << 1),
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_TRANSFORM_MODE                   = (1 << 2),
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_HIGH_PRECISION_MV                = (1 << 3),
    NVMEDIA_ENCODE_CONFIG_VP9_DISABLE_ERROR_RESILIENT                 = (1 << 4),
    NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_PROFILING                        = (1 << 5),
    NVMEDIA_ENCODE_CONFIG_VP9_INIT_QP                                 = (1 << 6),
    NVMEDIA_ENCODE_CONFIG_VP9_QP_MAX                                = (1 << 7)
} NvMediaEncodeVP9Features;

/**
 * \brief Holds VP9 encoder configuration parameters.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags. */
    uint32_t features;
    /** Holds the number of pictures in one GOP. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the IDR interval. */
    uint32_t idrPeriod;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_LOOP_FILTER_PARAMS to change
     *  the following parameters. */
    uint32_t filter_type;
    /** Specifies the loop filter strength for each segment. */
    uint32_t filter_level;
    /** Specifies Sharpness level. */
    uint32_t sharpness_level;
    /** Specifies the Loop filter strength adjustments based on frame type (intra, inter). */
    int8_t ref_lf_deltas[4];
    /** Specifies the Loop filter strength adjustments based on mode (zero, new mv). */
    int8_t mode_lf_deltas[2];
    /** Set it to true if MB-level loop filter adjustment is on. */
    bool bmode_ref_lf_delta_enabled;
    /** Set it to true if MB-level loop filter adjustment delta values are updated. */
    bool bmode_ref_lf_delta_update;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_QUANTIZATION_PARAMS to set
     *  the following parameters. */
    uint32_t base_qindex;
    /** Specifies explicit qindex adjustment for y dccoefficient, -15...15. */
    int32_t delta_y_dc_q;
    /** Specifies qindex adjustment for uv accoefficient, -15...15. */
    int32_t delta_uv_dc;
    /** Specifies qindex adjustment for uv dccoefficient, -15...15. */
    int32_t delta_uv_ac;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_TRANSFORM_MODE to set the
     *  following parameter. */
    uint32_t transform_mode;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_ENABLE_HIGH_PRECISION_MV to set the
     *  following parameter. */
    uint32_t high_prec_mv;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_VP9_DISABLE_ERROR_RESILIENT to set the
     *  following parameter. */
    bool error_resilient;

    /** Holds Initial QP parameters. */
    NvMediaEncodeQP initQP;
    /** Holds maximum QP parameters. */
    NvMediaEncodeQP maxQP;
} NvMediaEncodeConfigVP9;

/**
 * \brief Holds VP9-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width. */
    uint32_t encodeWidth;
    /** Holds the encode height. */
    uint32_t encodeHeight;
    /** Holds a flag indicating whether input is limited-RGB (16-235). */
    bool enableLimitedRGB;
    /** Holds the numerator for frame rate used for encoding in frames per second
     *  (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second
     *  (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateDen;
    /** Holds the max reference numbers used for encoding. */
    uint8_t maxNumRefFrames;
    /** Holds a flag indicating whether to enable or disable the external ME hints. */
    bool enableExternalMEHints;
    /** If Client wants to pass external motion vectors in NvMediaEncodePicParamsVP9 meExternalHints
     *  buffer it must specify the maximum number of hint candidates, per block and per direction,
     *  for the encode session. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Holds number of HW entropy cores for encoding. */
    uint8_t numEpCores;
    /** Holds number of log2Rows used in a frame. */
    uint32_t log2TileRows;
    /** Holds number of log2Cols used in a frame. */
    uint32_t log2TileCols;
    /** Skip Chroma Processing. */
    uint32_t vp9SkipChroma;
} NvMediaEncodeInitializeParamsVP9;

/**
 * \brief Holds VP9-specific encoder picture parameters, which are sent on a per frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags. */
    uint32_t encodePicFlags;
    /** Holds the number of B-frames that follow the current frame. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward if the
     *  NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the encodePicFlags. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of hint candidates per block per direction for the current frame. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame. */
    NvMediaEncodeExternalMEHint *meExternalHints;
} NvMediaEncodePicParamsVP9;

/**
 * \brief Defines AV1 encoder configuration features.
 */
typedef enum {
    NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS              = (1 << 1),
    NVMEDIA_ENCODE_CONFIG_AV1_DISABLE_CDF_UPDATE                      = (1 << 2),
    NVMEDIA_ENCODE_CONFIG_AV1_FRAME_END_CDF_UPDATE                    = (1 << 3),
    NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_PROFILING                        = (1 << 5),
    NVMEDIA_ENCODE_CONFIG_AV1_INIT_QP                                 = (1 << 6),
    NVMEDIA_ENCODE_CONFIG_AV1_QP_MAX                                  = (1 << 7)
} NvMediaEncodeAV1Features;

/**
 * \brief Specifies the frequency of the writing of Sequence header for AV1.
 */
typedef enum {
    NVMEDIA_ENCODE_SEQUENCEHDR_REPEAT_DISABLED          = 0,
    NVMEDIA_ENCODE_SEQUENCEHDR_REPEAT_INTRA_FRAMES      = 1,
    NVMEDIA_ENCODE_SEQUENCEHDR_REPEAT_IDR_FRAMES        = 2
} NvMediaEncodeAV1SeqHdrRepeatMode;

/**
 * \brief Holds AV1 encoder configuration parameters.
 */
typedef struct {
    /** Holds bit-wise OR`ed configuration feature flags. */
    uint32_t features;
    /** Holds the number of pictures in one GOP. */
    uint32_t gopLength;
    /** Holds the rate control parameters for the current encoding session. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the IDR interval. */
    uint32_t idrPeriod;

    /** Set the feature flag \ref NVMEDIA_ENCODE_CONFIG_AV1_ENABLE_QUANTIZATION_PARAMS to set the
     *  following parameters.
     *
     *  Specifies quant base index (used only when rc_mode = 0) for each segment 0...255. */
    uint32_t base_qindex;
    /** Specifies explicit qindex adjustment for y dccoefficient, -15...15. */
    int32_t delta_y_dc_q;
    /** Specifies qindex adjustment for uv accoefficient, -15...15. */
    int32_t delta_uv_dc;
    /** Specifies qindex adjustment for uv dccoefficient, -15...15. */
    int32_t delta_uv_ac;

    /** Holds Initial QP parameters. */
    NvMediaEncodeQP initQP;
    /** Holds maximum QP parameters. */
    NvMediaEncodeQP maxQP;
    /** Set to true to disable CDF update. */
    NvMediaBool disableCdfUpdate;
    /** Holds the encode quality pre-set. */
    NvMediaEncPreset encPreset;

    /** Holds the frequency of the writing of Sequence header. */
    NvMediaEncodeAV1SeqHdrRepeatMode repeatSeqHdr;
    /** Reserved Bytes. */
    uint32_t reserved[17];
} NvMediaEncodeConfigAV1;

/**
 * \brief Holds AV1-specific encode initialization parameters.
 */
typedef struct {
    /** Holds the encode width. */
    uint32_t encodeWidth;
    /** Holds the encode height. */
    uint32_t encodeHeight;
    /** Holds a flag indicating whether input is limited-RGB (16-235). */
    bool enableLimitedRGB;
    /** Holds the numerator for frame rate used for encoding in frames per second
     *  (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second
     *  (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateDen;
    /** Holds the encoding profile. */
    uint8_t profile;
    /** Holds the encoding level. */
    uint8_t level;
    /** Holds the max reference numbers used for encoding. */
    uint8_t maxNumRefFrames;
    /** Set to true to enable SSIM RDO. */
    bool enableSsimRdo;
    /** Set to true to enable Multiple tile mode. */
    bool enableTileEncode;
    /** Holds the log2 value of number of tiles used in a row. */
    uint8_t log2NumTilesInRow;
    /** Holds the log2 value of number of tiles used in a column. */
    uint8_t log2NumTilesInCol;
    /** Holds the frame restoration filter type. */
    uint8_t frameRestorationType;
    /** Set to true to enable bi-compound for B-frames (Currently not supported). */
    bool enableBiCompound;
    /** Set to true to enable uni-compound for P/B-frames (only support in P-frames). */
    bool enableUniCompound;
    /** Set to true to enable Internal High Bit depth. */
    bool enableInternalHighBitDepth;
    /** Holds a flag indicating whether to enable or disable the external ME hints. */
    bool enableExternalMEHints;
    /** If Client wants to pass external motion vectors in NvMediaEncodePicParamsAV1 meExternalHints
     *  buffer it must specify the maximum number of hint candidates, per block and per direction,
     *  for the encode session. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype maxMEHintCountsPerBlock[2];
    /** Reserved Bytes. */
    uint32_t reserved[20];
} NvMediaEncodeInitializeParamsAV1;

/**
 * \brief Holds AV1-specific encoder picture parameters, which are sent on a per frame basis.
 */
typedef struct {
    /** Holds input picture type. */
    NvMediaEncodePicType pictureType;
    /** Holds bit-wise OR`ed encode pic flags. */
    uint32_t encodePicFlags;
    /** Holds the number of B-frames that follow the current frame. */
    uint32_t nextBFrames;
    /** Holds the rate control parameters from the current frame onward if the
     *  NVMEDIA_ENCODE_PIC_FLAG_RATECONTROL_CHANGE is set in the encodePicFlags. */
    NvMediaEncodeRCParams rcParams;
    /** Holds the number of hint candidates per block per direction for the current frame. */
    NvMediaEncodeExternalMeHintCountsPerBlocktype meHintCountsPerBlock[2];
    /** Holds the pointer to ME external hints for the current frame. */
    NvMediaEncodeExternalMEHint *meExternalHints;
    /** Holds the numerator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateNum;
    /** Holds the denominator for frame rate used for encoding in frames per second (Frame rate = frameRateNum / frameRateDen). */
    uint32_t frameRateDen;
    /** Reserved Bytes. */
    uint32_t reserved[18];
} NvMediaEncodePicParamsAV1;

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_COMMON_ENCODE_H */

