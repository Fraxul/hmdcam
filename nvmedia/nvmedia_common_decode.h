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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__common__decode_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: Common Types for Image Decode
 *
 * This file contains common types and definitions for image decode operations.
 */

#ifndef NVMEDIA_COMMON_DECODE_H
#define NVMEDIA_COMMON_DECODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "nvmedia_core.h"
#include "nvmedia_common_encode_decode.h"


#define MAX_REFERENCE_FRAMES    16U

/**
 * \brief Specifies the decoder instance ID.
 */
typedef enum  {
    NVMEDIA_DECODER_INSTANCE_0 = 0,
    NVMEDIA_DECODER_INSTANCE_1,
    NVMEDIA_DECODER_INSTANCE_AUTO,
} NvMediaDecoderInstanceId;

/**
 * \brief NAL types found in a bitstream packet.
 */
enum
{
    NVMEDIA_SLH_PRESENT = 0x1,
    NVMEDIA_SPS_PRESENT = 0x2,
    NVMEDIA_PPS_PRESENT = 0x4
};

/** \brief A generic "picture information" pointer type. */
typedef void NvMediaPictureInfo;

/** \brief A generic "reference surface" pointer type. */
typedef void NvMediaRefSurface;

/**
 * \brief To Get the decoding status from HW decoder.
 */
typedef struct{
    /** Pass hw decode clock value to calculate decode time. */
    uint32_t hwClockValue;
    /** Non-zero value indicates error occured during decoding (codec-specific). */
    uint16_t decode_error;
    /** Number of macroblocks decoded. */
    uint32_t decoded_mbs;
    uint32_t concealed_mbs;        // number of macroblocks concealed
    /** 32 bits fields, each bit indicate the decoding progress. */
    uint32_t decoding_status;
    /** HW decoding time. */
    uint32_t hwDecodeTime;
} NvMediaIDEFrameStatus;

/**
 * \brief Macroblock types.
 */
typedef enum {
    NVMEDIA_MBTYPE_B,
    NVMEDIA_MBTYPE_P_FORWARD,
    NVMEDIA_MBTYPE_P_BACKWARD,
    NVMEDIA_MBTYPE_I
} NvMediaMacroBlockType;

/**
 * \brief MB types.
 */
typedef enum {
    NVMEDIA_SKIP,
    NVMEDIA_P,
    NVMEDIA_B,
    NVMEDIA_I,
    NVMEDIA_UNKNOWN_TYPE
} NvMedia_MB_Type_enum;

/**
 * \brief MB part.
 */
typedef enum {
    NVMEDIA_PART_16x16,
    NVMEDIA_PART_16x8,
    NVMEDIA_PART_8x16,
    NVMEDIA_PART_8x8,
    NVMEDIA_UNKNOWN_PART
} NvMedia_MB_Part_enum;

/**
 * \brief Per Macroblock header information.
 */
typedef struct {
    uint16_t mbNum;                 /**< Macroblock number. */
    NvMediaMacroBlockType MBType;   /**< Macroblock type. */
    int16_t mv_for_x;               /**< Forward motion vector in x direction. */
    int16_t mv_for_y;               /**< Forward motion vector in y direction. */
    int16_t mv_bac_x;               /**< Backward motion vector in x direction. */
    int16_t mv_bac_y;               /**< Backward motion vector in y direction. */
    uint8_t qp;                     /**< qp value. */
    NvMedia_MB_Type_enum mb_type;   /**< mb type. */
    NvMedia_MB_Part_enum mb_part;   /**< mb part. */
} NvMediaMotionVectorMB_Metadata;

/**
 * \brief Motion vector array to get the required current frame stats.
 */
typedef struct{
    /** FrameNumber in decoder order. */
    uint32_t frameNumDecodeOrder;
    /** Total number of macroblocks in current frame. */
    uint32_t mbCount;
    /** Flag to indicate whether motion vector dump is present or not. */
    uint32_t  bMVDumpPresent;
    /** Pointer to motion vector array. */
    NvMediaMotionVectorMB_Metadata *mv;
} NvMediaMotionVectorFrameMetaData;

/**
 * \brief Frame stats structure to get get ring entry idx and motion vector dump.
 */
typedef struct{
    /** This index is used to get the required decoded stats of current frame. */
    uint32_t uRingEntryIdx;
    /** Will be used in case of H264/HEVC to convey the nearest POC out of RPS/DPB. */
    uint32_t uErrorRefPOC;
    /** Motion vector dump for current frame. */
    NvMediaMotionVectorFrameMetaData mvfData;
} NvMediaIDEFrameStats;

/**
 * \brief Information about an H.264 reference frame.
 */
typedef struct {
    /** The surface that contains the reference image. Set to NULL for unused entries. */
    NvMediaRefSurface   *surface;
    /** Is this a long term reference (else short term). */
    uint32_t         is_long_term;
    /** Is the top field used as a reference. Set to false for unused entries. */
    uint32_t         top_is_reference;
    /** Is the bottom field used as a reference. Set to false for unused entries. */
    uint32_t         bottom_is_reference;
    /** [0]: top, [1]: bottom. */
    int32_t                 field_order_cnt[2];
    /** Copy of the H.264 bitstream field: frame_num from slice_header for short-term references,
     *  LongTermPicNum from decoding algorithm for long-term references. */
    uint16_t     FrameIdx;

    /** Parser only: Non-existing reference frame flag (corresponding PicIdx should be set to -1). */
    uint16_t     not_existing;
} NvMediaReferenceFrameH264;

/** \brief Maximum user defined sei payload size. */
#define MAX_USER_SEI_PAYLOAD    128U

/**
 * \brief H.264 SEI payload information Used by the parser only.
 */
typedef struct {
    /** Indicate the type of packing arrangement of the frames, as described in Annex D. */
    uint8_t frame_packing_arrangement_type;
    /** Indicate whether each colour component plane of each consituent frame is quincunx sampled or not, as described in Annex D. */
    uint8_t quincunx_sampling_flag;
    /** Indicates the intended interpretation of the constituent frames, as described in Annex D. */
    uint8_t content_interpretation_flag;
    /** Indicates that whether one of the two constituent frames is spatially flipped relative to its intended orientation for display, as described in Annex D. */
    uint8_t spatial_flipping_flag;
    /** Indicates which one of the two constituent frames is flipped, as described in Annex D. */
    uint8_t frame0_flipped_flag;
    /** Indicates whether all pictures in the current coded video sequence are coded as complementary field pairs, as described in Annex D. */
    uint8_t field_views_flag;
    /** Indicate whether current frame is frame 0, as described in Annex D. */
    uint8_t current_frame_is_frame0_flag;
    /** Flag whether stereo is enabled or not. */
    uint32_t bStereoEnabled;
    /** Specify the length of user data unregistered SEI message, as described in Annex D. */
    uint32_t uUserSeiPayloadLength;
    /** Holds user data unregistered SEI message, as described in Annex D. */
    uint8_t UserDefinedSeiPayload[MAX_USER_SEI_PAYLOAD];
} NvMediaSEIPayloadH264;

/**
 * \brief Picture parameter information for an H.264 picture.
 */
typedef struct {
    /** [0]: top, [1]: bottom. */
    int32_t            field_order_cnt[2];
    /** Will the decoded frame be used as a reference later. */
    uint32_t    is_reference;

    uint16_t chroma_format_idc;                     /**< Copy of the H.264 bitstream field. */
    uint16_t frame_num;                             /**< Copy of the H.264 bitstream field. */
    uint8_t  field_pic_flag;                        /**< Copy of the H.264 bitstream field. */
    uint8_t  bottom_field_flag;                     /**< Copy of the H.264 bitstream field. */
    uint8_t  num_ref_frames;                        /**< Copy of the H.264 bitstream field. */
    uint8_t  mb_adaptive_frame_field_flag;          /**< Copy of the H.264 bitstream field. */
    uint8_t  constrained_intra_pred_flag;           /**< Copy of the H.264 bitstream field. */
    uint8_t  weighted_pred_flag;                    /**< Copy of the H.264 bitstream field. */
    uint8_t  weighted_bipred_idc;                   /**< Copy of the H.264 bitstream field. */
    uint8_t  frame_mbs_only_flag;                   /**< Copy of the H.264 bitstream field. */
    uint8_t  transform_8x8_mode_flag;               /**< Copy of the H.264 bitstream field. */
    int8_t           chroma_qp_index_offset;        /**< Copy of the H.264 bitstream field. */
    int8_t           second_chroma_qp_index_offset; /**< Copy of the H.264 bitstream field. */
    int8_t           pic_init_qp_minus26;           /**< Copy of the H.264 bitstream field. */
    uint8_t  num_ref_idx_l0_active_minus1;          /**< Copy of the H.264 bitstream field. */
    uint8_t  num_ref_idx_l1_active_minus1;          /**< Copy of the H.264 bitstream field. */
    uint8_t  log2_max_frame_num_minus4;             /**< Copy of the H.264 bitstream field. */
    uint8_t  pic_order_cnt_type;                    /**< Copy of the H.264 bitstream field. */
    uint8_t  log2_max_pic_order_cnt_lsb_minus4;     /**< Copy of the H.264 bitstream field. */
    uint8_t  delta_pic_order_always_zero_flag;      /**< Copy of the H.264 bitstream field. */
    uint8_t  direct_8x8_inference_flag;             /**< Copy of the H.264 bitstream field. */
    uint8_t  entropy_coding_mode_flag;              /**< Copy of the H.264 bitstream field. */
    uint8_t  pic_order_present_flag;                /**< Copy of the H.264 bitstream field. */
    uint8_t  deblocking_filter_control_present_flag;/**< Copy of the H.264 bitstream field. */
    uint8_t  redundant_pic_cnt_present_flag;        /**< Copy of the H.264 bitstream field. */
    uint8_t  num_slice_groups_minus1;               /**< Copy of the H.264 bitstream field. */
    uint8_t  slice_group_map_type;                  /**< Copy of the H.264 bitstream field. */
    uint32_t   slice_group_change_rate_minus1;      /**< Copy of the H.264 bitstream field. */
    uint8_t *slice_group_map;                       /**< Slice group map. */
    uint8_t fmo_aso_enable;                         /**< Copy of the H.264 bitstream field. */
    uint8_t scaling_matrix_present;                 /**< Copy of the H.264 bitstream field. */

    uint8_t  scaling_lists_4x4[6][16];              /**< Converted to raster order. */
    uint8_t  scaling_lists_8x8[2][64];              /**< Converted to raster order. */

    NvMediaReferenceFrameH264 referenceFrames[16];
    /** Number of slices in this picture. */
    uint32_t nNumSlices;
    /** Disables error-concealment when NULL. */
    uint32_t *pSliceDataOffsets;
    /** 0:FrameType_B 1:FrameType_P 2:FrameType_I */
    uint8_t frameType;

    /** MVC extension. */
    struct {
        uint16_t num_views_minus1;
        uint16_t view_id;
        uint8_t inter_view_flag;
        uint8_t num_inter_view_refs_l0;
        uint8_t num_inter_view_refs_l1;
        uint8_t MVCReserved8Bits;
        uint16_t InterViewRefsL0[16];
        uint16_t InterViewRefsL1[16];
    } mvcext;

    /** Parser only: SEI payload info. */
    NvMediaSEIPayloadH264 seiPayloadInfo;
    uint32_t pic_width_in_mbs_minus1;               /**< Required for OTF. */
    uint32_t pic_height_in_map_units_minus1;        /**< Required for OTF. */
    int32_t last_sps_id;                            /**< Required for OTF. */
    int32_t last_pps_id;                            /**< Required for OTF. */
    /** Copy of the H.264 bitstream field. */
    int32_t qpprime_y_zero_transform_bypass_flag;
} NvMediaPictureInfoH264;

/**
 * \brief Mastering display data for an H.265 picture. Used by the parser only.
 *
 * For the chromaticity arrays: [0] = G, [1] = B, [2] = R.
 */
typedef struct
{
    /** Normalized x chromaticity cordinate. Range 0 to 50000. */
    uint16_t display_primaries_x[3];
    /** Normalized y chromaticity cordinate. Range 0 to 50000. */
    uint16_t display_primaries_y[3];
    /** Normalized x chromaticity cordinate of white point of mastering display. */
    uint16_t white_point_x;
    /** Normalized y chromaticity cordinate of white point of mastering display. */
    uint16_t white_point_y;
    /** Nominal maximum display luminance in units of candelas per square metre. */
    uint16_t max_display_parameter_luminance;
    /** Nominal minimum display luminance in units of 0.0001 candelas per square metre. */
    uint16_t min_display_parameter_luminance;
} NvMediaMasteringDisplayData;

/**
 * \brief Content Light Level info for an H.265 picture. Used by the parser only. Optional parameter.
 */
typedef struct
{
    /** Maximum content light level in units of candelas per square metre. */
    uint16_t max_content_light_level;
    /** Maximum frame average light level in units of candelas per square metre. */
    uint16_t max_pic_average_light_level;
} NvMediaContentLightLevelInfo;



/**
 * \brief Slice level data used with slice level decoding.
 */
typedef struct
{
    /** Number of bytes in bitstream data buffer. */
    uint32_t uBitstreamDataLen;
    /** Ptr to bitstream data for this picture (slice-layer). */
    uint8_t *pBitstreamData;
    /** Number of slices in this SliceData. */
    uint32_t uNumSlices;
    /** If not NULL, nNumSlices entries, contains offset of each slice within the bitstream data buffer. */
    uint32_t *pSliceDataOffsets;
    /** Number of CTB present in this CTB. */
    uint32_t uCTBCount;
    /** CTB number of first CTB in the slice data. */
    uint32_t uFirstCtbAddr;
    /** First slice flag: whether this SliceData contains first slice of frame. */
    uint32_t bFirstSlice;
    /** Last slice flag: whether this SliceData contains last slice of frame. */
    uint32_t bLastSlice;
    /** Error flag if some parsing error detected. */
    uint32_t bErrorFlag;
} NvMediaSliceDecodeData;

/**
 * \brief Holds picture parameter information for an H.265 picture.
 */
typedef struct {
    // sps
    uint32_t pic_width_in_luma_samples;                     /**< Copy of the H.265 bitstream field. */
    uint32_t pic_height_in_luma_samples;                    /**< Copy of the H.265 bitstream field. */
    uint8_t log2_min_luma_coding_block_size_minus3;         /**< Copy of the H.265 bitstream field. */
    uint8_t log2_diff_max_min_luma_coding_block_size;       /**< Copy of the H.265 bitstream field. */
    uint8_t log2_min_transform_block_size_minus2;           /**< Copy of the H.265 bitstream field. */
    uint8_t log2_diff_max_min_transform_block_size;         /**< Copy of the H.265 bitstream field. */
    uint8_t pcm_enabled_flag;                               /**< Copy of the H.265 bitstream field. */
    uint8_t log2_min_pcm_luma_coding_block_size_minus3;     /**< Copy of the H.265 bitstream field. */
    uint8_t log2_diff_max_min_pcm_luma_coding_block_size;   /**< Copy of the H.265 bitstream field. */
    uint8_t bit_depth_luma;                                 /**< Copy of the H.265 bitstream field. */
    uint8_t bit_depth_chroma;                               /**< Copy of the H.265 bitstream field. */
    uint8_t pcm_sample_bit_depth_luma_minus1;               /**< Copy of the H.265 bitstream field. */
    uint8_t pcm_sample_bit_depth_chroma_minus1;             /**< Copy of the H.265 bitstream field. */
    uint8_t pcm_loop_filter_disabled_flag;                  /**< Copy of the H.265 bitstream field. */
    uint8_t strong_intra_smoothing_enabled_flag;            /**< Copy of the H.265 bitstream field. */
    uint8_t max_transform_hierarchy_depth_intra;            /**< Copy of the H.265 bitstream field. */
    uint8_t max_transform_hierarchy_depth_inter;            /**< Copy of the H.265 bitstream field. */
    uint8_t amp_enabled_flag;                               /**< Copy of the H.265 bitstream field. */
    uint8_t separate_colour_plane_flag;                     /**< Copy of the H.265 bitstream field. */
    uint8_t log2_max_pic_order_cnt_lsb_minus4;              /**< Copy of the H.265 bitstream field. */
    uint8_t num_short_term_ref_pic_sets;                    /**< Copy of the H.265 bitstream field. */
    uint8_t long_term_ref_pics_present_flag;                /**< Copy of the H.265 bitstream field. */
    uint8_t num_long_term_ref_pics_sps;                     /**< Copy of the H.265 bitstream field. */
    uint8_t sps_temporal_mvp_enabled_flag;                  /**< Copy of the H.265 bitstream field. */
    uint8_t sample_adaptive_offset_enabled_flag;            /**< Copy of the H.265 bitstream field. */
    uint8_t scaling_list_enable_flag;                       /**< Copy of the H.265 bitstream field. */
    uint8_t chroma_format_idc;                              /**< Copy of the chroma_format_idc. */
    uint8_t reserved1[3];                                   /**< Copy of the H.265 bitstream field. */

    // pps
    uint8_t dependent_slice_segments_enabled_flag;          /**< Copy of the H.265 bitstream field. */
    uint8_t slice_segment_header_extension_present_flag;    /**< Copy of the H.265 bitstream field. */
    uint8_t sign_data_hiding_enabled_flag;                  /**< Copy of the H.265 bitstream field. */
    uint8_t cu_qp_delta_enabled_flag;                       /**< Copy of the H.265 bitstream field. */
    uint8_t diff_cu_qp_delta_depth;                         /**< Copy of the H.265 bitstream field. */
    int8_t init_qp_minus26;                                 /**< Copy of the H.265 bitstream field. */
    int8_t pps_cb_qp_offset;                                /**< Copy of the H.265 bitstream field. */
    int8_t pps_cr_qp_offset;                                /**< Copy of the H.265 bitstream field. */

    uint8_t constrained_intra_pred_flag;                    /**< Copy of the H.265 bitstream field. */
    uint8_t weighted_pred_flag;                             /**< Copy of the H.265 bitstream field. */
    uint8_t weighted_bipred_flag;                           /**< Copy of the H.265 bitstream field. */
    uint8_t transform_skip_enabled_flag;                    /**< Copy of the H.265 bitstream field. */
    uint8_t transquant_bypass_enabled_flag;                 /**< Copy of the H.265 bitstream field. */
    uint8_t entropy_coding_sync_enabled_flag;               /**< Copy of the H.265 bitstream field. */
    uint8_t log2_parallel_merge_level_minus2;               /**< Copy of the H.265 bitstream field. */
    uint8_t num_extra_slice_header_bits;                    /**< Copy of the H.265 bitstream field. */

    uint8_t loop_filter_across_tiles_enabled_flag;          /**< Copy of the H.265 bitstream field. */
    uint8_t loop_filter_across_slices_enabled_flag;         /**< Copy of the H.265 bitstream field. */
    uint8_t output_flag_present_flag;                       /**< Copy of the H.265 bitstream field. */
    uint8_t num_ref_idx_l0_default_active_minus1;           /**< Copy of the H.265 bitstream field. */
    uint8_t num_ref_idx_l1_default_active_minus1;           /**< Copy of the H.265 bitstream field. */
    uint8_t lists_modification_present_flag;                /**< Copy of the H.265 bitstream field. */
    uint8_t cabac_init_present_flag;                        /**< Copy of the H.265 bitstream field. */
    uint8_t pps_slice_chroma_qp_offsets_present_flag;       /**< Copy of the H.265 bitstream field. */

    uint8_t deblocking_filter_control_present_flag;         /**< Copy of the H.265 bitstream field. */
    uint8_t deblocking_filter_override_enabled_flag;        /**< Copy of the H.265 bitstream field. */
    uint8_t pps_deblocking_filter_disabled_flag;            /**< Copy of the H.265 bitstream field. */
    int8_t pps_beta_offset_div2;                            /**< Copy of the H.265 bitstream field. */
    int8_t pps_tc_offset_div2;                              /**< Copy of the H.265 bitstream field. */
    uint8_t tiles_enabled_flag;                             /**< Copy of the H.265 bitstream field. */
    uint8_t uniform_spacing_flag;                           /**< Copy of the H.265 bitstream field. */
    uint8_t num_tile_columns_minus1;                        /**< Copy of the H.265 bitstream field. */
    uint8_t num_tile_rows_minus1;                           /**< Copy of the H.265 bitstream field. */

    uint16_t column_width_minus1[22];                       /**< Copy of the H.265 bitstream field. */
    uint16_t row_height_minus1[20];                         /**< Copy of the H.265 bitstream field. */

    // RefPicSets
    uint32_t iCur;                                          /**< Copy of the H.265 bitstream field. */
    uint32_t IDRPicFlag;                                    /**< Copy of the H.265 bitstream field. */
    uint32_t RAPPicFlag;                                    /**< Copy of the H.265 bitstream field. */
    uint32_t NumDeltaPocsOfRefRpsIdx;                       /**< Copy of the H.265 bitstream field. */
    uint32_t NumPocTotalCurr;                               /**< Copy of the H.265 bitstream field. */
    uint32_t NumPocStCurrBefore;                            /**< Copy of the H.265 bitstream field. */
    uint32_t NumPocStCurrAfter;                             /**< Copy of the H.265 bitstream field. */
    uint32_t NumPocLtCurr;                                  /**< Copy of the H.265 bitstream field. */
    uint32_t NumBitsToSkip;                                 /**< Copy of the H.265 bitstream field. */
    int32_t CurrPicOrderCntVal;                             /**< Copy of the H.265 bitstream field. */
    /** The surface that contains the reference image. */
    NvMediaRefSurface *RefPics[16];
    int32_t PicOrderCntVal[16];
    uint8_t IsLongTerm[16];                                 /**< 1 = long-term reference. */
    int8_t RefPicSetStCurrBefore[8];
    int8_t RefPicSetStCurrAfter[8];
    int8_t RefPicSetLtCurr[8];

    // scaling lists (diag order)
    uint8_t ScalingList4x4[6][16];                          /**< Copy of the H.265 bitstream field. */
    uint8_t ScalingList8x8[6][64];                          /**< Copy of the H.265 bitstream field. */
    uint8_t ScalingList16x16[6][64];                        /**< Copy of the H.265 bitstream field. */
    uint8_t ScalingList32x32[2][64];                        /**< Copy of the H.265 bitstream field. */
    uint8_t ScalingListDCCoeff16x16[6];                     /**< Copy of the H.265 bitstream field. */
    uint8_t ScalingListDCCoeff32x32[2];                     /**< Copy of the H.265 bitstream field. */

    uint32_t NumDeltaPocs[64];                              /**< Copy of the H.265 bitstream field. */

    uint8_t sps_range_extension_present_flag;
    // sps extension HEVC-main 444
    /** Holds the SPS extension for transform_skip_rotation_enabled_flag. */
    uint8_t transformSkipRotationEnableFlag;
    /** Holds the SPS extension for transform_skip_context_enabled_flag. */
    uint8_t transformSkipContextEnableFlag;
    /** Holds the SPS implicit_rdpcm_enabled_flag. */
    uint8_t implicitRdpcmEnableFlag;
    /** Holds the SPS explicit_rdpcm_enabled_flag. */
    uint8_t explicitRdpcmEnableFlag;
    /** Holds the SPS extended_precision_processing_flag; always 0 in current profile. */
    uint8_t extendedPrecisionProcessingFlag;
    /** Holds the SPS intra_smoothing_disabled_flag. */
    uint8_t intraSmoothingDisabledFlag;
    /** Holds the SPS high_precision_offsets_enabled_flag. */
    uint8_t highPrecisionOffsetsEnableFlag;
    /** Holds the SPS fast_rice_adaptation_enabled_flag. */
    uint8_t fastRiceAdaptationEnableFlag;
    /** Holds the SPS cabac_bypass_alignment_enabled_flag; always 0 in current profile. */
    uint8_t cabacBypassAlignmentEnableFlag;
    /** Holds the SPS intraBlockCopyEnableFlag; always 0 not currently used by spec. */
    uint8_t intraBlockCopyEnableFlag;

    uint8_t pps_range_extension_present_flag;
    // pps extension HEVC-main 444
    /** Holds the PPS extension log2_max_transform_skip_block_size_minus2, 0...5. */
    uint8_t log2MaxTransformSkipSize;
    /** Holds the PPS cross_component_prediction_enabled_flag. */
    uint8_t crossComponentPredictionEnableFlag;
    /** Holds the PPS chroma_qp_adjustment_enabled_flag. */
    uint8_t chromaQpAdjustmentEnableFlag;
    /** Holds the PPS diff_cu_chroma_qp_adjustment_depth, 0...3. */
    uint8_t diffCuChromaQpAdjustmentDepth;
    /** Holds the PPS chroma_qp_adjustment_table_size_minus1+1, 1...6. */
    uint8_t chromaQpAdjustmentTableSize;
    /** Holds the PPS log2_sao_offset_scale_luma, max(0,bitdepth-10). */
    uint8_t log2SaoOffsetScaleLuma;
    /** Holds the PPS log2_sao_offset_scale_chroma. */
    uint8_t log2SaoOffsetScaleChroma;
    int8_t cb_qp_adjustment[6];     /**< Holds -[12,+12]. */
    int8_t cr_qp_adjustment[6];     /**< Holds -[12,+12]. */
    uint8_t reserved2;              /**< Ensures alignment to 4 bytes. */
    uint8_t frameType;              /**< 0:FrameType_B 1:FrameType_P 2:FrameType_I */

    /** Parser Only: Flag to indicated mastering display data is present. */
    uint8_t masterin_display_data_present;
    /** Parser Only: Mastering display data if present. */
    NvMediaMasteringDisplayData MasteringDispData;
    /** Flag to indicate slice decode is enabled. */
    uint32_t  bSliceDecEnable;
    /** Slice decode data when Slice decode is enabled. */
    NvMediaSliceDecodeData sliceDecData;

    struct {
        /** Indicates hecv-mv is present in the stream. */
        uint32_t mv_hevc_enable;
        uint32_t nuh_layer_id;                  /**< Copy of the H.265-MV bitstream field. */
        uint32_t default_ref_layers_active_flag;/**< Copy of the H.265-MV bitstream field. */
        uint32_t NumDirectRefLayers;            /**< Copy of the H.265-MV bitstream field. */
        uint32_t max_one_active_ref_layer_flag; /**< Copy of the H.265-MV bitstream field. */
        uint32_t NumActiveRefLayerPics;         /**< Copy of the H.265-MV bitstream field. */
        uint32_t poc_lsb_not_present_flag;      /**< Copy of the H.265-MV bitstream field. */
        uint32_t  NumActiveRefLayerPics0;       /**< Copy of the H.265-MV bitstream field. */
        uint32_t  NumActiveRefLayerPics1;       /**< Copy of the H.265-MV bitstream field. */
        int32_t  RefPicSetInterLayer0[32];      /**< Copy of the H.265-MV bitstream field. */
        int32_t  RefPicSetInterLayer1[32];      /**< Copy of the H.265-MV bitstream field. */
    } mvext;
    /** Parser Only: Flag to indicated content light level data is present. */
    uint32_t content_light_level_info_present;
    /** Parser Only: Content light level info data if present. */
    NvMediaContentLightLevelInfo ContentLightLevelInfo;
} NvMediaPictureInfoH265;

/**
 * \brief Holds picture parameter information for an MPEG 1 or MPEG 2 picture.
 */
typedef struct {
    /** Reference used by B and P frames. Set to NULL when not used. */
    NvMediaRefSurface *forward_reference;
    /** Reference used by B frames. Set to NULL when not used. */
    NvMediaRefSurface *backward_reference;

    uint8_t picture_structure;          /**< Holds a copy of the MPEG bitstream field. */
    uint8_t picture_coding_type;        /**< Holds a copy of the MPEG bitstream field. */
    uint8_t intra_dc_precision;         /**< Holds a copy of the MPEG bitstream field. */
    uint8_t frame_pred_frame_dct;       /**< Holds a copy of the MPEG bitstream field. */
    uint8_t concealment_motion_vectors; /**< Holds a copy of the MPEG bitstream field. */
    uint8_t intra_vlc_format;           /**< Holds a copy of the MPEG bitstream field. */
    uint8_t alternate_scan;             /**< Holds a copy of the MPEG bitstream field. */
    uint8_t q_scale_type;               /**< Holds a copy of the MPEG bitstream field. */
    uint8_t top_field_first;            /**< Holds a copy of the MPEG bitstream field. */
    /** Holds a copy of the MPEG-1 bitstream field. For MPEG-2, set to 0. */
    uint8_t full_pel_forward_vector;
    /** Holds a copy of the MPEG-1 bitstream field. For MPEG-2, set to 0. */
    uint8_t full_pel_backward_vector;
    /** Holds a copy of the MPEG bitstream field. For MPEG-1, fill both horizontal and vertical entries. */
    uint8_t f_code[2][2];
    /** Holds a copy of the MPEG bitstream field, converted to raster order. */
    uint8_t intra_quantizer_matrix[64];
    /** Holds a copy of the MPEG bitstream field, converted to raster order. */
    uint8_t non_intra_quantizer_matrix[64];
    /** Holds the number of slices in this picture. Required for nvdec. */
    uint32_t nNumSlices;
    /** Passing NULL for pSliceDataOffsets disables error-concealment. */
    uint32_t *pSliceDataOffsets;
    /** Indicates whether the MPEG slices span across multiple rows. */
    uint8_t flag_slices_across_multiple_rows;
} NvMediaPictureInfoMPEG1Or2;

/**
 * \brief Holds picture parameter information for an MPEG-4 Part 2 picture.
 */
typedef struct {
    /** Reference used by B and P frames. Set to NULL when not used. */
    NvMediaRefSurface *forward_reference;
    /** Reference used by B frames. Set to NULL when not used. */
    NvMediaRefSurface *backward_reference;

    int32_t            trd[2];                  /**< Holds a copy of the bitstream field. */
    int32_t            trb[2];                  /**< Holds a copy of the bitstream field. */
    uint16_t vop_time_increment_resolution;     /**< Holds a copy of the bitstream field. */
    uint32_t   vop_time_increment_bitcount;     /**< Holds a copy of the bitstream field. */
    uint8_t  vop_coding_type;                   /**< Holds a copy of the bitstream field. */
    uint8_t  vop_fcode_forward;                 /**< Holds a copy of the bitstream field. */
    uint8_t  vop_fcode_backward;                /**< Holds a copy of the bitstream field. */
    uint8_t  resync_marker_disable;             /**< Holds a copy of the bitstream field. */
    uint8_t  interlaced;                        /**< Holds a copy of the bitstream field. */
    uint8_t  quant_type;                        /**< Holds a copy of the bitstream field. */
    uint8_t  quarter_sample;                    /**< Holds a copy of the bitstream field. */
    uint8_t  short_video_header;                /**< Holds a copy of the bitstream field. */
    uint8_t  rounding_control;                  /**< Derived from vop_rounding_type bitstream field. */
    uint8_t  alternate_vertical_scan_flag;      /**< Holds a copy of the bitstream field. */
    uint8_t  top_field_first;                   /**< Holds a copy of the bitstream field. */
    uint8_t  intra_quantizer_matrix[64];        /**< Holds a copy of the bitstream field. */
    uint8_t  non_intra_quantizer_matrix[64];    /**< Holds a copy of the bitstream field. */
    uint8_t  data_partitioned;                  /**< Holds a copy of the bitstream field. */
    uint8_t  reversible_vlc;                    /**< Holds a copy of the bitstream field. */
    /** Number of slices in this picture; entries contain bitstream offsets per slice. */
    uint32_t nNumSlices;
    /** Passing NULL for pSliceDataOffsets disables error-concealment. */
    uint32_t *pSliceDataOffsets;

    uint16_t video_object_layer_width;          /**< Parser only: video object layer width. */
    uint16_t video_object_layer_height;         /**< Parser only: video object layer height. */
    uint32_t divx_flags;                        /**< Parser only: DivX flags. */
    /** DivX GMC concealment flag. Forbid B frame decoding during the GMC sequence. */
    uint32_t bGMCConceal;
} NvMediaPictureInfoMPEG4Part2;

/**
 * \brief Holds picture parameter information for a VC1 picture.
 */
typedef struct {
    /** Reference used by B and P frames. Set to NULL when not used. */
    NvMediaRefSurface *forward_reference;
    /** Reference used by B frames. Set to NULL when not used. */
    NvMediaRefSurface *backward_reference;
    /** Reference used for range mapping. Set to NULL when not used. */
    NvMediaRefSurface *range_mapped;

    /** I=0, P=1, B=3, BI=4 from 7.1.1.4. */
    uint8_t  picture_type;
    /** Progressive=0, Frame-interlace=2, Field-interlace=3; see VC-1 7.1.1.15. */
    uint8_t  frame_coding_mode;
    /** Bottom field flag TopField=0 BottomField=1. */
    uint8_t  bottom_field_flag;


    uint8_t postprocflag;       /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.1.5. */
    uint8_t pulldown;           /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.1.8. */
    uint8_t interlace;          /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.1.9. */
    uint8_t tfcntrflag;         /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.1.10. */
    uint8_t finterpflag;        /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.1.11. */
    uint8_t psf;                /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.1.3. */
    uint8_t dquant;             /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.8. */
    uint8_t panscan_flag;       /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.3. */
    uint8_t refdist_flag;       /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.4. */
    uint8_t quantizer;          /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.11. */
    uint8_t extended_mv;        /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.7. */
    uint8_t extended_dmv;       /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.14. */
    uint8_t overlap;            /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.10. */
    uint8_t vstransform;        /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.9. */
    uint8_t loopfilter;         /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.5. */
    uint8_t fastuvmc;           /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.6. */
    uint8_t range_mapy_flag;    /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.12.15. */
    uint8_t range_mapy;         /**< Holds a copy of the VC-1 bitstream field. */
    uint8_t range_mapuv_flag;   /**< Holds a copy of the VC-1 bitstream field. See VC-1 6.2.16. */
    uint8_t range_mapuv;        /**< Holds a copy of the VC-1 bitstream field. */

    /** Copy of the VC-1 bitstream field. See VC-1 J.1.10. Only used by simple/main profiles. */
    uint8_t multires;
    /** Copy of the VC-1 bitstream field. See VC-1 J.1.16. Only used by simple/main profiles. */
    uint8_t syncmarker;
    /** VC-1 SP/MP range reduction control. See VC-1 J.1.17. Only used by simple/main profiles. */
    uint8_t rangered;
    /** Copy of the VC-1 bitstream field. See VC-1 7.1.13. Only used by simple/main profiles. */
    uint8_t rangeredfrm;
    /** Copy of the VC-1 bitstream field. See VC-1 J.1.17. Only used by simple/main profiles. */
    uint8_t maxbframes;
    /** Number of slices in this picture. nNumSlices entries contain the offset of each slice
     *  within bitstream data buffer. Required for nvdec. */
    uint32_t nNumSlices;
    /** Passing NULL for pSliceDataOffsets disables error-concealment. */
    uint32_t *pSliceDataOffsets;

    uint8_t profile;        /**< Parser only: Profile. */
    uint16_t frameWidth;    /**< Parser only: Actual frame width. */
    uint16_t frameHeight;   /**< Parser only: Actual frame height. */
} NvMediaPictureInfoVC1;

/**
 * \brief Picture parameter information for a VP8 picture.
 */
typedef struct {
    NvMediaRefSurface *LastReference;       /**< Last reference frame. */
    NvMediaRefSurface *GoldenReference;     /**< Golden reference frame. */
    NvMediaRefSurface *AltReference;        /**< Alternate reference frame. */
    uint8_t key_frame;                      /**< Holds a copy of the VP8 bitstream field. */
    uint8_t version;                        /**< Holds a copy of the VP8 bitstream field. */
    uint8_t show_frame;                     /**< Holds a copy of the VP8 bitstream field. */
    /** Copy of the VP8 bitstream field. 0 = clamp needed in decoder, 1 = no clamp needed. */
    uint8_t clamp_type;
    uint8_t segmentation_enabled;           /**< Holds a copy of the VP8 bitstream field. */
    uint8_t update_mb_seg_map;              /**< Holds a copy of the VP8 bitstream field. */
    uint8_t update_mb_seg_data;             /**< Holds a copy of the VP8 bitstream field. */
    /** Copy of the VP8 bitstream field. 0 means delta, 1 means absolute value. */
    uint8_t update_mb_seg_abs;
    uint8_t filter_type;                    /**< Holds a copy of the VP8 bitstream field. */
    uint8_t loop_filter_level;              /**< Holds a copy of the VP8 bitstream field. */
    uint8_t sharpness_level;                /**< Holds a copy of the VP8 bitstream field. */
    uint8_t mode_ref_lf_delta_enabled;      /**< Holds a copy of the VP8 bitstream field. */
    uint8_t mode_ref_lf_delta_update;       /**< Holds a copy of the VP8 bitstream field. */
    uint8_t num_of_partitions;              /**< Holds a copy of the VP8 bitstream field. */
    uint8_t dequant_index;                  /**< Holds a copy of the VP8 bitstream field. */
    int8_t deltaq[5];                       /**< Holds a copy of the VP8 bitstream field. */

    uint8_t golden_ref_frame_sign_bias;     /**< Holds a copy of the VP8 bitstream field. */
    uint8_t alt_ref_frame_sign_bias;        /**< Holds a copy of the VP8 bitstream field. */
    uint8_t refresh_entropy_probs;          /**< Holds a copy of the VP8 bitstream field. */
    uint8_t CbrHdrBedValue;                 /**< Holds a copy of the VP8 bitstream field. */
    uint8_t CbrHdrBedRange;                 /**< Holds a copy of the VP8 bitstream field. */
    uint8_t mb_seg_tree_probs [3];          /**< Holds a copy of the VP8 bitstream field. */

    int8_t seg_feature[2][4];               /**< Holds a copy of the VP8 bitstream field. */
    int8_t ref_lf_deltas[4];                /**< Holds a copy of the VP8 bitstream field. */
    int8_t mode_lf_deltas[4];               /**< Holds a copy of the VP8 bitstream field. */

    uint8_t BitsConsumed;                   /**< Bits consumed for the current bitstream byte. */
    uint8_t AlignByte[3];                   /**< Holds a copy of the VP8 bitstream field. */
    uint32_t hdr_partition_size;            /**< Remaining header parition size. */
    uint32_t hdr_start_offset;              /**< Start of header partition. */
    uint32_t hdr_processed_offset;          /**< Offset to byte which is parsed in cpu. */
    uint32_t coeff_partition_size[8];       /**< Holds a copy of the VP8 bitstream field. */
    uint32_t coeff_partition_start_offset[8];/**< Holds a copy of the VP8 bitstream field. */
    /** Number of slices in this picture. nNumSlices entries contain offset of each slice. */
    uint32_t nNumSlices;
    /** Passing NULL for pSliceDataOffsets disables error-concealment. */
    uint32_t *pSliceDataOffsets;
    /** Number of bytes in VP8 Coeff partition (for OTF case). */
    uint32_t uCoeffPartitionDataLen;
    /** Handle to VP8 Coeff partition (for OTF case). */
    uint32_t uCoeffPartitionBufferHandle;

    uint32_t RetRefreshGoldenFrame;         /**< Parser only: RetRefreshGoldenFrame. */
    uint32_t RetRefreshAltFrame;            /**< Parser only: RetRefreshAltFrame. */
    uint32_t RetRefreshLastFrame;           /**< Parser only: RetRefreshLastFrame. */
} NvMediaPictureInfoVP8;

/**
 * \brief Holds VP9 counters for adaptive entropy contexts.
 */
typedef struct {
    uint32_t inter_mode_counts[7][3][2];        /**< Holds a copy of the VP9 bitstream field. */
    uint32_t sb_ymode_counts[4][10];            /**< Holds a copy of the VP9 bitstream field. */
    uint32_t uv_mode_counts[10][10];            /**< Holds a copy of the VP9 bitstream field. */
    uint32_t partition_counts[16][4];           /**< Holds a copy of the VP9 bitstream field. */
    uint32_t switchable_interp_counts[4][3];    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t intra_inter_count[4][2];           /**< Holds a copy of the VP9 bitstream field. */
    uint32_t comp_inter_count[5][2];            /**< Holds a copy of the VP9 bitstream field. */
    uint32_t single_ref_count[5][2][2];         /**< Holds a copy of the VP9 bitstream field. */
    uint32_t comp_ref_count[5][2];              /**< Holds a copy of the VP9 bitstream field. */
    uint32_t tx32x32_count[2][4];               /**< Holds a copy of the VP9 bitstream field. */
    uint32_t tx16x16_count[2][3];               /**< Holds a copy of the VP9 bitstream field. */
    uint32_t tx8x8_count[2][2];                 /**< Holds a copy of the VP9 bitstream field. */
    uint32_t mbskip_count[3][2];                /**< Holds a copy of the VP9 bitstream field. */
    uint32_t joints[4];                         /**< Holds a copy of the VP9 bitstream field. */
    uint32_t sign[2][2];                        /**< Holds a copy of the VP9 bitstream field. */
    uint32_t classes[2][11];                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t class0[2][2];                      /**< Holds a copy of the VP9 bitstream field. */
    uint32_t bits[2][10][2];                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t class0_fp[2][2][4];                /**< Holds a copy of the VP9 bitstream field. */
    uint32_t fp[2][4];                          /**< Holds a copy of the VP9 bitstream field. */
    uint32_t class0_hp[2][2];                   /**< Holds a copy of the VP9 bitstream field. */
    uint32_t hp[2][2];                          /**< Holds a copy of the VP9 bitstream field. */
    uint32_t countCoeffs[2][2][6][6][4];        /**< Holds a copy of the VP9 bitstream field. */
    uint32_t countCoeffs8x8[2][2][6][6][4];     /**< Holds a copy of the VP9 bitstream field. */
    uint32_t countCoeffs16x16[2][2][6][6][4];   /**< Holds a copy of the VP9 bitstream field. */
    uint32_t countCoeffs32x32[2][2][6][6][4];   /**< Holds a copy of the VP9 bitstream field. */
    uint32_t countEobs[4][2][2][6][6];          /**< Holds a copy of the VP9 bitstream field. */
} NvMediaVP9BackwardUpdates;

/**
 * \brief Holds VP9 entropy contexts.
 *
 * Formatted for 256-bit memory: probs 0 to 7 for all tables followed by probs 8 to N for all tables.
 */
typedef struct {

    uint8_t kf_bmode_prob[10][10][8];           /**< Holds a copy of the VP9 bitstream field. */
    uint8_t kf_bmode_probB[10][10][1];          /**< Holds a copy of the VP9 bitstream field. */
    uint8_t ref_pred_probs[3];                  /**< Holds a copy of the VP9 bitstream field. */
    uint8_t mb_segment_tree_probs[7];           /**< Holds a copy of the VP9 bitstream field. */
    uint8_t segment_pred_probs[3];              /**< Holds a copy of the VP9 bitstream field. */
    uint8_t ref_scores[4];                      /**< Holds a copy of the VP9 bitstream field. */
    uint8_t prob_comppred[2];                   /**< Holds a copy of the VP9 bitstream field. */
    uint8_t pad1[9];                            /**< Holds a copy of the VP9 bitstream field. */

    uint8_t kf_uv_mode_prob[10][8];             /**< Holds a copy of the VP9 bitstream field. */
    uint8_t kf_uv_mode_probB[10][1];            /**< Holds a copy of the VP9 bitstream field. */
    uint8_t pad2[6];                            /**< Holds a copy of the VP9 bitstream field. */

    uint8_t inter_mode_prob[7][4];              /**< Holds a copy of the VP9 bitstream field. */
    uint8_t intra_inter_prob[4];                /**< Holds a copy of the VP9 bitstream field. */

    uint8_t uv_mode_prob[10][8];                /**< Holds a copy of the VP9 bitstream field. */
    uint8_t tx8x8_prob[2][1];                   /**< Holds a copy of the VP9 bitstream field. */
    uint8_t tx16x16_prob[2][2];                 /**< Holds a copy of the VP9 bitstream field. */
    uint8_t tx32x32_prob[2][3];                 /**< Holds a copy of the VP9 bitstream field. */
    uint8_t sb_ymode_probB[4][1];               /**< Holds a copy of the VP9 bitstream field. */
    uint8_t sb_ymode_prob[4][8];                /**< Holds a copy of the VP9 bitstream field. */

    uint8_t partition_prob[2][16][4];           /**< Holds a copy of the VP9 bitstream field. */

    uint8_t uv_mode_probB[10][1];               /**< Holds a copy of the VP9 bitstream field. */
    uint8_t switchable_interp_prob[4][2];       /**< Holds a copy of the VP9 bitstream field. */
    uint8_t comp_inter_prob[5];                 /**< Holds a copy of the VP9 bitstream field. */
    uint8_t mbskip_probs[3];                    /**< Holds a copy of the VP9 bitstream field. */
    uint8_t pad3[1];                            /**< Holds a copy of the VP9 bitstream field. */

    uint8_t joints[3];                          /**< Holds a copy of the VP9 bitstream field. */
    uint8_t sign[2];                            /**< Holds a copy of the VP9 bitstream field. */
    uint8_t class0[2][1];                       /**< Holds a copy of the VP9 bitstream field. */
    uint8_t fp[2][3];                           /**< Holds a copy of the VP9 bitstream field. */
    uint8_t class0_hp[2];                       /**< Holds a copy of the VP9 bitstream field. */
    uint8_t hp[2];                              /**< Holds a copy of the VP9 bitstream field. */
    uint8_t classes[2][10];                     /**< Holds a copy of the VP9 bitstream field. */
    uint8_t class0_fp[2][2][3];                 /**< Holds a copy of the VP9 bitstream field. */
    uint8_t bits[2][10];                        /**< Holds a copy of the VP9 bitstream field. */

    uint8_t single_ref_prob[5][2];              /**< Holds a copy of the VP9 bitstream field. */
    uint8_t comp_ref_prob[5];                   /**< Holds a copy of the VP9 bitstream field. */
    uint8_t pad4[17];                           /**< Holds a copy of the VP9 bitstream field. */

    uint8_t probCoeffs[2][2][6][6][4];          /**< Holds a copy of the VP9 bitstream field. */
    uint8_t probCoeffs8x8[2][2][6][6][4];       /**< Holds a copy of the VP9 bitstream field. */
    uint8_t probCoeffs16x16[2][2][6][6][4];     /**< Holds a copy of the VP9 bitstream field. */
    uint8_t probCoeffs32x32[2][2][6][6][4];     /**< Holds a copy of the VP9 bitstream field. */
} NvmediaVP9EntropyProbs;

/**
 * \brief Holds picture parameter information for a VP9 picture.
 */
typedef struct {
    /** Holds a pointer to the last reference frame. */
    NvMediaRefSurface *LastReference;
    /** Holds a pointer to the golden reference frame. */
    NvMediaRefSurface *GoldenReference;
    /** Holds a pointer to the alternate reference frame. */
    NvMediaRefSurface *AltReference;
    uint32_t    width;                          /**< Holds a copy of the VP9 bitstream field. */
    uint32_t    height;                         /**< Holds a copy of the VP9 bitstream field. */
    uint32_t    ref0_width;                     /**< Holds a copy of the VP9 bitstream field. */
    uint32_t    ref0_height;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t    ref1_width;                     /**< Holds a copy of the VP9 bitstream field. */
    uint32_t    ref1_height;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t    ref2_width;                     /**< Holds a copy of the VP9 bitstream field. */
    uint32_t    ref2_height;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t keyFrame;                          /**< Holds a copy of the VP9 bitstream field. */
    uint32_t bit_depth;                         /**< Holds the depth per pixel. */
    uint32_t prevIsKeyFrame;                    /**< If previous frame is key frame. */
    uint32_t PrevShowFrame;                     /**< Previous frame is show frame. */
    uint32_t resolutionChange;                  /**< Resolution change. */
    uint32_t errorResilient;                    /**< Error Resilient. */
    uint32_t intraOnly;                         /**< Intra only. */
    uint8_t frameContextIdx;                    /**< Holds a copy of the VP9 bitstream field. */
    uint8_t refFrameSignBias[4];                /**< Holds a copy of the VP9 bitstream field. */
    uint32_t loopFilterLevel;                   /**< Holds a copy of the VP9 bitstream field. */
    uint32_t loopFilterSharpness;               /**< Holds a copy of the VP9 bitstream field. */
    int32_t qpYAc;                              /**< Holds a copy of the VP9 bitstream field. */
    int32_t qpYDc;                              /**< Holds a copy of the VP9 bitstream field. */
    int32_t qpChAc;                             /**< Holds a copy of the VP9 bitstream field. */
    int32_t qpChDc;                             /**< Holds a copy of the VP9 bitstream field. */
    uint32_t lossless;                          /**< Holds a copy of the VP9 bitstream field. */
    uint32_t transform_mode;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t allow_high_precision_mv;           /**< Holds a copy of the VP9 bitstream field. */
    uint32_t allow_comp_inter_inter;            /**< Holds a copy of the VP9 bitstream field. */
    uint32_t mcomp_filter_type;                 /**< Holds a copy of the VP9 bitstream field. */
    uint32_t comp_pred_mode;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t comp_fixed_ref;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t comp_var_ref[2];                   /**< Holds a copy of the VP9 bitstream field. */
    uint32_t log2_tile_columns;                 /**< Holds a copy of the VP9 bitstream field. */
    uint32_t log2_tile_rows;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t segmentEnabled;                    /**< Holds a copy of the VP9 bitstream field. */
    uint32_t segmentMapUpdate;                  /**< Holds a copy of the VP9 bitstream field. */
    int32_t segmentMapTemporalUpdate;           /**< Holds a copy of the VP9 bitstream field. */
    uint32_t segmentFeatureMode;                /**< Holds a copy of the VP9 bitstream field. */
    uint8_t segmentFeatureEnable[8][4];         /**< Holds a copy of the VP9 bitstream field. */
    short segmentFeatureData[8][4];             /**< Holds a copy of the VP9 bitstream field. */
    uint32_t modeRefLfEnabled;                  /**< Holds a copy of the VP9 bitstream field. */
    uint32_t mbRefLfDelta[4];                   /**< Holds a copy of the VP9 bitstream field. */
    uint32_t mbModeLfDelta[2];                  /**< Holds a copy of the VP9 bitstream field. */
    uint32_t offsetToDctParts;                  /**< Holds a copy of the VP9 bitstream field. */
    uint32_t frameTagSize;                      /**< Holds a copy of the VP9 bitstream field. */
    NvmediaVP9EntropyProbs entropy;             /**< Holds a copy of the VP9 bitstream field. */

    /** Parser only: Backward update counts. */
    NvMediaVP9BackwardUpdates backwardUpdateCounts;
} NvMediaPictureInfoVP9;
#define AV1_MAX_TILES  256                         // defined in c8b0_drv.h

typedef struct _NvMediaPicEntry_AV1_Short {
    uint8_t Index;
} NvMediaPicEntry_AV1_Short;

typedef struct _NvMediaPicEntry_AV1 {
    uint8_t Index;
    uint32_t width;
    uint32_t height;
    // Global motion parameters
    //struct {
    uint8_t invalid ;
    uint8_t wmtype ;
    int32_t wmmat[6];
    //} global_motion;

} NvMediaPicEntry_AV1;

typedef struct _NvMediaTile_AV1 {
    uint32_t   dataOffset;
    uint32_t   dataSize;
    uint16_t   row;
    uint16_t   column;
    NvMediaPicEntry_AV1_Short anchor_frame;
    uint8_t Reserved24Bits[3];
} NvMediaTile_AV1;

/**
 * \brief Holds picture parameter information for AV1 picture.
 */
typedef struct _NvMediaPictureInfo_AV1 {
    uint32_t width;
    uint32_t height;
    uint32_t superres_width;                // Not defined in DXVA
    uint32_t max_width;
    uint32_t max_height;

    NvMediaPicEntry_AV1_Short CurrPic;
    NvMediaPicEntry_AV1_Short FgsPic;     // Not defined in DXVA

    uint8_t superres_denom;
    uint8_t BitDepth;
    uint8_t profile;

    // Tiles:
    struct {
        uint8_t cols;
        uint8_t rows;
        uint16_t widths[AV1_MAX_TILES];            // AV1_MAX_TILES = 256 as defined in c8b0_drv.h
        uint16_t heights[AV1_MAX_TILES];
        uint16_t context_update_id;
        uint32_t tile_info[AV1_MAX_TILES *2];
    } tiles;

    // Coding Tools
     uint8_t use_128x128_superblock;
     uint8_t intra_edge_filter;
     uint8_t interintra_compound;
     uint8_t masked_compound;
     uint8_t warped_motion;
     uint8_t dual_filter;
     uint8_t jnt_comp;
     uint8_t screen_content_tools;
     uint8_t integer_mv;
     uint8_t cdef_enable;
     uint8_t restoration;
     uint8_t film_grain_enable;
     uint8_t intrabc;
     uint8_t high_precision_mv;
     uint8_t switchable_motion_mode;
     uint8_t filter_intra;
     uint8_t disable_frame_end_update_cdf;
     uint8_t disable_cdf_update;
     uint8_t reference_mode;
     uint8_t skip_mode;
     uint8_t reduced_tx_set;
     uint8_t superres;
     uint8_t tx_mode;
     uint8_t use_ref_frame_mvs;
     uint8_t reference_frame_update;

    // Format & Picture Info flags
    uint8_t frame_type;
    uint8_t show_frame;
    uint8_t showable_frame;
    uint8_t subsampling_x;
    uint8_t subsampling_y;
    uint8_t mono_chrome;
    uint8_t chroma_format_idc;
    NvMediaPicEntry_AV1_Short primary_ref_frame;
    uint8_t order_hint;
    uint8_t order_hint_bits_minus_1;

    // References
    NvMediaPicEntry_AV1 frame_refs[7];
    NvMediaRefSurface *RefPics[8];
    int8_t ref_frame_map_index[8];

    uint32_t enable_order_hint;
    uint32_t skip_ref0 : 4;
    uint32_t skip_ref1 : 4;
    uint32_t reserved : 24;
    int32_t show_existing_frame_index;

    // Loop filter parameters
    struct {
        uint8_t filter_level[2];
        uint8_t filter_level_u;
        uint8_t filter_level_v;
        uint8_t sharpness_level;

        uint8_t mode_ref_delta_enabled;
        uint8_t mode_ref_delta_update;
        uint8_t delta_lf_present;
        uint8_t delta_lf_multi;

        int8_t ref_deltas[8];
        int8_t mode_deltas[2];
        uint8_t delta_lf_res;
        // loop restoration
        uint16_t restoration_unit_size[3];
        uint8_t frame_restoration_type[3];
    } loop_filter;

    // Quantization
    struct {
        uint8_t delta_q_present;
        uint8_t delta_q_res;
        uint8_t base_qindex;
        int8_t y_dc_delta_q;
        int8_t u_dc_delta_q;
        int8_t v_dc_delta_q;
        int8_t u_ac_delta_q;
        int8_t v_ac_delta_q;
        uint8_t qm_y;
        uint8_t qm_u;
        uint8_t qm_v;
    } quantization;

    // Cdef parameters
    struct {
                uint8_t damping;
                uint8_t bits;

        union {
            struct {
                uint8_t primary : 6;
                uint8_t secondary : 2;
            };
            uint8_t combined;
        } y_strengths[8];

        union {
            struct {
                uint8_t primary : 6;
                uint8_t secondary : 2;
            };
            uint8_t combined;
        }  uv_strengths[8];
    } cdef;

    uint8_t interp_filter;

    // Segmentation
    struct {
        uint8_t enabled;
        uint8_t update_map;
        uint8_t update_data;
        uint8_t temporal_update;
        uint8_t segid_preskip;
        uint8_t feature_mask[8];
        int16_t feature_data[8][8];
    } segmentation;

    // film grain
    NvMediaRefSurface *fgsPic;
    struct {
        uint16_t apply_grain;
        uint16_t scaling_shift_minus8;
        uint16_t chroma_scaling_from_luma;
        uint16_t ar_coeff_lag;
        uint16_t ar_coeff_shift_minus6;
        uint16_t grain_scale_shift;
        uint16_t overlap_flag;
        uint16_t clip_to_restricted_range;
        uint16_t grain_seed;
        uint8_t scaling_points_y[14][2];
        uint8_t num_y_points;
        uint8_t scaling_points_cb[10][2];
        uint8_t num_cb_points;
        uint8_t scaling_points_cr[10][2];
        uint8_t num_cr_points;
        int16_t ar_coeffs_y[24];
        int16_t ar_coeffs_cb[25];
        int16_t ar_coeffs_cr[25];
        uint8_t cb_mult;
        uint8_t cb_luma_mult;
        int16_t cb_offset;
        uint8_t cr_mult;
        uint8_t cr_luma_mult;
        int16_t cr_offset;
    } film_grain;
    uint8_t  refresh_frame_flags;
} NvMediaPictureInfoAV1;
#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_COMMON_DECODE_H */

