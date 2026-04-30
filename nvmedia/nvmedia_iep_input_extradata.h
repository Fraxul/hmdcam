/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

// Based on the NvMedia API from DriveOS 6.0.10
//
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__iep__input__extradata_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: NvMedia Image Encode Processing Input ExtraData
 */

#ifndef NVMEDIA_IEP_INPUT_EXTRA_DATA_H
#define NVMEDIA_IEP_INPUT_EXTRA_DATA_H

#include <stdbool.h>
#include "nvmedia_core.h"
#include "nvmedia_iep.h"

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Maximum number of reference pictures including current frame. */
#define NVMEDIA_ENCODE_MAX_RPS_SIZE 17U

/** \brief Maximum number of region of interests. */
#define NVMEDIA_MAX_ROI_REGIONS 8U

typedef struct {
    NvMediaRect  roiRect;       /**< Region of interest rectangle. */
    int32_t   lQPdelta;         /**< QP delta for Region. */
} NvMediaEncROIParams;

/**
 * \brief Defines video frame flags. Each bit of bEncodeParamsFlag indicate parameter for specific featue.
 */
typedef enum {
    NvMediaVideoEncFrame_PPEMetadata         = (1<<6),  /**< enable preprocessing enhancements buffer */
    NvMediaVideoEncFrame_QPDeltaBuffer       = (1<<7),  /**< enable QP Delta Buffer */
    NvMediaVideoEncFrame_ROIParams           = (1<<4),  /**< passing region of interest parameters */
    /* Add other flags using bitfields */
} NvMediaVideoEncEncFrameFlags;

/**
 * \brief Holds an Video encoder input extradata configuration.
 */
typedef struct {
    /** Size of this extradata structure. This needs to be filled correctly by client This size is used as sanity check before reading input extradata from this buffer */
    uint32_t ulExtraDataSize;
    /** bit fields defined in NvMediaVideoEncEncFrameFlags to indicate valid frame parameters */
    uint32_t EncodeParamsFlag;
    /** Preprocessing enhancements metadata. */
    uint8_t *PPEMetadata;
    /** Parameter to program QP Delta Buffer. */
    signed char *QPDeltaBuffer;
    /** Parameter to program QP Delta Buffer Size. */
    uint32_t QPDeltaBufferSize;
    /** Region of interest parameters. They are valid when EncodeParamsFlag set with NvMediaVideoEncFrame_ROIParams Num Of ROI Regions */
    uint32_t ulNumROIRegions;
    /** ROI parameters for all regions. */
    NvMediaEncROIParams ROIParams[NVMEDIA_MAX_ROI_REGIONS];
    /** Parameter to enable addition of ROI information to SEI NAL. */
    uint8_t bSEIforROIEnable;
} NvMediaEncodeInputExtradata;
/**
 * \brief Defines the resolution change parameters. This is used if encoding resolution change on the fly.
 */
typedef struct
{
    uint16_t ulDRCWidth;        /**< Holds the encode Width. It will be used for DRC */
    uint16_t ulDRCHeight;       /**< Holds the encode Height. It will be used for DRC */
} NvMediaEncodeDRCConfig;

/**
 * \brief Specifies the Video encoder set attribute type.
 */
typedef enum {
    NvMediaEncSetAttr_DRCParams,        /**< This is used to pass dynamic resolution change specific information */
    NvMediaEncSetAttr_HWConfigFiles,    /**< This attribute is used to pass HW config files...for current encoding session. */
} NvMediaEncSetAttrType;

/**
 * \brief Defines the structure for holding the encoding attributes for HW validation.
 */
typedef struct
{
    char* ExtHintsFileBaseName;         /**< Ext Hints config file name for HW validation. */
} NvMediaEncodeCfgFiles;


/**
 * \brief Set the encoder extra data for current frame for encoding. This should be called before before every NvMediaIEPFeedFrame so that parameters can be applied for current frame.
 * \param[in] encoder The encoder to use.
 * \param[in] extradata encode input extra data for current frame. Input extradata ia defined as NvMediaEncodeInputExtradata
 * \return The completion status of the operation. Possible values are: NVMEDIA_STATUS_OK [or] NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPSetInputExtraData(
    const NvMediaIEP *encoder,
    const void *extradata
);

/**
 * \brief Set the encoder attribute for current encoding session. This can be called before passing the first frame for encoding.
 * \param[in] encoder The encoder to use.
 * \param[in] attrType attribute type as defined in NvMediaEncSetAttrType
 * \param[in] attrSize size of the data structure associated with attribute Possible values are: NvMediaEncSetAttr_DRCParams if resolution of encoding changes.
 * \param[in] AttributeData pointer to data structure associated with attribute
 * \return The completion status of the operation. Possible values are: NVMEDIA_STATUS_OK, NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPSetAttribute(
    const NvMediaIEP *encoder,
    NvMediaEncSetAttrType attrType,
    uint32_t attrSize,
    const void *AttributeData
);

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IEP_INPUT_EXTRA_DATA_H */
