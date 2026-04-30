/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__tensormetadata_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: Tensor Metadata Interface
 *
 * This file defines the Tensor metadata structure to be used for Tensor Interop.
 */

#ifndef NVM_TENSORMETADATA_H
#define NVM_TENSORMETADATA_H

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Defines the maximum number of tensor dimensions. */
#define NVMEDIA_TENSOR_MAX_DIMENSIONS                       (8u)

/** \brief NVM_TENSOR_ATTR_DIMENSION_ORDER flags. */
#define NVM_TENSOR_ATTR_DIMENSION_ORDER_NHWC                (0x00000001u)

/** \brief Specifies the NCHW dimension order for 4-D tensors. */
#define NVM_TENSOR_ATTR_DIMENSION_ORDER_NCHW                (0x00000002u)

/** \brief Specifies the NCxHWCx dimension order for 4-D tensors. */
#define NVM_TENSOR_ATTR_DIMENSION_ORDER_NCxHWx              (0x00000003u)

#if (NV_IS_SAFETY == 0)

/** \brief Specifies the unsigned integer tensor data type. */
#define NVM_TENSOR_ATTR_DATA_TYPE_UINT                      (0x00000001u)
#endif

/** \brief Specifies the integer tensor data type. */
#define NVM_TENSOR_ATTR_DATA_TYPE_INT                       (0x00000002u)

/** \brief Specifies the float tensor data type. */
#define NVM_TENSOR_ATTR_DATA_TYPE_FLOAT                     (0x00000003u)

#if (NV_IS_SAFETY == 0)

/** \brief Indicates that each element is 64 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_64                 (64U)

/** \brief Indicates that each element is 32 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_32                 (32U)
#endif /* (NV_IS_SAFETY == 0) */

/** \brief Indicates that each element is 16 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_16                 (16U)

/** \brief Indicates that each element is 8 bits wide. */
#define NVM_TENSOR_ATTR_BITS_PER_ELEMENT_8                  (8U)

#if (NV_IS_SAFETY == 0)

/** \brief Defines the stride index for NHWC element. */
#define NVM_TENSOR_NHWC_E_STRIDE_INDEX         0U

/** \brief Defines the stride index for NHWC channel. */
#define NVM_TENSOR_NHWC_C_STRIDE_INDEX         NVM_TENSOR_NHWC_E_STRIDE_INDEX

/** \brief Defines the stride index for NHWC column along the W dimension. */
#define NVM_TENSOR_NHWC_W_STRIDE_INDEX         1U

/** \brief Defines the stride index for NHWC line along the H dimension. */
#define NVM_TENSOR_NHWC_H_STRIDE_INDEX         2U

/** \brief Defines the stride index for NHWC plane along the N dimension. */
#define NVM_TENSOR_NHWC_N_STRIDE_INDEX         3U

/** \brief Defines the stride index for NCHW element. */
#define NVM_TENSOR_NCHW_E_STRIDE_INDEX         0U

/** \brief Defines the stride index for NCHW column along the W dimension. */
#define NVM_TENSOR_NCHW_W_STRIDE_INDEX         NVM_TENSOR_NCHW_E_STRIDE_INDEX

/** \brief Defines the stride index for NCHW line along the H dimension. */
#define NVM_TENSOR_NCHW_H_STRIDE_INDEX         1U

/** \brief Defines the stride index for NCHW channel. */
#define NVM_TENSOR_NCHW_C_STRIDE_INDEX         2U

/** \brief Defines the stride index for NCHW plane along the N dimension. */
#define NVM_TENSOR_NCHW_N_STRIDE_INDEX         3U

/** \brief Defines the stride index for NCxHWx element. */
#define NVM_TENSOR_NCxHWx_E_STRIDE_INDEX       0U

/** \brief Defines the stride index for NCxHWx channel. */
#define NVM_TENSOR_NCxHWx_X_STRIDE_INDEX       NVM_TENSOR_NCxHWx_E_STRIDE_INDEX

/** \brief Defines the stride index for NCxHWx column along the W dimension. */
#define NVM_TENSOR_NCxHWx_W_STRIDE_INDEX       1U

/** \brief Defines the stride index for NCxHWx line along the H dimension. */
#define NVM_TENSOR_NCxHWx_H_STRIDE_INDEX       2U

/** \brief Defines the stride index for NCxHWx channel along the Cx dimension. */
#define NVM_TENSOR_NCxHWx_Cx_STRIDE_INDEX      3U

/** \brief Defines the stride index for NCxHWx plane along the N dimension. */
#define NVM_TENSOR_NCxHWx_N_STRIDE_INDEX       4U

/** \brief Defines the C dimension index for NHWC. */
#define NVM_TENSOR_NHWC_C_DIMSZ_INDEX          0U

/** \brief Defines the W dimension index for NHWC. */
#define NVM_TENSOR_NHWC_W_DIMSZ_INDEX          1U

/** \brief Defines the H dimension index for NHWC. */
#define NVM_TENSOR_NHWC_H_DIMSZ_INDEX          2U

/** \brief Defines the N dimension index for NHWC. */
#define NVM_TENSOR_NHWC_N_DIMSZ_INDEX          3U

/** \brief Defines the W dimension index for NCHW. */
#define NVM_TENSOR_NCHW_W_DIMSZ_INDEX          0U

/** \brief Defines the H dimension index for NCHW. */
#define NVM_TENSOR_NCHW_H_DIMSZ_INDEX          1U

/** \brief Defines the C dimension index for NCHW. */
#define NVM_TENSOR_NCHW_C_DIMSZ_INDEX          2U

/** \brief Defines the N dimension index for NCHW. */
#define NVM_TENSOR_NCHW_N_DIMSZ_INDEX          3U

/** \brief Defines the x dimension index for NCxHWx. */
#define NVM_TENSOR_NCxHWx_x_DIMSZ_INDEX        0U

/** \brief Defines the W dimension index for NCxHWx. */
#define NVM_TENSOR_NCxHWx_W_DIMSZ_INDEX        1U

/** \brief Defines the H dimension index for NCxHWx. */
#define NVM_TENSOR_NCxHWx_H_DIMSZ_INDEX        2U

/** \brief Defines the Cx dimension index for NCxHWx. */
#define NVM_TENSOR_NCxHWx_Cx_DIMSZ_INDEX       3U

/** \brief Defines the N dimension index for NCxHWx. */
#define NVM_TENSOR_NCxHWx_N_DIMSZ_INDEX        4U

#endif /* (NV_IS_SAFETY == 0) */

/**
 * \brief Holds the tensor metadata.
 */
typedef struct {
    /** Holds the number of valid elements in dimSizes[] and dimstrides[]. */
    uint32_t dimsNum;
    /** Holds the size of each dimension. */
    uint32_t dimSizes[NVMEDIA_TENSOR_MAX_DIMENSIONS];
    /** Holds strides(in bytes) for each dimension present in the tensor. */
    uint32_t dimstrides[NVMEDIA_TENSOR_MAX_DIMENSIONS];
    /** Holds the order of the dimensions. */
    uint32_t dimsOrder;
    /** Holds the bitsPerElement such as NVM_TENSOR_ATTR_BITS_PER_ELEMENT_8/16. */
    uint32_t bitsPerElement;
    /** Holds the tensor datatype, such as NVM_TENSOR_ATTR_DATA_TYPE_INT/FLOAT. */
    uint32_t dataType;
    /** Holds the tensor attribute N. */
    uint32_t attrib4D_N;
    /** Holds the tensor attribute C. */
    uint32_t attrib4D_C;
    /** Holds the tensor attribute H. */
    uint32_t attrib4D_H;
    /** Holds the tensor attribute W. */
    uint32_t attrib4D_W;
    /** Holds the tensor attribute X. */
    uint32_t attrib4D_X;
} NvMediaTensorMetaData;

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVM_TENSORMETADATA_H */

