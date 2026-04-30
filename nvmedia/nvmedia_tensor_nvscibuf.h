/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__tensor__nvscibuf_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NvMedia Interfaces for Tensor for NvSciBuf
 *
 * This file contains APIs for allocation of NvMediaTensor using NvSciBuf.
 */

#ifndef NVMEDIA_TENSOR_NVSCIBUF_H
#define NVMEDIA_TENSOR_NVSCIBUF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvscibuf.h"
#include "nvmedia_core.h"
#include "nvmedia_tensor.h"

/** \brief Major version number. */
#define NVMEDIA_TENSOR_NVSCIBUF_VERSION_MAJOR   (1u)

/** \brief Minor version number. */
#define NVMEDIA_TENSOR_NVSCIBUF_VERSION_MINOR   (7u)

/** \brief Patch version number. */
#define NVMEDIA_TENSOR_NVSCIBUF_VERSION_PATCH   (0u)

/**
 * \brief Initializes the NvMediaTensor NvSciBuf APIs.
 *
 * \return NvMediaStatus, the completion status of the operation:
 * - NVMEDIA_STATUS_OK if the initialization is successful.
 * - NVMEDIA_STATUS_ERROR if there is an error in the initialization.
 * - NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 *
 * \pre This function must be called before calling NvMediaTensorFillNvSciBufAttrs() and NvMediaTensorCreateFromNvSciBuf().
 * \pre This function must be called only once per process
 *
 * Usage considerations
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
 */
NvMediaStatus
NvMediaTensorNvSciBufInit(void);

/**
 * \brief De-initializes the NvMediaTensor NvSciBuf APIs.
 *
 * \pre This function must be called only once per process
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 */
void
NvMediaTensorNvSciBufDeinit(void);

/**
 * \brief Fills the NvSciBuf attributes used to allocate a tensor.
 *
 * This API maps the information in attrs and flags to NvSciBuf attributes and fills them into NvSciBufAttrList referenced by attr_h.
 *
 * After calling this function, the application can call NvSciBufObjAlloc() with attr_h as input and get an NvSciBufObj as output.
 * Then it can call NvMediaTensorCreateFromNvSciBuf() to create an NvMediaTensor from the NvSciBufObj.
 *
 * This function updates the input NvSciBufAttrList with values equivalent to the following public attribute key-values:
 * NvSciBufGeneralAttrKey_PeerHwEngineArray set to
 *  NvSciBufHwEngName: NvSciBufHwEngName_DLA
 *  and
 *  NvSciBufPlatformName: The platform this API is used on
 *
 * \param[in] device A pointer to NvMediaDevice which has been DEPRECATED. Should be set to NULL.
 * \param[in] attrs A valid, non-NULL pointer to NvMediaTensorAttr.
 * \param[in] numAttrs The number of attributes in the array.
 * \param[in] flags Flags for module hint (used in future).
 * \param[in,out] attr_h A valid non-NULL handle to NvSciBufAttrlist to hold the NvSciBuf attributes for the requested NvMediaTensor.
 *
 * \return NvMediaStatus, the completion status of the operation:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_NOT_SUPPORTED indicates that overflow/underflow occured.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL or invalid or out of range.
 * - NVMEDIA_STATUS_OUT_OF_MEMORY if there is a failure to allocate an internal struct.
 * - NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - NVMEDIA_STATUS_ERROR if any other error occurred.
 *
 * \pre This function assumes that attr_h is a valid NvSciBufAttrList created by the application.
 * \pre NvSciBuf library must have been loaded and NvMediaTensor NvSciBuf interfaces must have been initialized.
 * \pre This function must be called before allocation of NvSciBufObj.
 *
 * Usage considerations
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
 */
NvMediaStatus
NvMediaTensorFillNvSciBufAttrs(
    const NvMediaDevice *device,
    const NvMediaTensorAttr *attrs,
    uint32_t numAttrs,
    uint32_t flags,
    NvSciBufAttrList attr_h
);

/**
 * \brief Creates NvMediaTensor from an NvSciBuf handle.
 *
 * When the application is done using NvMediaTensor, it must call NvMediaTensorDestroy() with NvMediaTensor.
 *
 * \param[in] device A pointer to NvMediaDevice which has been DEPRECATED. Should be set to NULL.
 * \param[in] nvSciBufObjInstance An NvSciBufObj for which an NvMediaTensor is to be imported.
 * \param[in,out] nvmTensor A pointer to a location in which a pointer to an imported NvMediaTensor in stored.
 *
 * \return NvMediaStatus, the completion status of the operation:
 * - NVMEDIA_STATUS_OK indicates that NvMediaTensor was successfully created from NvSciBufObj.
 * - NVMEDIA_STATUS_NOT_SUPPORTED indicates that overflow/underflow occured.
 * - NVMEDIA_STATUS_ERROR indicates that another error occurred, such as failure to create NvMediaTensor from NvSciBufObj.
 * - NVMEDIA_STATUS_BAD_PARAMETER indicates that one of the pointer parameters is NULL, or NvSciBufObj was not allocated using the attributes filled by NvMediaTensorGetNvSciBufAttrs().
 * - NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 *
 * \pre This API assumes that nvSciBufObjInstance is a pointer to a valid NvSciBufObj.
 * \pre Application must allocate the nvSciBufObjInstance before this function is called, using the NvSciBufAttrList filled by NvMediaTensorFillNvSciBufAttrs().
 *
 * Usage considerations
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
 */
NvMediaStatus
NvMediaTensorCreateFromNvSciBuf(
    NvMediaDevice *device,
    NvSciBufObj nvSciBufObjInstance,
    NvMediaTensor **nvmTensor
);

/**
 * \brief Returns version information for the NvMediaTensor NvSciBuf API.
 *
 * \param[out] version A valid, non-NULLpointer to a structure in which the function may store version information.
 *
 * \return NvMediaStatus, the completion status of the operation:
 * - NVMEDIA_STATUS_OK if the function call is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if version is invalid.
 * - NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaTensorNvSciBufGetVersion(
    NvMediaVersion *version
);

/*
 * \defgroup history_nvmedia_tensor_nvscibuf History
 * Provides change history for the NvMedia NvSciBuf API.
 *
 * \section history_nvmedia_tensor_nvscibuf Version History
 *
 * <b> Version 1.0 </b> Jan 03, 2019
 * - Initial release
 *
 * <b> Version 1.1 </b> Jan 03, 2019
 * - Adding const in NvMediaTensorFillNvSciBufAttrs
 *   in support of Misra Rule 8.13
 * - Changing param name in NvMediaTensorCreateFromNvSciBuf
 *   in support of Misra rule 8.3
 *
 * <b> Version 1.2 </b> Jan 15, 2020
 * - Fixed the comments for NvMediaTensorCreateFromNvSciBuf
 *
 * <b> Version 1.3 </b> Feb 13, 2020
 * - Updated the comments for NvMediaTensorFillNvSciBufAttrs
 *
 * <b> Version 1.4 </b> Mar 25, 2020
 * - Updated doxygen comments for functions
 *
 * <b> Version 1.5 </b> Apr 29, 2020
 * - Updated doxygen comments for NvMediaTensorCreateFromNvSciBuf
 *   and NvMediaTensorFillNvSciBufAttrs
 *
 * <b> Version 1.6 </b> July 8, 2020
 * - Updated doxygen comments for NvMediaTensorFillNvSciBufAttrs
 *
 * <b> Version 1.7 </b> August 30, 2021
 * - Update doxygen comments for all APIs to have Thread safety information and API Group information
 *
 * <b> Version 1.7.0 </b> May 10, 2022
 * - Added patch version number macro: NVMEDIA_TENSOR_INTERNAL_VERSION_PATCH.
 *
 */
#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_TENSOR_NVSCIBUF_H */
