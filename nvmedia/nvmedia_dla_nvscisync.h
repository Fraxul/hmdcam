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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__dla__nvscisync_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief This file contains the NvMediaDla and NvSciSync related APIs.
 */

#ifndef NVMEDIA_DLA_NVSCISYNC_H
#define NVMEDIA_DLA_NVSCISYNC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvscisync.h"
#include "nvmedia_dla.h"

/** \brief Major version number. */
#define NVMEDIA_DLA_NVSCISYNC_VERSION_MAJOR   1

/** \brief Minor version number. */
#define NVMEDIA_DLA_NVSCISYNC_VERSION_MINOR   6

/** \brief Patch version number. */
#define NVMEDIA_DLA_NVSCISYNC_VERSION_PATCH   0

/** \brief NvMediaDlaInsertPreNvSciSyncFence API can be called at most
 * NVMEDIA_DLA_MAX_PRENVSCISYNCFENCES times before each Dla submit call. */
#define NVMEDIA_DLA_MAX_PRENVSCISYNCFENCES  (8U)

/**
 * \brief Returns the version information for the NvMedia DLA NvSciSync library.
 *
 * \param[in,out] version A pointer to an NvMediaVersion structure filled by the DLA NvSciSync library.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if version is NULL.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 *
 * \pre None
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
NvMediaDlaNvSciSyncGetVersion(
    NvMediaVersion *version
);

/**
 * \brief Fills the NvMediaDla specific NvSciSync attributes.
 *
 * \param[in] dla        An NvMedia DLA device handle.
 * \param[in,out] attrlist   A pointer to an NvSciSyncAttrList structure where NvMedia places NvSciSync attributes.
 * \param[in] clienttype Indicates whether the attrlist is requested for an NvMediaDla signaler or an NvMediaDla waiter or an NvMediaDla signaler-waiter.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the call is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if attrlist is NULL, or any of the above listed public attributes are already set, or if client type is invalid.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre attrlist must be a valid NvSciSyncAttrList. This function must be called before allocating the NvSciSyncObj.
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
NvMediaDlaFillNvSciSyncAttrList(
    const NvMediaDla                *dla,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * \brief Fills the NvMediaDla specific NvSciSync deterministic attributes.
 *
 * \param[in] dla        An NvMedia DLA device handle.
 * \param[in,out] attrlist   A pointer to an NvSciSyncAttrList structure where NvMedia places NvSciSync attributes.
 * \param[in] clienttype Indicates whether the attrlist is requested for an NvMediaDla signaler or an NvMediaDla waiter.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the call is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if attrlist is NULL, or any of the above listed public attributes are already set, or if client type is invalid.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre attrlist must be a valid NvSciSyncAttrList. This function must be called before allocating the NvSciSyncObj.
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
NvMediaDlaFillNvSciSyncDeterministicAttrList(
    const NvMediaDla* dla,
    NvSciSyncAttrList attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * \brief Registers an NvSciSyncObj with NvMediaDla.
 *
 * \param[in] dla         An NvMedia DLA device handle.
 * \param[in] syncobjtype Determines how nvscisync is used by dla.
 * \param[in] nvscisync   The NvSciSyncObj to be registered with dla.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL or syncobjtype is not a valid NvMediaNvSciSyncObjType.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if nvscisync is not a compatible NvSciSyncObj which NvMediaDla can support.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObj objects are already registered for the given syncobjtype, OR if nvscisync is already registered with the same dla handle for a different syncobjtype.
 *
 * \pre nvscisync to be registered must have been created with the NvSciSyncAttrList returned by NvMedia-DLA.
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
NvMediaDlaRegisterNvSciSyncObj(
    NvMediaDla                *dla,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * \brief Unregisters an NvSciSyncObj with NvMediaDla.
 *
 * \param[in] dla        An NvMedia DLA device handle.
 * \param[in] scisyncobj An NvSciSyncObj to be unregistered with dla.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL, or nvscisync is not registered with dla.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR if dla is destroyed before this function is called.
 *
 * \pre scisyncobj have been created and registered with the input NvMediaDLA handle.
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
 *   - De-Init: Yes
 */
NvMediaStatus
NvMediaDlaUnregisterNvSciSyncObj(
    NvMediaDla                *dla,
    NvSciSyncObj               scisyncobj
);

/**
 * \brief Sets the NvSciSyncObj to be used for a Start of Frame (SOF) NvSciSyncFence.
 *
 * \param[in] dla          An NvMedia DLA device handle.
 * \param[in] nvscisyncSOF A registered NvSciSyncObj to be associated with SOF NvSciSyncFence.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL, or if nvscisyncSOF is not registered with dla as either type NVMEDIA_SOFSYNCOBJ or type NVMEDIA_SOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if nvscisyncSOF is backed by deterministic primitive.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre nvscisyncSOF must have been created and registered with the input NvMediaDLA handle using NvMediaDlaRegisterNvSciSyncObj().
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
NvMediaDlaSetNvSciSyncObjforSOF(
    NvMediaDla                *dla,
    NvSciSyncObj               nvscisyncSOF
);

/**
 * \brief Sets an NvSciSyncObj to be used for a End of Frame (EOF) NvSciSyncFence.
 *
 * \param[in] dla          An NvMedia DLA device handle.
 * \param[in] nvscisyncEOF A registered NvSciSyncObj which is to be associated with EOF NvSciSyncFence.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL, OR if nvscisyncEOF is not registered with dla as either type NVMEDIA_EOFSYNCOBJ or type NVMEDIA_EOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if nvscisyncEOF is backed by deterministic primitive.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre nvscisyncEOF must have been created and registered with the input NvMediaDLA handle NvMediaDlaRegisterNvSciSyncObj().
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
NvMediaDlaSetNvSciSyncObjforEOF(
    NvMediaDla                *dla,
    NvSciSyncObj               nvscisyncEOF
);

/**
 * \brief Sets an NvSciSyncFence as a prefence for a DLA submit operation.
 *
 * \param[in] dla               An NvMedia DLA device handle.
 * \param[in] prenvscisyncfence A pointer to NvSciSyncFence.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is not a valid NvMediaDla handle, or prenvscisyncfence is NULL, or prenvscisyncfence is not generated with an NvSciSyncObj that was registered with dla as either type NVMEDIA_PRESYNCOBJ or type NVMEDIA_EOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if NvMediaDlaInsertPreNvSciSyncFence is already called at least NVMEDIA_DLA_MAX_PRENVSCISYNCFENCES times with the same dla NvMediaDla handle before a DLA submit call.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvSciSyncObj associated with prenvscisyncfence must have been registered with the input NvMediaDLA handle using NvMediaDlaRegisterNvSciSyncObj().
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
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaInsertPreNvSciSyncFence(
    NvMediaDla                *dla,
    const NvSciSyncFence      *prenvscisyncfence
);

/**
 * \brief Gets an SOF NvSciSyncFence for a DLA submit operation.
 *
 * \param[in] dla                An NvMedia DLA device handle.
 * \param[in] sofnvscisyncobj    The SOF NvSciSyncObj associated with the NvSciSyncFence being requested.
 * \param[in,out] sofnvscisyncfence  A pointer to the SOF NvSciSyncFence.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is not a valid NvMediaDla handle, or sofnvscisyncfence is NULL, or sofnvscisyncobj is not registered with dla as type NVMEDIA_SOFSYNCOBJ or type NVMEDIA_SOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if sofnvscisyncobj is backed by deterministic primitive.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR if the function is called before setting the loadable as current or if there is a failure while updating the sofnvscisyncfence with the fence from the sofnvscisyncobj.
 *
 * \pre sofnvscisyncobj must have been created and registered with the input NvMediaDLA handle using NvMediaDlaRegisterNvSciSyncObj().
 *      sofnvscisyncobj must have been set as the active SOF NvSciSyncObj using NvMediaDlaSetNvSciSyncObjforSOF() for non-deterministic
 *      NvSciSyncObj and NvMediaDlaInsertSOFNvSciSyncObj() for deterministic NvSciSyncObj. A task must have been submitted to the DLA
 *      engine using NvMediaDlaSubmit().
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
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetSOFNvSciSyncFence(
    const NvMediaDla                *dla,
    NvSciSyncObj               sofnvscisyncobj,
    NvSciSyncFence            *sofnvscisyncfence
);

/**
 * \brief Gets an EOF NvSciSyncFence for a DLA submit operation.
 *
 * \param[in] dla                An NvMedia DLA device handle.
 * \param[in] eofnvscisyncobj    An EOF NvSciSyncObj associated with the NvSciSyncFence being requested.
 * \param[in,out] eofnvscisyncfence  A pointer to the EOF NvSciSyncFence.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is not a valid NvMediaDla handle, or eofnvscisyncfence is NULL, or eofnvscisyncobj is not registered with dla as type NVMEDIA_EOFSYNCOBJ or type NVMEDIA_EOF_PRESYNCOBJ.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED if eofnvscisyncobj is backed by deterministic primitive.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR if the function is called before setting the loadable as current or if there is a failure while updating the eofnvscisyncfence with the fence from the eofnvscisyncobj.
 *
 * \pre eofnvscisyncobj must have been created and registered with the input NvMediaDLA handle using NvMediaDlaRegisterNvSciSyncObj().
 *      eofnvscisyncobj must have been set as the active EOF NvSciSyncObj using NvMediaDlaSetNvSciSyncObjforEOF() for non-deterministic
 *      NvSciSyncObj and NvMediaDlaInsertEOFNvSciSyncObj() for deterministic NvSciSyncObj. A task must have been submitted to the DLA
 *      engine using NvMediaDlaSubmit().
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
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetEOFNvSciSyncFence(
    const NvMediaDla                *dla,
    NvSciSyncObj               eofnvscisyncobj,
    NvSciSyncFence            *eofnvscisyncfence
);

/**
 * \brief Sets NvSciSyncObj as a SOF for a DLA submit operation.
 *
 * \param[in] dla     An NvMedia DLA device handle.
 * \param[in] syncObj NvSciSyncObj that needs to be used as SOF for current submission.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is not a valid NvMediaDla handle.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR if current loadable is not set, if syncObj is not registered with NvMediaDla for SOF operation, if function fails to set syncObj as active SOF event for the current submission.
 *
 * \pre syncObj must have been created and registered with the input NvMediaDLA handle using NvMediaDlaRegisterNvSciSyncObj().
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
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaInsertSOFNvSciSyncObj(
    NvMediaDla* dla,
    NvSciSyncObj syncObj
);

/**
 * \brief Sets NvSciSyncObj as a EOF for a DLA submit operation.
 *
 * \param[in] dla     An NvMedia DLA device handle.
 * \param[in] syncObj NvSciSyncObj that needs to be used as EOF for current submission.
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if the function is successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if dla is not a valid NvMediaDla handle.
 * - \ref NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS state that is not allowed as per the API Group.
 * - \ref NVMEDIA_STATUS_ERROR if current loadable is not set, if syncObj is not registered with NvMediaDla for EOF operation, if function fails to set syncObj as active EOF event for the current submission.
 *
 * \pre syncObj must have been created and registered with the input NvMediaDLA handle using NvMediaDlaRegisterNvSciSyncObj().
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
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaInsertEOFNvSciSyncObj(
    NvMediaDla* dla,
    NvSciSyncObj syncObj
);

/*
 * \defgroup history_nvmedia_dla_nvscisync History
 * Provides change history for the NvMedia Dla NvSciSync API
 *
 * \section history_nvmedia_dla_nvscisync Version History
 *
 * <b> Version 1.0 </b> March 14, 2019
 * - Initial release
 *
 * <b> Version 1.1 </b> April 11, 2019
 * - Add new API NvMediaDlaSetNvSciSyncObjforSOF and NvMediaDlaGetEOFNvSciSyncFence
 * - Rename NvMediaDlaUnRegisterNvSciSyncObj to NvMediaDlaUnregisterNvSciSyncObj
 *
 * <b> Version 1.2 </b> Jan 22, 2020
 * - Disable NvMediaDlaSetNvSciSyncObjforSOF and NvMediaDlaGetSOFNvSciSyncFence in
 *   safety build as they are currently unsupported.
 *
 * <b> Version 1.3 </b> Jul 20, 2020
 * - Added support for NvSciSyncObj backed by deterministic primitive.
 * - Currently timestamp feature is disabled with NvSciSyncObj backed by
 *  deterministic primitive.
 * - Added new APIs: NvMediaDlaInsertEOFNvSciSyncObj,
 *      NvMediaDlaInsertSOFNvSciSyncObj (disabled in safety),
 *      NvMediaDlaFillNvSciSyncDeterministicAttrList
 *
 * <b> Version 1.4 </b> July 26, 2021
 * - Update comments for NvMediaDlaGetEOFNvSciSyncFence and NvMediaDlaGetSOFNvSciSyncFence
 *
 * <b> Version 1.5 </b> August 20, 2021
 * - Update doxygen comments for All APIs to have Thread safety information and API Group information
 *
 * <b> Version 1.6 </b> October 25, 2021
 * - Enable SOF feature in safety builds.
 * - Enable timestamp support for all primitives.
 *
 * <b> Version 1.6.0 </b> May 10, 2022
 * - Added patch version number macro: NVMEDIA_DLA_NVSCISYNC_VERSION_PATCH.
 *
 */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_DLA_NVSCISYNC_H */
