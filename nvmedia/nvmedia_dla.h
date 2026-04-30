/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__dla_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: DLA
 *
 * This file contains the DLA runtime APIs.
 */

#ifndef NVMEDIA_DLA_H
#define NVMEDIA_DLA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "nvmedia_core.h"
#include "nvmedia_tensor.h"

/** \brief Major version number. */
#define NVMEDIA_DLA_VERSION_MAJOR   4

/** \brief Minor version number. */
#define NVMEDIA_DLA_VERSION_MINOR   0

/** \brief Patch version number. */
#define NVMEDIA_DLA_VERSION_PATCH   0

/** \brief Maximum length of the name of the tensor to store in the descriptor. */
#define NVMEDIA_DLA_TENSOR_DESC_NAME_MAX_LEN (80U)

/** \brief Default task timeout. */
#define NVMEDIA_DLA_DEFAULT_TASKTIMEOUT (100000U)

/**
 * \brief Holds a handle to the NvMedia DLA device.
 *
 * The NvMediaDla handle must be created using NvMediaDlaCreate() and destroyed
 * using NvMediaDlaDestroy().
 */
typedef struct NvMediaDla NvMediaDla;

/**
 * \brief Holds the DLA UMD version.
 */
typedef struct {
    /** Major version.
     *  Valid value range is [0 , UINT8MAX] */
    uint8_t major;
    /** Minor version.
     *  Valid value range is [0 , UINT8MAX] */
    uint8_t minor;
    /** Sub-minor version.
     *  Valid value range is [0 , UINT8MAX] */
    uint8_t subMinor;
} NvMediaDlaUMDVersion;

/**
 * \brief Defines the data types that DLA can operate on.
 */
typedef enum {
    /** NvMediaTensor data type. */
    NVMEDIA_DLA_DATA_TYPE_TENSOR
} NvMediaDlaDataType;

/**
 * \brief Holds pointers to the DLA data.
 */
typedef struct {
    /** A pointer to NvMediaTensor.
     *  Valid value is non NULL. */
    NvMediaTensor *tensor;
} NvMediaDlaDataPointer;

/**
 * \brief Specifies the data type and data pointer.
 */
typedef struct {
    /** An NvMediaDlaDataType type.
     *  Valid value is NVMEDIA_DLA_DATA_TYPE_TENSOR. */
    NvMediaDlaDataType type;
    /** A pointer to NvMediaDlaDataPointer.
     *  Valid value is non NULL. */
    NvMediaDlaDataPointer pointer;
} NvMediaDlaData;

/**
 * \brief Holds a handle to NvMediaDlaLoadable.
 *
 * The handle must be created using NvMediaDlaLoadableCreate() and destroyed
 * using NvMediaDlaLoadableDestroy().
 */
typedef struct NvMediaDlaLoadable NvMediaDlaLoadable;

/**
 * \brief Holds attributes for populating binary loadables for NvMediaDla.
 */
typedef struct {
    /** Pointer to in-memory loadable.
     *  Valid value must be non NULL. */
    uint8_t const *loadable;
    /** Size (in bytes) of in-memory loadable.
     *  Valid value must not be 0U. */
    uint64_t loadableSize;
} NvMediaDlaBinaryLoadable;

/**
 * \brief Holds input and output DLA data.
 */
typedef struct {
    /** Pointer to array of NvMediaDlaData. Valid value must be non NULL. */
    NvMediaDlaData *dlaData;
    /** Number of NvMediaDlaData. Valid value range is [0 , UINTMAX32] */
    uint32_t numArgs;
} NvMediaDlaArgs;

/**
 * \brief Holds tensor attributes.
 */
typedef struct {
    /** Holds the name of the tensor. */
    char name[NVMEDIA_DLA_TENSOR_DESC_NAME_MAX_LEN + 1];
    /** Holds tensor attributes NvMediaTensorAttr. */
    NvMediaTensorAttr tensorAttrs[NVM_TENSOR_ATTR_MAX];
    /** Holds the number of tensor attributes in tensorAttrs[ ]. */
    uint32_t numAttrs;
} NvMediaDlaTensorDescriptor;

/**
 * \brief Returns the version information for the NvMediaDla library.
 *
 * \param[in,out] version A pointer to an NvMediaVersion structure filled by
 *                        the NvMediaDla library.
 *                        Input range: A non-null pointer to an NvMediaVersion
 *                        structure.
 *                        Output range: A non-null pointer to an NvMediaVersion
 *                        structure if successful, otherwise the value pointed
 *                        to by version remains unchanged.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if version is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetVersion(
    NvMediaVersion *version
);

/**
 * \brief Checks the status of the DLA engine.
 *
 * This function sends a ping to the DLA engine identified by dlaId to fetch
 * its status.
 *
 * \param[in] dlaId Id of the DLA engine to ping.
 *                  Input range: An uint32_t value between 0 (zero) and the
 *                  result of NvMediaDlaGetNumEngines() - 1.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if dlaId is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaPingById(
    const uint32_t dlaId
);

/**
 * \brief Creates a default context for NvMediaDla.
 *
 * Use the function NvMediaDlaInit() to initialize the newly-created context
 * for use as an engine instance.
 *
 * \return NvMediaDla A handle to the context.
 *         Output range: A non-null pointer to an NvMediaDla if successful, or
 *         NULL otherwise.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaDla *
NvMediaDlaCreate(void);

/**
 * \brief Destroys a DLA engine instance created by NvMediaDlaCreate().
 *
 * \param[in] dla A handle to the instance to destroy.
 *                Input range: A non-null pointer to an NvMediaDla instance
 *                created with NvMediaDlaCreate().
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvMediaStatus
NvMediaDlaDestroy(
     NvMediaDla *dla
);

/**
 * \brief Returns the version information for the NvMedia DLA UMD library.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla instance
 *                created with NvMediaDlaCreate().
 * \param[in,out] version A pointer to an NvMediaDlaUMDVersion structure filled
 *                        by the NvMediaDla library.
 *                        Input range: A non-null pointer to an
 *                        NvMediaDlaUMDVersion.
 *                        Output range: A non-null pointer to an
 *                        NvMediaDlaUMDVersion if successful, otherwise the
 *                        value pointed to by version remains unchanged.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetUMDVersion(
    const NvMediaDla *dla,
    NvMediaDlaUMDVersion *version
);

/**
 * \brief Returns the number of DLA hardware engines available.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer an NvMediaDla instance
 *                created with NvMediaDlaCreate().
 * \param[in,out] numEngines A pointer to the number of DLA hardware engines.
 *                           Input range: A non-null pointer to an uint16_t.
 *                           Output range: A non-null pointer to an uint16_t
 *                           if successful, otherwise the value pointed to by
 *                           numEngines remains unchanged.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetNumEngines(
    const NvMediaDla *dla,
    uint16_t *numEngines
);

/**
 * \brief Returns the maximum number of tasks that can be queued to an instance
 *        of an engine.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in,out] maxOutstandingTasks A pointer to the maximum number of tasks
 *                                    that can be queued.
 *                                    Input range: A non-null pointer to an
 *                                    uint32_t.
 *                                    Output range: A non-null pointer to an
 *                                    uint32_t if successful, otherwise the
 *                                    value pointed to by numEngines remains
 *                                    unchanged.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetMaxOutstandingTasks(
    const NvMediaDla *dla,
    uint32_t *maxOutstandingTasks
);

/**
 * \brief Configures the context for a particular DLA engine.
 *
 * Once the dlaId and numTasks have been assigned, their values cannot be
 * modified.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] dlaId The DLA engine ID.
 *                  Input range: An uint32_t value between 0 (zero) and the
 *                  result of NvMediaDlaGetNumEngines() - 1.
 * \param[in] numTasks The number of simultaneous tasks that can be submitted
 *                     to a DLA instance at a time.
 *                     Input range: An uint32_t value between 1 and the result
 *                     of NvMediaDlaGetMaxOutstandingTasks().
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaInit(
    NvMediaDla *dla,
    uint32_t dlaId,
    uint32_t numTasks
);

/**
 * \brief Returns the instance id of the NvMediaDla.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in,out] dlaId A pointer to the NvMediaDla instance id.
 *                      Input range: A non-null pointer to an uint32_t.
 *                      Output range: A non-null pointer to an uint32_t between
 *                      0 (zero) and the result of NvMediaDlaGetNumEngines() - 1
 *                      if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_NOT_INITIALIZED if NvMediaDla handle has not been
 *         initialzied.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR if dlaId of the NvMediaDla has not been set.
 *
 * \pre dla must have been initialized using NvMediaDlaInit()
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetInstanceId(
    const NvMediaDla *dla,
    uint32_t *dlaId
);

/**
 * \brief Returns the number of outstanding tasks of NvMediaDla.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in,out] numTasks A pointer to the number of outstanding tasks.
 *                         Input range: A non-null pointer to an uint32_t.
 *                         Output range: A non-null pointer to an uint32_t
 *                         between 0 (zero) and the result of
 *                         NvMediaDlaGetMaxOutstandingTasks() if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_NOT_INITIALIZED if NvMediaDla handle has not been
 *         initialzied.
 *         NVMEDIA_STATUS_ERROR if numTasks of the NvMediaDla has not been set.
 *
 * \pre dla must have been initialized using NvMediaDlaInit()
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetNumTasks(
    const NvMediaDla *dla,
    uint32_t *numTasks
);

/**
 * \brief Creates a loadable handle.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in,out] loadable A pointer to an NvMediaDlaLoadable.
 *                         Input range: A non-null pointer to an
 *                         NvMediaDlaLoadable.
 *                         Output range: A non-null pointer to a valid loadable
 *                         if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaLoadableCreate(
    NvMediaDla *dla,
    NvMediaDlaLoadable **loadable
);

/**
 * \brief Destroys a loadable handle.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] loadable A pointer to an NvMediaDlaLoadable.
 *                     Input range: A non-null pointer to an NvMediaDlaLoadable
 *                     created with NvMediaDlaLoadableCreate().
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvMediaStatus
NvMediaDlaLoadableDestroy(
    const NvMediaDla *dla,
    NvMediaDlaLoadable *loadable
);

/**
 * \brief Appends a loadable to the NvMediaDla context.
 *
 * NvMediaDlaAppendLoadable() appends the value in loadable and returns to the
 * calling function. Currently, only one loadable can be appended.
 *
 * The application may call this function or NvMediaDlaAppendDiagnosticLoadable
 * only once per NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] binaryLoadable Holds a non-null pointer to NvMediaDlaBinaryLoadable.
 *                           Input range: The loadableSize must be greater than 0.
 * \param[in,out] loadable A pointer to an NvMediaDlaLoadable filled by NvMediaDla.
 *                         Input range: A non-null pointer to an NvMediaDlaLoadable
 *                         created using NvMediaDlaLoadableCreate()
 *                         Output range: A non-null pointer to a valid loadable
 *                         if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre binaryLoadable must be a valid handle to a loadable created using
 *      TensorRT APIs.
 *      The user must provide a valid NvMediaDlaLoadable handle and the
 *      corresponding NvMediaDla handle
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaAppendLoadable(
    const NvMediaDla *dla,
    const NvMediaDlaBinaryLoadable binaryLoadable,
    NvMediaDlaLoadable *loadable
);

/**
 * \brief Appends diagnostic loadable to the NvMediaDla context.
 *
 * NvMediaDlaAppendDiagnosticLoadable() appends the value in loadable and
 * returns to the calling function. Currently, only one loadable can be
 * appended.
 *
 * The application may call this function or NvMediaDlaAppendLoadable only
 * once per NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created
 *                using NvMediaDlaCreate().
 * \param[in,out] loadable A pointer to an NvMediaDlaLoadable filled by
 *                         NvMediaDla.
 *                         Input range: A non-null pointer to an
 *                         NvMediaDlaLoadable created using
 *                         NvMediaDlaLoadableCreate()
 *                         Output range: A non-null pointer to a valid loadable
 *                         if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre None
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaAppendDiagnosticLoadable(
    const NvMediaDla *dla,
    NvMediaDlaLoadable *loadable
);

/**
 * \brief Sets the current loadable for the NvMediaDla.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] loadable A pointer to an NvMediaDlaLoadable.
 *                     Input range: A non-null pointer to an NvMediaDlaLoadable
 *                     created with NvMediaDlaLoadableCreate().
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre A binary loadable must have been appended to loadable using
 *      NvMediaDlaAppendLoadable().
 *      The user must provide a valid NvMediaDlaLoadable handle and the
 *      corresponding NvMediaDla handle
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
NvMediaDlaSetCurrentLoadable(
    NvMediaDla *dla,
    const NvMediaDlaLoadable *loadable
);

/**
 * \brief Gets the number of input tensors for the current loadable in the
 *        NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in,out] numOfInputTensors The number of input tensors.
 *                                  Input range: A non-null pointer to an
 *                                  int32_t value.
 *                                  Output range: A non-null pointer to an
 *                                  int32_t if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre NvMediaDlaLoadable associated with dla, the input NvMediaDla handle,
 *      must have been set as current using NvMediaDlaSetCurrentLoadable().
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetNumOfInputTensors(
    const NvMediaDla *dla,
    int32_t *numOfInputTensors
);

/**
 * \brief Gets the input tensor descriptor for the current loadable in the
 *        NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] idx A tensor index.
 *                Input range: A value between 0 (zero) and the result of
 *                NvMediaDlaGetNumOfInputTensors() - 1.
 * \param[in,out] descriptor Descriptor for the tensor.
 *                           Input range: A non-null pointer to an
 *                           NvMediaDlaTensorDescriptor.
 *                           Output range: A non-null pointer to an
 *                           NvMediaDlaTensorDescriptor if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL or if Tensor
 *         Index overflow occurs.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvMediaDlaLoadable associated with dla, the input NvMediaDla
 *      handle, must have been set as current using
 *      NvMediaDlaSetCurrentLoadable().
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetInputTensorDescriptor(
    const NvMediaDla *dla,
    const uint32_t idx,
    NvMediaDlaTensorDescriptor *descriptor
);

/**
 * \brief Gets the number of output tensors for the current loadable in the
 *        NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in,out] numOfOutputTensors The number of output tensors.
 *                                   Input range: A non-null pointer to an
 *                                   int32_t value.
 *                                   Output range: A non-null pointer to an
 *                                   int32_t value if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvMediaDlaLoadable associated with dla, the input NvMediaDla
 *      handle, must have been set as current using
 *      NvMediaDlaSetCurrentLoadable().
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetNumOfOutputTensors(
    const NvMediaDla *dla,
    int32_t *numOfOutputTensors
);

/**
 * \brief Gets the output tensor descriptor for the current loadable in the
 *        NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] idx The tensor index.
 *                Input range: A value between 0 (zero) and the result of
 *                NvMediaDlaGetNumOfOutputTensors() - 1.
 * \param[in,out] descriptor A descriptor for the tensor.
 *                           Input range: A non-null pointer to an
 *                           NvMediaDlaTensorDescriptor.
 *                           Output range: A non-null pointer to an
 *                           NvMediaDlaTensorDescriptor if successful.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL or if Tensor
 *         Index overflow occurs.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvMediaDlaLoadable associated with dla, the input NvMediaDla
 *      handle, must have been set as current using
 *      NvMediaDlaSetCurrentLoadable().
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaGetOutputTensorDescriptor(
    const NvMediaDla *dla,
    const uint32_t idx,
    NvMediaDlaTensorDescriptor *descriptor
);

/**
 * \brief Registers an NvMediaDlaData for use with an NvMediaDla handle.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] dlaData A pointer to NvMediaDlaData to register.
 *                    Input range: A non-null pointer to an NvMediaDlaData.
 * \param[in] flags Reserved for future use. Should be set to zero.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvMediaDlaLoadable associated with dla, the input NvMediaDla
 *      handle, must have been set as current using
 *      NvMediaDlaSetCurrentLoadable().
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaDataRegister(
    const NvMediaDla *dla,
    const NvMediaDlaData *dlaData,
    uint32_t flags
);

/**
 * \brief Unregisters an NvMediaDlaData after use.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] dlaData A pointer to a registered NvMediaDlaData.
 *                    Input range: A non-null pointer to an NvMediaDlaData
 *                    registered with NvMediaDlaDataRegister().
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if any argument is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvMediaDlaLoadable associated with dla, the input NvMediaDla
 *      handle, must have been set as current using
 *      NvMediaDlaSetCurrentLoadable()
 *      dlaData must have been registered
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvMediaStatus
NvMediaDlaDataUnregister(
    const NvMediaDla *dla,
    const NvMediaDlaData *dlaData
);

/**
 * \brief Loads the current loadable to the provided NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA device.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvMediaDlaLoadable associated with dla, the input NvMediaDla
 *      handle, must have been set as current using
 *      NvMediaDlaSetCurrentLoadable()
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: Yes
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaLoadLoadable(
    NvMediaDla *dla
);

/**
 * \brief Removes the current loadable from the provided NvMediaDla context.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if dla is NULL.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \pre The NvMediaDlaLoadable associated with dla, the input NvMediaDla
 *      handle, must have been set as current using
 *      NvMediaDlaSetCurrentLoadable().
 *      The NvMediaDlaLoadable associated with dla, must have been loaded
 *      using NvMediaDlaLoadLoadable().
 *      The user must provide a valid NvMediaDlaLoadable handle and the
 *      corresponding NvMediaDla handle
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvMediaStatus
NvMediaDlaRemoveLoadable(
    NvMediaDla *dla
);

/**
 * \brief Submits a job to the DLA to run the network on a set of input
 *        NvMediaDla arguments and a timeout value.
 *
 * The scratchpadArgs parameter is currently not supported. The application
 * must pass a NULL in this argument.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] inputArgs A pointer to input NvMediaDlaArgs.
 *                      Input range: A non-null pointer to a valid
 *                      NvMediaDlaArgs.
 * \param[in] scratchpadArgs A pointer to scratchpad arguments for NvMediaDla.
 *                           Input range: A NULL value. This parameter is
 *                           currently not supported. The application must pass
 *                           NULL in this argument.
 * \param[in] outputArgs Holds output NvMediaDlaArgs.
 *                       Input range: A non-null pointer to a valid
 *                       NvMediaDlaArgs.
 * \param[in] taskTimeout The maximum time allocated for completing the task
 *                        (in milliseconds).
 *                        Input range: The valid range is from 1 millisecond to
 *                        1000000 milliseconds. The value 0 (zero) is mapped to
 *                        1000000 milliseconds.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if invalid parameters or if overflows
 *         occur.
 *         NVMEDIA_STATUS_NOT_SUPPORTED if input/output parameters are not
 *         valid or dont match.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \note [1] NvMediaDlaArgs should hold a non-NULL NvMediaDlaData.
 *       [2] NvMediaDlaDataType of NvMediaDlaData shall be equal to
 *       NVMEDIA_DLA_DATA_TYPE_TENSOR.
 *       [3] NvMediaDlaDataPointer of NvMediaDlaData shall be non-NULL.
 *       [4] NvMediaTensor of NvMediaDlaDataPointer shall match the tensor
 *       descriptor that is queried by NvMediaDlaGetInputTensorDescriptor or
 *       NvMediaDlaGetOutputTensorDescriptor.
 *       [5] numArgs of NvMediaDlaArgs shall be equal to the number of tensors
 *       queried by NvMediaDlaGetNumOfInputTensors or
 *       NvMediaDlaGetNumOfOutputTensors.
 *       The task will error out if it does not finish within the time set in
 *       taskTimeout. Error details can be obtained: [1] using
 *       NvMediaTensorGetStatus() for output tensors or [2] using
 *       NvSciSyncFenceGetTaskStatus from the fence associated with EOF
 *       NvSciSyncObj
 *
 * \pre Client must have created a NvMediaDla handle.
 *      The binary loadable corresponding to the network must have been loaded.
 *      The input and output tensors that are part of inputArgs and outputArgs
 *      respectively, must have been registered with the NvMediaDla handle.
 *      If any prefences are available, they must have associated using
 *      NvMediaDlaInsertPreNvSciSyncFence().
 *      If any EOF/SOF NvSciSyncObjs are registered, they must have been set
 *      as current.
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaSubmit(
    NvMediaDla *dla,
    const NvMediaDlaArgs *inputArgs,
    const NvMediaDlaArgs *scratchpadArgs,
    const NvMediaDlaArgs *outputArgs,
    uint32_t taskTimeout
);

/**
 * \brief Submits a job to the DLA but its execution is skipped.
 *
 * The scratchpadArgs parameter is currently not supported. The application
 * must pass a NULL in this argument.
 *
 * \param[in] dla A handle to the DLA context.
 *                Input range: A non-null pointer to an NvMediaDla created with
 *                NvMediaDlaCreate().
 * \param[in] inputArgs A pointer to input NvMediaDlaArgs.
 *                      Input range: A non-null pointer to a valid
 *                      NvMediaDlaArgs.
 * \param[in] scratchpadArgs A pointer to scratchpad arguments for NvMediaDla.
 *                           Input range: A NULL value. This parameter is
 *                           currently not supported. The application must pass
 *                           NULL in this argument.
 * \param[in] outputArgs Holds output NvMediaDlaArgs.
 *                       Input range: A non-null pointer to a valid
 *                       NvMediaDlaArgs.
 *
 * \return NVMEDIA_STATUS_OK if the function is successful.
 *         NVMEDIA_STATUS_BAD_PARAMETER if invalid parameters or if overflows
 *         occur.
 *         NVMEDIA_STATUS_NOT_SUPPORTED if input/output parameters are not
 *         valid or dont match.
 *         NVMEDIA_STATUS_INVALID_STATE if the API is triggered in the DRIVEOS
 *         state that is not allowed as per the API Group.
 *         NVMEDIA_STATUS_ERROR for any other error.
 *
 * \note [1] NvMediaDlaArgs should hold a non-NULL NvMediaDlaData.
 *       [2] NvMediaDlaDataType of NvMediaDlaData shall be equal to
 *       NVMEDIA_DLA_DATA_TYPE_TENSOR.
 *       [3] NvMediaDlaDataPointer of NvMediaDlaData shall be non-NULL.
 *       [4] NvMediaTensor of NvMediaDlaDataPointer shall match the tensor
 *       descriptor that is queried by NvMediaDlaGetInputTensorDescriptor or
 *       NvMediaDlaGetOutputTensorDescriptor.
 *       [5] numArgs of NvMediaDlaArgs shall be equal to the number of tensors
 *       queried by NvMediaDlaGetNumOfInputTensors or
 *       NvMediaDlaGetNumOfOutputTensors.
 *       Error details can be obtained: [1] using NvMediaTensorGetStatus() for
 *       output tensors or [2] using NvSciSyncFenceGetTaskStatus from the fence
 *       associated with EOF NvSciSyncObj
 *
 * \pre Client must have created a NvMediaDla handle.
 *      The binary loadable corresponding to the network must have been loaded.
 *      The input and output tensors that are part of inputArgs and outputArgs
 *      respectively, must have been registered with the NvMediaDla handle.
 *      If any prefences are available, they must have associated using
 *      NvMediaDlaInsertPreNvSciSyncFence().
 *      If any EOF/SOF NvSciSyncObjs are registered, they must have been set
 *      as current.
 *
 * Usage considerations
 * - Interrupt handler: No
 * - Signal handler: No
 * - Thread-safe: Yes
 * - Re-entrant: No
 * - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMediaDlaSubmitBypass(
    NvMediaDla *dla,
    const NvMediaDlaArgs *inputArgs,
    const NvMediaDlaArgs *scratchpadArgs,
    const NvMediaDlaArgs *outputArgs
);

/*
 * \defgroup history_nvmedia_dla History
 * Provides change history for the NvMedia DLA API.
 *
 * \section history_nvmedia_dla Version History
 *
 * <b> Version 1.0 </b> Oct 31, 2017
 * - Initial Release.
 *
 * <b> Version 1.1 </b> Feb 12, 2018
 * - Add separate functionality to load from memory.
 *
 * <b> Version 1.2 </b> Mar 12, 2018
 * - Add query API's for version, num engines, maxOutstandingRequests.
 *
 * <c> Version 1.3 </b> Mar 26, 2018
 * - Add helper to query tensor descriptor.
 *
 * <c> Version 1.4 </b> May 3, 2018
 * - Add dynamic task timeout support.
 *
 * <c> Version 2.0 </b> June 26, 2018
 * - Remove load using filename functionality.
 *
 * <c> Version 2.1 </b> August 13, 2018
 * - Add NvMediaDlaPingById. NvMediaDlaPing will be deprecated in the future.
 *
 * <c> Version 2.2 </b> Oct 1, 2018
 * - Add new API NvMediaDlaGetUMDVersion and data structure NvMediaDlaUMDVersion.
 *
 * <c> Version 2.3 </b> Dec 14, 2018
 * - Add new API NvMediaLoadableIdCreate, NvMediaLoadableIdDestroy,
 *   NvMediaDlaAppendLoadable, NvMediaDlaLoadLoadable, NvMediaDlaSetCurrentLoadable,
 *   NvMediaDlaInit, NvMediaDlaGetScratchpadDescriptor and NvMediaDlaBindScratchpad.
 * - Add new structure NvMediaDlaLoadableId, NvMediaDlaLoadable and NvMediaDlaScratchpadDescriptor.
 *
 * <c> Version 2.4 </b> Jan 9, 2019
 * - Remove NvMediaImage as Dla data.
 *
 * <c> Version 2.5 </b> Mar 7, 2019
 * - Rename NvMediaLoadableIdCreate to NvMediaDlaLoadableIdCreate.
 * - Rename NvMediaLoadableIdDestroy to NvMediaDlaLoadableIdDestroy.
 *
 * <c> Version 2.6 </b> Mar 11, 2019
 * - Add new API NvMediaDlaDataRegister and NvMediaDlaDataUnRegister.
 *
 * <c> Version 2.7 </b> Mar 13, 2019
 * - Deprecate NvMediaDlaPing API.
 *
 * <c> Version 2.8 </b> Mar 14, 2019
 * - Deprecate NvMediaDlaSubmit API.
 *
 * <c> Version 3.0 </b> April 19, 2019
 * - Add NvMediaDlaSubmit as new API for task submission.
 * - Move to new APIs for NvMediaDla programming.
 * - Mark NvMediaDlaLoadFromMemory, NvMediaDlaSubmitTimeout
 *   and NvMediaDlaGetMaxOutstandingRequests as deprecated APIs.
 *
 * <c> Version 3.1 </b> July 08, 2019
 * - Remove the restriction of calling sequence for data registration.
 *
 * <c> Version 3.2 </b> July 30, 2019
 * - Deleting NvMediaDlaGetScratchpadDescriptor and NvMediaDlaBindScratchpad.
 * - added const for parameters in APIs that do not modify the parameter
 *
 * <c> Version 3.3 </b> Sep 9th, 2019
 * - Deprecate following APIs: NvMediaDlaLoadFromMemory, NvMediaDlaSubmitTimeout
 *   and NvMediaDlaGetMaxOutstandingRequests
 *
 * <c> Version 3.4 </b> Jan 22, 2020
 * - Add const to NvMediaDlaSubmit to comply with MISRA rule 8.13
 *
 * <c> Version 3.5 </b> Mar 28, 2020
 * - Update possible return status of NvMediaDlaGetNumEngines, NvMediaDlaGetInstanceId,
 *   NvMediaDlaGetNumTasks, NvMediaDlaSubmit.
 *
 * <c> Version 3.6 </b> Apr 24, 2020
 * - Removed const from NvMediaDlaSetCurrentLoadable and
 *   NvMediaDlaRemoveLoadable to support new functionality
 *
 * <c> Version 3.7 </b> Nov 30, 2020
 * - Updated doxygen comments for NvMediaDlaSubmit.
 *
 * <c> Version 3.8 </b> Jan 5, 2021
 * - Updated last sentence of NvMediaDlaSubmit restrictions.
 *
 * <c> Version 3.9 </b> Mar 1, 2021
 * - Updated doxy comments for NvMediaDlaSubmit,NvMediaDlaGetInput/OutputTensorDescriptor.
 *
 * <c> Version 3.10 </b> Mar 22, 2021
 * - Updated doxy comments for NvMediaDlaAppendLoadable.
 *
 * <c> Version 3.11 </b> Jul 20, 2021
 * - Added "submit with bypass execution" feature (NvMediaDlaSubmitBypass).
 *
 * <b> Version 3.12 </b> August 20, 2021
 * - Update doxygen comments for All APIs to have Thread safety information and API Group information
 *
 * <b> Version 4.0 </b> September 2, 2021
 * - Adding Const Qualifier for loadable member in NvMediaDlaBinaryLoadable
 *
  * <b> Version 4.0 </b> February 08, 2022
 * - Updated the doxygen comments with usage considerations for all APIs.
 *
 * <b> Version 4.0 </b> April 04, 2022
 * - Updated doxy comments for NvMediaDlaAppendLoadable, NvMediaDlaRemoveLoadable.
 * - Updated valid range for structures.
 *
 * <b> Version 4.0.0 </b> May 10, 2022
 * - Added patch version number macro: NVMEDIA_DLA_VERSION_PATCH.
 *
 */
#ifdef __cplusplus
}
#endif

#endif // NVMEDIA_DLA_H
