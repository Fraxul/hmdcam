/*
 * SPDX-FileCopyrightText: Copyright (c) 2013-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__2d_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: 2D Processing Control
 *
 * This file contains the #image_2d_api "Image 2D Processing API."
 */

#ifndef NVMEDIA_2D_H
#define NVMEDIA_2D_H

#include "nvmedia_core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/** \brief Major version number of NvMedia 2D header. */
#define NVMEDIA_2D_VERSION_MAJOR 8

/** \brief Minor version number of NvMedia 2D header. */
#define NVMEDIA_2D_VERSION_MINOR 0

/** \brief Patch version number of NvMedia 2D header. */
#define NVMEDIA_2D_VERSION_PATCH 0

/**
 * \brief 2D filter mode.
 */
typedef enum {
  NVMEDIA_2D_FILTER_OFF = 0x1,

  NVMEDIA_2D_FILTER_LOW,

  NVMEDIA_2D_FILTER_MEDIUM,

  NVMEDIA_2D_FILTER_HIGH

} NvMedia2DFilter;

/**
 * \brief 2D rotation/transform.
 */
typedef enum {
  NVMEDIA_2D_TRANSFORM_NONE = 0x0,

  NVMEDIA_2D_TRANSFORM_ROTATE_90,

  NVMEDIA_2D_TRANSFORM_ROTATE_180,

  NVMEDIA_2D_TRANSFORM_ROTATE_270,

  NVMEDIA_2D_TRANSFORM_FLIP_HORIZONTAL,

  NVMEDIA_2D_TRANSFORM_INV_TRANSPOSE,

  NVMEDIA_2D_TRANSFORM_FLIP_VERTICAL,

  NVMEDIA_2D_TRANSFORM_TRANSPOSE

} NvMedia2DTransform;

/**
 * \brief Blending to use when compositing surfaces.
 */
typedef enum {
  NVMEDIA_2D_BLEND_MODE_DISABLED,

  NVMEDIA_2D_BLEND_MODE_CONSTANT_ALPHA,

  NVMEDIA_2D_BLEND_MODE_STRAIGHT_ALPHA,

  NVMEDIA_2D_BLEND_MODE_PREMULTIPLIED_ALPHA

} NvMedia2DBlendMode;

/**
 * \brief Attributes structure for NvMedia2DCreate().
 */
typedef struct
{
  /** Number of compose parameters objects to allocate. */
  uint32_t numComposeParameters;

  /** Maximum number of buffers that can be registered. */
  uint32_t maxRegisteredBuffers;

  /** Maximum number of sync objects that can be registered. */
  uint32_t maxRegisteredSyncs;

  /** Maximum number of filter buffers that can be created. */
  uint32_t maxFilterBuffers;

  /** Internal use only. */
  uint32_t flags;

} NvMedia2DAttributes;

/** \brief Stores configuration for the NvMedia2DCompose() operation. */
typedef uint32_t NvMedia2DComposeParameters;

/** \brief Stores a filter buffer which coefficients can be configured. */
typedef uint32_t NvMedia2DFilterBuffer;

/**
 * \brief Stores information returned from NvMedia2DCompose().
 */
typedef struct
{
  /** ID number for operation that was submitted to NvMedia2DCompose(). The
   * number will wrap once the uint64_t range has been exceeded. A value of 0
   * indicates that no operation was submitted. */
  uint64_t operationId;

} NvMedia2DComposeResult;

/**
 * \brief Coefficients values structure for 5-tap custom filter.
 */
typedef struct
{
  /** Array of coefficients values, ordered by phase, then by tap. */
  int16_t coeffs[32][5];
} NvMedia2DFilterCoefficients5Tap;

/**
 * \brief Coefficients values structure for 10-tap custom filter.
 */
typedef struct
{
  /** Array of coefficients values, ordered by phase, then by tap. The
   * coefficient values are interpreted as 10-bit signed binary fixed point
   * numbers in two's complement format, with (from least to most significant
   * bits): 8 fraction bits, 1 integer bit, 1 sign bit. Other bits of the
   * values are discarded. */
  int16_t coeffs[32][10];
} NvMedia2DFilterCoefficients10Tap;

/**
 * \brief Returns the version number of the NvMedia 2D library.
 *
 * This function returns the major and minor version number of the NvMedia 2D
 * library. The client must pass an NvMediaVersion struct to this function, and
 * the version information will be returned in this struct. This allows the
 * client to verify that the version of the library matches and is compatible
 * with the the version number of the header file they are using.
 *
 * \param[out] version  Pointer to an NvMediaVersion struct that will be
 *                      populated with the version information.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Version information returned successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER \a version is NULL.
 *
 * \pre None.
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMediaVersion object
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: Yes
 */
NvMediaStatus
NvMedia2DGetVersion(NvMediaVersion* const version);

/** \brief NvMedia2D Context. */
typedef struct NvMedia2D NvMedia2D;

/**
 * \brief Creates a new NvMedia2D context.
 *
 * This function creates a new instance of an NvMedia2D context, and returns a
 * pointer to that context. Ownership of this context is passed to the caller.
 * When no longer in use, the caller must destroy the context using the
 * NvMedia2DDestroy() function.
 *
 * Default attributes (when not specified by caller):
 * - numComposeParameters: 1
 * - maxRegisteredBuffers: 64
 * - maxRegisteredSyncs:   16
 * - maxFilterBuffers:     0
 * - flags:                0
 *
 * \param[out] handle  Pointer to receive the handle to the new NvMedia2D
 *                     context.
 * \param[in]  attr    Pointer to NvMedia2DAttributes struct, or NULL for
 *                     default attributes.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Context created successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER \a handle is NULL, or \a attr has bad
 *                                     attribute values.
 * - \ref NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect
 *                                     system state.
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED NvMedia 2D is not supported on this
 *                                     hardware platform.
 * - \ref NVMEDIA_STATUS_OUT_OF_MEMORY Memory allocation failed for internal
 *                                     data structures or device memory
 *                                     buffers.
 * - \ref NVMEDIA_STATUS_ERROR An internal failure occurred when trying to
 *                             create the context.
 *
 * \pre None.
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DCreate(NvMedia2D** const handle,
  NvMedia2DAttributes const* const attr);

/**
 * \brief Destroys the NvMedia2D context.
 *
 * This function destroys the specified NvMedia2D context.
 *
 * Before calling this function, the caller must ensure:
 * - There are no NvSciSync or NvSyncBuf objects still registered against the
 *   NvMedia2D context.
 * - All previous 2D operations submitted using NvMedia2DCompose() have
 *    completed.
 *
 * \param[in] handle  Pointer to the NvMedia2D context.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Context destroyed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER \a handle is NULL.
 * - \ref NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect
 *                                     system state.
 * - \ref NVMEDIA_STATUS_PENDING There are still some NvSciSync or NvSciBuf
 *                               objects registered.
 * - \ref NVMEDIA_STATUS_ERROR An internal failure occurred when trying to
 *                             destroy the context.
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvMediaStatus
NvMedia2DDestroy(NvMedia2D const* const handle);

/**
 * \brief Returns an NvMedia2DComposeParameters instance.
 *
 * This functions returns a handle to an NvMedia2DComposeParameters object.
 * The object will be initialized and ready to use. The caller takes ownership
 * of this handle. Ownership will be passed back to the NvMedia2D context when
 * it is subsequently used in the NvMedia2DCompose() operation.
 *
 * The handle returned in \a params is tied to the specific NvMedia2D context
 * instance passed in \a handle and cannot be used with other context
 * instances.
 *
 * The object will be initialized with these default values:
 * - source rectangle: set to the dimensions of the source surface
 * - destination rectangle: set to the dimensions of the destination surface
 * - filter: NVMEDIA_2D_FILTER_OFF
 * - transform: NVMEDIA_2D_TRANSFORM_NONE
 *
 * \param[in]  handle  Pointer to the NvMedia2D context.
 * \param[out] params  Pointer to an NvMedia2DComposeParameters, which will be
 *                     populated with the handle.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Parameters instance is initialized successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value, either:
 *                                     - \a handle is NULL
 *                                     - \a params is NULL
 * - \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING There is no free instance
 *                                              available.
 * - \ref NVMEDIA_STATUS_ERROR An internal failure occurred when trying to
 *                             retrieve the parameters object.
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DGetComposeParameters(NvMedia2D const* const handle,
  NvMedia2DComposeParameters* const params);

/**
 * \brief Performs a 2D compose operation.
 *
 * A compose operation transfers pixels from a set of source surfaces to a
 * destination surface, applying a variety of transformations to the pixel
 * values on the way.
 *
 * The surfaces can have different pixel formats. NvMedia 2D does the
 * necessary conversions between the formats.
 *
 * \note For a YUV surface type with 16-bit depth, only scale and crop are
 * supported. Pixel format conversion, transformations or multi-surface
 * composition is not supported.
 *
 * If the dimensions of the source rectangle do not match the dimensions of
 * the destination rectangle, the operation scales the pixels to fit the
 * destination rectangle. The filtering mode for scale defaults to
 * NVMEDIA_2D_FILTER_OFF. Additional filtering modes can be used by setting
 * the corresponding parameter using NvMedia2DSetSrcFilter().
 *
 * \param[in]  handle  Pointer to the NvMedia2D context.
 * \param[in]  params  An NvMedia2DComposeParameters handle.
 * \param[out] result  Pointer to NvMedia2DComposeResult struct that will be
 *                     populated with result info. May be NULL.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Operation submitted successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value:
 *                                     - \a handle is NULL
 *                                     - \a params is invalid
 *                                     - some of the compose parameters
 *                                       configured through \a params have
 *                                       invalid values
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED Requested operation is not supported by
 *                                     current platform.
 * - \ref NVMEDIA_STATUS_TIMED_OUT No space available in the command buffer
 *                                 for this operation, because previous
 *                                 operations are still pending.
 * - \ref NVMEDIA_STATUS_ERROR An internal failure occurred when trying to
 *                             perform the compose operation.
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a params must be valid NvMedia2DComposeParameters handle created with
 *      NvMedia2DGetComposeParameters().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DCompose(NvMedia2D const* const handle,
  NvMedia2DComposeParameters const params,
  NvMedia2DComposeResult* const result);

/**
 * \brief Sets the geometry for a source layer.
 *
 * \param[in] handle     Pointer to the NvMedia2D context.
 * \param[in] params     An NvMedia2DComposeParameters handle.
 * \param[in] index      Index of source layer to configure. Must be in range
 *                       [0, 15].
 * \param[in] srcRect    Pointer to an NvMediaRect that contains the source
 *                       rectangle, or NULL for default rectangle.
 * \param[in] dstRect    Pointer to an NvMediaRect that contains the
 *                       destination rectangle, or NULL for default rectangle.
 * \param[in] transform  An NvMedia2DTransform to apply the content region.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Parameters set successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value. This could be:
 *                                     - \a handle is NULL
 *                                     - \a params is invalid
 *                                     - \a index is out of range
 *                                     - \a srcRect is invalid
 *                                     - \a dstRect is invalid
 *                                     - \a transform is invalid
 * - \ref NVMEDIA_STATUS_NOT_SUPPORTED Requested operation is not supported by
 *                                     current platform.
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a params must be valid NvMedia2DComposeParameters handle created with
 *      NvMedia2DGetComposeParameters().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DSetSrcGeometry(NvMedia2D const* const handle,
  NvMedia2DComposeParameters const params,
  uint32_t const index,
  NvMediaRect const* const srcRect,
  NvMediaRect const* const dstRect,
  NvMedia2DTransform const transform);

/**
 * \brief Sets the filter mode for a source layer.
 *
 * \param[in] handle  Pointer to the NvMedia2D context.
 * \param[in] params  An NvMedia2DComposeParameters handle.
 * \param[in] index   Index of source layer to configure. Must be in range
 *                    [0, 15].
 * \param[in] filter  An NvMedia2DFilter to use when reading the layer's
 *                    source surface.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Parameters set successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value. This could be:
 *                                     - \a handle is NULL
 *                                     - \a params is invalid
 *                                     - \a index is out of range
 *                                     - \a filter is invalid
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a params must be valid NvMedia2DComposeParameters handle created with
 *      NvMedia2DGetComposeParameters().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DSetSrcFilter(NvMedia2D const* const handle,
  NvMedia2DComposeParameters const params,
  uint32_t const index,
  NvMedia2DFilter const filter);

/**
 * \brief Sets the blend mode for a source layer.
 *
 * \param[in] handle         Pointer to the NvMedia2D context.
 * \param[in] params         An NvMedia2DComposeParameters handle.
 * \param[in] index          Index of source layer to configure. Must be in
 *                           range [0, 15].
 * \param[in] blendMode      Blend mode to set.
 * \param[in] constantAlpha  Constant alpha factor to use in blending. Must be
 *                           in range [0, 1].
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Parameters set successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value. This could be:
 *                                     - \a handle is NULL
 *                                     - \a params is invalid
 *                                     - \a index is out of range
 *                                     - \a blendMode is invalid
 *                                     - \a constantAlpha is out of range
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a params must be valid NvMedia2DComposeParameters handle created with
 *      NvMedia2DGetComposeParameters().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DSetSrcBlendMode(NvMedia2D const* const handle,
  NvMedia2DComposeParameters const params,
  uint32_t const index,
  NvMedia2DBlendMode const blendMode,
  float const constantAlpha);

/**
 * \brief Creates and returns an NvMedia2DFilterBuffer instance.
 *
 * This functions returns a handle to an NvMedia2DFilterBuffer object. The
 * filter buffer can be used to provide custom 5-tap and 10-tap filter
 * coefficients for a compose operation.
 *
 * The handle returned in \a filterBuffer is tied to the specific NvMedia2D
 * context instance passed in \a handle and cannot be used with other context
 * instances.
 *
 * The buffer instance must be destroyed with NvMedia2DDestroyFilterBuffer()
 * during the De-Init stage.
 *
 * \param[in]  handle        Pointer to the NvMedia2D context.
 * \param[out] filterBuffer  Pointer to an NvMedia2DFilterBuffer, which will
 *                           be populated with the handle.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Buffer created successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value, either:
 *                                     - \a handle is NULL
 *                                     - \a filterBuffer is NULL
 * - \ref NVMEDIA_STATUS_INSUFFICIENT_BUFFERING Maximum number of buffers has
 *                                              been created.
 * - \ref NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect
 *                                     system state.
 * - \ref NVMEDIA_STATUS_OUT_OF_MEMORY Failed to allocate memory for the
 *                                     buffer.
 * - \ref NVMEDIA_STATUS_ERROR An internal failure occurred when trying to
 *                             create the buffer.
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DCreateFilterBuffer(NvMedia2D const* const handle,
  NvMedia2DFilterBuffer* const filterBuffer);

/**
 * \brief Destroys an NvMedia2DFilterBuffer instance.
 *
 * This functions destroys an NvMedia2DFilterBuffer object.
 *
 * \param[in] handle        Pointer to the NvMedia2D context.
 * \param[in] filterBuffer  An NvMedia2DFilterBuffer handle.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Buffer are destroyed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value, either:
 *                                     - \a handle is NULL
 *                                     - \a filterBuffer is invalid
 * - \ref NVMEDIA_STATUS_INVALID_STATE The function was called in incorrect
 *                                     system state.
 * - \ref NVMEDIA_STATUS_PENDING The buffer is still being used by a pending
 *                               operation.
 * - \ref NVMEDIA_STATUS_ERROR An internal failure occurred when trying to
 *                             destroy the buffer.
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a filterBuffer must be valid NvMedia2DFilterBuffer handle created
 *      with NvMedia2DCreateFilterBuffer().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: Yes
 */
NvMediaStatus
NvMedia2DDestroyFilterBuffer(NvMedia2D const* const handle,
  NvMedia2DFilterBuffer const filterBuffer);

/**
 * \brief Sets the filter buffer for an NvMedia2DComposeParameters instance.
 *
 * This function updates the NvMedia2DComposeParameters instance to indicate
 * that the specified filter buffer object shall be used to provide the custom
 * filter coefficients for the compose operation.
 *
 * After this function returns successfully, there are a few additional
 * limitations on the compose operation:
 * - Only the first 5 source layers can be used (indexes 0 to 4).
 * - If a source layer's NvMedia2DFilter is set to NVMEDIA_2D_FILTER_MEDIUM or
 *   NVMEDIA_2D_FILTER_HIGH, the custom 5-tap or 10-tap coefficients,
 *   respectively, for such layer shall have been properly set in the
 *   NvMedia2DFilterBuffer.
 *
 * Due to the filter buffer object being read-only from the compose operation
 * perspective, there is no limitation for the same filter buffer object to be
 * set for multiple NvMedia2DComposeParameters instances.
 *
 * \param[in] handle        Pointer to the NvMedia2D context.
 * \param[in] params        An NvMedia2DComposeParameters handle.
 * \param[in] filterBuffer  An NvMedia2DFilterBuffer handle.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Filter buffer was set successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value, either:
 *                                     - \a handle is NULL
 *                                     - \a params is invalid
 *                                     - \a filterBuffer is invalid
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a params must be valid NvMedia2DComposeParameters handle created with
 *      NvMedia2DGetComposeParameters().
 * \pre \a filterBuffer must be valid NvMedia2DFilterBuffer handle created
 *      with NvMedia2DCreateFilterBuffer().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DSetFilterBuffer(NvMedia2D const* const handle,
  NvMedia2DComposeParameters const params,
  NvMedia2DFilterBuffer const filterBuffer);

/**
 * \brief Computes the 5-tap filter coefficients for an NvMedia2DFilterBuffer.
 *
 * This function computes the filter coefficients values for a specific source
 * layer based on the contents of four NvMedia2DFilterCoefficients5Tap
 * structures.
 *
 * There is no restriction on multiple parameters pointing to the same
 * NvMedia2DFilterCoefficients5Tap structure.
 *
 * \param[in] handle        Pointer to the NvMedia2D context.
 * \param[in] filterBuffer  An NvMedia2DFilterBuffer handle.
 * \param[in] index         Index of source layer to configure. Must be in
 *                          range [0, 4].
 * \param[in] lumaX,lumaY   Pointers to NvMedia2DFilterCoefficients5Tap. These
 *                          configure to the luma component for YUV formats,
 *                          or all the components for RGB formats. There is
 *                          one pointer for the horizontal direction, and one
 *                          pointer for the vertical direction.
 * \param[in] chromaX,chromaY  Pointers to NvMedia2DFilterCoefficients5Tap.
 *                             These configure the chroma component for YUV
 *                             formats. There is one pointer for the
 *                             horizontal direction, and one pointer for the
 *                             vertical direction.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Filter buffer was updated successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value, either:
 *                                     - \a handle is NULL
 *                                     - \a filterBuffer is invalid
 *                                     - \a index is out of range
 *                                     - \a lumaX is NULL
 *                                     - \a lumaY is NULL
 *                                     - \a chromaX is NULL
 *                                     - \a chromaY is NULL
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a filterBuffer must be valid NvMedia2DFilterBuffer handle created
 *      with NvMedia2DCreateFilterBuffer().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DComputeFilterCoefficients5Tap(NvMedia2D const* const handle,
  NvMedia2DFilterBuffer const filterBuffer,
  uint32_t const index,
  NvMedia2DFilterCoefficients5Tap const* const lumaX,
  NvMedia2DFilterCoefficients5Tap const* const lumaY,
  NvMedia2DFilterCoefficients5Tap const* const chromaX,
  NvMedia2DFilterCoefficients5Tap const* const chromaY);

/**
 * \brief Computes the 10-tap filter coefficients for an NvMedia2DFilterBuffer.
 *
 * This function computes the filter coefficients values for a specific source
 * layer based on the contents of four NvMedia2DFilterCoefficients10Tap
 * structures.
 *
 * There is no restriction on multiple parameters pointing to the same
 * NvMedia2DFilterCoefficients10Tap structure.
 *
 * \param[in] handle        Pointer to the NvMedia2D context.
 * \param[in] filterBuffer  An NvMedia2DFilterBuffer handle.
 * \param[in] index         Index of source layer to configure. Must be in
 *                          range [0, 4].
 * \param[in] lumaX,lumaY   Pointers to NvMedia2DFilterCoefficients10Tap.
 *                          These configure to the luma component for YUV
 *                          formats, or all the components for RGB formats.
 *                          There is one pointer for the horizontal direction,
 *                          and one pointer for the vertical direction.
 * \param[in] chromaX,chromaY  Pointers to NvMedia2DFilterCoefficients10Tap.
 *                             These configure the chroma component for YUV
 *                             formats. There is one pointer for the
 *                             horizontal direction, and one pointer for the
 *                             vertical direction.
 *
 * \return An NvMediaStatus return code.
 * - \ref NVMEDIA_STATUS_OK Filter buffer was updated successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER One of the parameters has an invalid
 *                                     value, either:
 *                                     - \a handle is NULL
 *                                     - \a filterBuffer is invalid
 *                                     - \a index is out of range
 *                                     - \a lumaX is NULL
 *                                     - \a lumaY is NULL
 *                                     - \a chromaX is NULL
 *                                     - \a chromaY is NULL
 *
 * \pre \a handle must be valid NvMedia2D handle created with NvMedia2DCreate().
 * \pre \a filterBuffer must be valid NvMedia2DFilterBuffer handle created
 *      with NvMedia2DCreateFilterBuffer().
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions: Each thread uses
 *     different NvMedia2D handle
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: Yes
 *   - De-Init: No
 */
NvMediaStatus
NvMedia2DComputeFilterCoefficients10Tap(NvMedia2D const* const handle,
  NvMedia2DFilterBuffer const filterBuffer,
  uint32_t const index,
  NvMedia2DFilterCoefficients10Tap const* const lumaX,
  NvMedia2DFilterCoefficients10Tap const* const lumaY,
  NvMedia2DFilterCoefficients10Tap const* const chromaX,
  NvMedia2DFilterCoefficients10Tap const* const chromaY);


//
// Version History
//
// Version 1.1 February 1, 2016
// - Initial release
//
// Version 1.2 May 11, 2016
// - Added #NvMedia2DCheckVersion API
//
// Version 1.3 May 5, 2017
// - Removed compositing, blending and alpha related defines and structures
//
// Version 2.0 May 11, 2017
// - Deprecated NvMedia2DBlit API
// - Deprecated NvMedia2DCheckVersion API
// - Deprecated NvMedia2DColorStandard, NvMedia2DColorRange and
//   NvMedia2DColorMatrix types
// - Added #NvMedia2DGetVersion API
//
// Version 2.1 May 17, 2017
// - Moved transformation to nvmedia_common.h
// - Renamed NvMedia2DTransform to #NvMediaTransform
//
// Version 2.2 September 4, 2018
// - Added deprecated warning message for #NvMedia2DCopyPlane,
//   NvMedia2DWeave
// - Added APIs #NvMedia2DCopyPlaneNew, #NvMedia2DWeaveNew
//
// Version 3.0 October 30, 2018
// - Deprecated #NvMedia2DCopyPlane API
// - Deprecated #NvMedia2DWeave API
//
// Version 3.1 December 11, 2018
// - Fixed MISRA-C Rule 21.1 and 21.2 Violations
//
// Version 3.2 January 21, 2019
// - Moved #NvMediaTransform from nvmedia_common.h to this header
//
// Version 3.3 Feb 21, 2019
// - Changed #NvMedia2D type from void to struct
//
// Version 3.4 March 5, 2019
// - Fixed MISRA-C Rule 8.13 Violations
//
// Version 3.5 March 14, 2019
// - Removing NvMedia2DBlitFlags enum definition
// - updated #NvMedia2DBlitParametersOut structure definition
//
// Version 3.6 March 18, 2019
// - Added APIs #NvMedia2DImageRegister, #NvMedia2DImageUnRegister
//
// Version 3.7 March 22, 2019
// - Unnecessary header include nvmedia_common.h has been removed
//
// Version 3.8 May 18, 2020
// - Changes related to MISRA-C Rule 8.13 Violations fixes.
//
// Version 3.9 Nov 12, 2020
// - Improved comments and documentation
// - Introduce NvMedia2DDestroyEx, which returns an error unlike NvMedia2DDestroy
// - NvMedia2DDestroy is marked as deprecated
// - NVMEDIA_STATUS_UNDEFINED_STATE is returned
//   instead of NVMEDIA_STATUS_BAD_PARAMETER if error happens
//   after submit is started
//
// Version 3.10 January 25, 2021
// - Remove NvMedia2DWeaveNew API.
//
// Version 4.0 September 23, 2021
// - Remove NvMedia2DCopyPlaneNew API.
// - Remove NvMedia2DBlitEx API.
// - Remove NvMedia2DImageRegister API.
// - Remove NvMedia2DImageUnregister API.
// - Remove NvMedia2DDestroyEx API.
// - Remove NvMedia2DNvSciSyncGetVersion API.
// - Remove NVMEDIA_2D_NVSCISYNC_VERSION_MAJOR token.
// - Remove NVMEDIA_2D_NVSCISYNC_VERSION_MINOR token.
// - Change prototype for NvMedia2DCreate API.
// - Change prototype for NvMedia2DDestroy API.
// - Change prototype for NvMedia2DSetNvSciSyncObjforEOF API.
// - Change prototype for NvMedia2DInsertPreNvSciSyncFence API.
// - Change prototype for NvMedia2DGetEOFNvSciSyncFence API.
// - Rename NvMedia2DStretchFilter to NvMedia2DFilter.
// - Rename NvMediaTransform to NvMedia2DTransform.
// - Add NVMEDIA_2D_VERSION_PATCH token.
// - Add NvMedia2DCompose API.
// - Add NvMedia2DFillNvSciBufAttrList API.
// - Add NvMedia2DRegisterNvSciBufObj API.
// - Add NvMedia2DUnregisterNvSciBufObj API.
// - Add NvMedia2DSetSrcNvSciBufObj API.
// - Add NvMedia2DSetDstNvSciBufObj API.
// - Set default filter mode to NVMEDIA_2D_FILTER_OFF.
//
// Version 4.1 November 15, 2021
// - Add NvMedia2DSetSrcBlendMode API.
//
// Version 4.2 November 29, 2021
// - Add refcounting to NvMedia2DRegisterNvSciBufObj/UnregisterNvSciBufObj API.
//
// Version 4.3 March 8, 2022
// - Add NvMedia2DCreateFilterBuffer API.
// - Add NvMedia2DDestroyFilterBuffer API.
// - Add NvMedia2DSetFilterBuffer API.
// - Add NvMedia2DComputeFilterCoefficients5Tap API.
// - Add NvMedia2DComputeFilterCoefficients10Tap API.
//
// Version 5.0.0 March 28, 2022
// - Add support for NvSciSync task statuses
// - Max pre-fence count changed from 32 to 16
//
// Version 6.0.0 June 3, 2022
// - Change default for maxRegisteredBuffers attribute from 256 to 64
// - Forbid registering same buffer multiple times
// - Error codes changed for multiple APIs
//
// Version 7.0.0 July 8, 2022
// - New error NVMEDIA_STATUS_INVALID_STATE added for multiple APIs
//
// Version 7.0.1 August 25, 2022
// - Allow NULL context handle in NvMedia2DFillNvSciBufAttrList
//
// Version 7.0.2 September 2, 2022
// - Always treat compose parameters and filter buffer handle value 0 as invalid
//
// Version 8.0.0 October 17, 2023
// - Update the logic to compute the filter buffer coefficients values
//

#ifdef __cplusplus
}
#endif

#endif // NVMEDIA_2D_H
