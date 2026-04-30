/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__ide_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: The NvMedia Decode Processing API
 *
 * This file contains the Decode Processing API.
 */

#ifndef NVMEDIA_IDE_H
#define NVMEDIA_IDE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_common_decode.h"
#include "nvmedia_core.h"
#include "nvscibuf.h"
#include "nvscisync.h"

/** \brief Major Version number. */
#define NVMEDIA_IDE_VERSION_MAJOR   1

/** \brief Minor Version number. */
#define NVMEDIA_IDE_VERSION_MINOR   0

/** \brief Patch Version number. */
#define NVMEDIA_IDE_VERSION_PATCH   0

/**
 * \brief Specifies the maximum number of times
 * NvMediaIDEInsertPreNvSciSyncFence() can be called before each call to
 * NvMediaIDEFeedFrame().
 */
#define NVMEDIA_IDE_MAX_PRENVSCISYNCFENCES  (16U)



/**
 * \brief An opaque NvMediaIDE object created by NvMediaIDECreate.
 */
typedef struct NvMediaIDE NvMediaIDE;

/** \brief Defines 10-bit decode. */
#define NVMEDIA_IDE_10BIT_DECODE (1U<<0)

/** \brief Rec_2020 color format for the decoded surface. */
#define NVMEDIA_IDE_PIXEL_REC_2020 (1U<<1)

/** \brief Use 16 bit surfaces if contents is higher than 8 bit. */
#define NVMEDIA_IDE_OUTPUT_16BIT_SURFACES (1U<<2)

/** \brief Create decoder for encrypted content decoding. */
#define NVMEDIA_IDE_ENABLE_AES  (1U<<3)

/** \brief Create decoder to output in NV24 format. */
#define NVMEDIA_IDE_NV24_OUTPUT (1U<<4)

/** \brief Enable decoder profiling support. */
#define NVMEDIA_IDE_PROFILING   (1U<<5)

/** \brief Enable decoder motion vector dump. */
#define NVMEDIA_IDE_DUMP_MV     (1U<<6)

/**
 * \brief Retrieves the version information for the NvMediaIDE library.
 *
 * \pre None
 * \post None
 *
 * <b>Usage considerations</b>
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
 *
 * \param[in] version A pointer to a NvMediaVersion structure of the client.
 *
 * \return NvMediaStatus The status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK
 * - NVMEDIA_STATUS_BAD_PARAMETER if the pointer is invalid.
 */
NvMediaStatus
NvMediaIDEGetVersion(
    NvMediaVersion *version
);

/**
 * \brief Creates an NvMediaIDE object for the specified codec. Each
 * decoder object may be accessed by a separate thread.
 *
 * \pre NvMediaIDEGetVersion()
 * \post NvMediaIDE object is created.
 *
 * <b>Usage considerations</b>
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
 *
 * \param[in] codec Codec type. The following types are supported:
 *   - NVMEDIA_VIDEO_CODEC_HEVC
 *   - NVMEDIA_VIDEO_CODEC_H264
 *   - NVMEDIA_VIDEO_CODEC_VC1
 *   - NVMEDIA_VIDEO_CODEC_VC1_ADVANCED
 *   - NVMEDIA_VIDEO_CODEC_MPEG1
 *   - NVMEDIA_VIDEO_CODEC_MPEG2
 *   - NVMEDIA_VIDEO_CODEC_MPEG4
 *   - NVMEDIA_VIDEO_CODEC_MJPEG
 *   - NVMEDIA_VIDEO_CODEC_VP8
 *   - NVMEDIA_VIDEO_CODEC_VP9
 *   - NVMEDIA_VIDEO_CODEC_AV1 [Supported only on T234 and further chips]
 * \param[in] width Decoder width in luminance pixels.
 * \param[in] height Decoder height in luminance pixels.
 * \param[in] maxReferences The maximum number of reference frames used.
 *   This limits internal allocations.
 * \param[in] maxBitstreamSize The maximum size for bitstream.
 *   This limits internal allocations.
 * \param[in] inputBuffering How many frames can be in flight at any given
 *   time. If this value is 1, NvMediaIDERender() blocks until the
 *   previous frame has finished decoding. If this is 2, NvMediaIDERender
 *   may block until the frame before the previous frame has finished
 *   decoding. Note that the decoder may limit the actual number of frames
 *   in flight to a smaller number than this value. The maximum number is 8.
 * \param[in] flags Set the flags of the decoder. The following flags are
 *   supported:
 *   - NVMEDIA_IDE_10BIT_DECODE
 * \param[in] instanceId The ID of the engine instance.
 *   The following instances are supported:
 *   - NVMEDIA_DECODER_INSTANCE_0
 *   - NVMEDIA_DECODER_INSTANCE_1
 *   - NVMEDIA_DECODER_INSTANCE_AUTO
 *
 * \return NvMediaIDE The created NvMediaIDE handle or NULL if unsuccessful.
 */
NvMediaIDE *
NvMediaIDECreate(
    NvMediaVideoCodec codec,
    uint16_t width,
    uint16_t height,
    uint16_t maxReferences,
    uint64_t maxBitstreamSize,
    uint8_t inputBuffering,
    uint32_t flags,
    NvMediaDecoderInstanceId instanceId
);

/**
 * \brief Create an NvMediaIDE object instance.
 *
 * \pre NvMediaIDEGetVersion()
 * \pre NvMediaIDENvSciSyncGetVersion() [for use with IMGDEC-NvSciSync APIs]
 * \post NvMediaIDE object is created
 *
 * <b>Usage considerations</b>
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
 *
 * \return The created NvMediaIDE handle or NULL if unsuccessful.
 */
NvMediaIDE *
NvMediaIDECreateCtx(
    void
);

/**
 * \brief Initialize an NvMediaIDE object instance.
 *
 * \pre NvMediaIDECreateCtx()
 * \post NvMediaIDE object is initialized.
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] decoder The decoder to use.
 * \param[in] codec Codec type. The following types are supported:
 *   - NVMEDIA_VIDEO_CODEC_HEVC
 *   - NVMEDIA_VIDEO_CODEC_H264
 *   - NVMEDIA_VIDEO_CODEC_VC1
 *   - NVMEDIA_VIDEO_CODEC_VC1_ADVANCED
 *   - NVMEDIA_VIDEO_CODEC_MPEG1
 *   - NVMEDIA_VIDEO_CODEC_MPEG2
 *   - NVMEDIA_VIDEO_CODEC_MPEG4
 *   - NVMEDIA_VIDEO_CODEC_MJPEG
 *   - NVMEDIA_VIDEO_CODEC_VP8
 *   - NVMEDIA_VIDEO_CODEC_VP9
 *   - NVMEDIA_VIDEO_CODEC_AV1 [Supported only on T234 and further chips]
 * \param[in] width Decoder width in luminance pixels.
 * \param[in] height Decoder height in luminance pixels.
 * \param[in] maxReferences The maximum number of reference frames used.
 *   This limits internal allocations.
 * \param[in] maxBitstreamSize The maximum size for bitstream.
 *   This limits internal allocations.
 * \param[in] inputBuffering How many frames can be in flight at any given
 *   time. If this value is 1, NvMediaIDERender() blocks until the
 *   previous frame has finished decoding. If this is 2, NvMediaIDERender
 *   may block until the frame before the previous frame has finished
 *   decoding. Note that the decoder may limit the actual number of frames
 *   in flight to a smaller number than this value. The maximum number is 8.
 * \param[in] flags Set the flags of the decoder. The following flags are
 *   supported:
 *   - NVMEDIA_IDE_10BIT_DECODE
 * \param[in] instanceId The ID of the engine instance.
 *   The following instances are supported:
 *   - NVMEDIA_DECODER_INSTANCE_0
 *   - NVMEDIA_DECODER_INSTANCE_1
 *   - NVMEDIA_DECODER_INSTANCE_AUTO
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK
 * - NVMEDIA_STATUS_BAD_PARAMETER if input parameters are invalid.
 * - NVMEDIA_STATUS_ERROR if called after decoder initialization.
 */
NvMediaStatus
NvMediaIDEInit(
    NvMediaIDE *decoder,
    NvMediaVideoCodec codec,
    uint16_t width,
    uint16_t height,
    uint16_t maxReferences,
    uint64_t maxBitstreamSize,
    uint8_t inputBuffering,
    uint32_t flags,
    NvMediaDecoderInstanceId instanceId
);

/**
 * \brief Destroys an NvMediaIDE object.
 *
 * \pre NvMediaIDEUnregisterNvSciBufObj()
 * \pre NvMediaIDEUnregisterNvSciSyncObj()
 * \post NvMediaIDE object is destroyed
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \param[in] decoder The decoder to be destroyed.
 *
 * \return NvMediaStatus The completion status of the operation.
 */
NvMediaStatus
NvMediaIDEDestroy(
   const NvMediaIDE *decoder
);

/**
 * \brief Registers an NvSciBufObj for use with an NvMediaIde handle.
 * The NvMediaIde handle maintains a record of all the objects registered
 * using this API.
 *
 * Maximum of 192 NvSciBufObj handles can be registered.
 *
 * \pre NvMediaIDEInit()
 * \pre NvMediaIDERegisterNvSciSyncObj()
 * \post NvSciBufObj is registered with NvMediaIde object
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIde object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] decoder NvMediaIde handle.
 *   Input range: Non-NULL - valid pointer address
 * \param[in] bufObj  An NvSciBufObj object.
 *   Input range: A valid NvSciBufObj
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if decoder, bufObj or accessMode is invalid.
 * - NVMEDIA_STATUS_ERROR in following cases
 *          - if total number of registered NvSciBufObj handles exceed 192.
 *          - if access mode in NvSciBufObj is set to access more than once.
 *          - if NvSciBufObj is already registered.
 */
NvMediaStatus
NvMediaIDERegisterNvSciBufObj (
    NvMediaIDE         *decoder,
    NvSciBufObj        bufObj
);

/**
 * \brief Un-registers NvSciBufObj which was previously registered with
 * NvMediaIde using NvMediaIDERegisterNvSciBufObj().
 *
 * For all NvSciBufObj handles registered with NvMediaIde,
 * NvMediaIDEUnregisterNvSciBufObj must be called before calling
 * NvMediaIDEDestroy(). NvMediaIde will remove corresponding record of all the
 * objects from its internal data structures and frees up corresponding internal
 * resources.
 *
 * \pre NvMediaIDEUnregisterNvSciSyncObj() [verify that processing is complete]
 * \post NvSciBufObj is un-registered from NvMediaIde object
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIde object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \param[in] decoder NvMediaIde handle.
 *   Input range: Non-NULL - valid pointer address
 * \param[in] bufObj  An NvSciBufObj object.
 *   Input range: A valid NvSciBufObj
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if decoder or bufObj is invalid.
 * - NVMEDIA_STATUS_ERROR in following cases
 *          - if NvSciBufObj is not previously registered with NvMediaIde.
 *          - if NvSciBufObj is unregistered multiple times.
 */
NvMediaStatus
NvMediaIDEUnregisterNvSciBufObj (
    const NvMediaIDE *decoder,
    NvSciBufObj       bufObj
);

/**
 * \brief Decodes a compressed field/frame and render the result into a
 * NvSciBufObj target.
 *
 * \pre NvMediaIDERegisterNvSciSyncObj()
 * \pre NvMediaIDESetNvSciSyncObjforEOF()
 * \pre NvMediaIDEInsertPreNvSciSyncFence()
 * \post Decoding task is submitted
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] decoder The decoder object that will perform the
 *   decode operation.
 * \param[in] target  The NvSciBufObj that will contain the decoded content,
 *   allocated with a call to NvSciBufObjAlloc.
 * \param[in] pictureInfo A (pointer to a) structure containing
 *   information about the picture to be decoded. Note that the appropriate
 *   type of NvMediaPictureInfo* structure must be provided to match to
 *   profile that the decoder was created for.
 * \param[in] encryptParams A (pointer to a) structure containing
 *   information about encryption parameter used to decrypt the video
 *   content on the fly.
 * \param[in] numBitstreamBuffers The number of bitstream
 *   buffers containing compressed data.
 * \param[in] bitstreams An array of bitstream buffers.
 * \param[out] FrameStatsDump A (pointer to a) structure containing frame
 *   coding specific informations such as frame type, motion vector dumps,
 *   etc. when the corresponding flag is enabled in NvMediaIDECreate().
 * \param[in] instanceId The ID of the engine instance.
 *   The following instances are supported if NVMEDIA_DECODER_INSTANCE_AUTO
 *   was used in NvMediaIDECreate API:
 *   - NVMEDIA_DECODER_INSTANCE_0
 *   - NVMEDIA_DECODER_INSTANCE_1
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters is invalid.
 */
NvMediaStatus
NvMediaIDEDecoderRender(
    const NvMediaIDE *decoder,
    NvSciBufObj target,
    const NvMediaPictureInfo *pictureInfo,
    const void *encryptParams,
    uint32_t numBitstreamBuffers,
    const NvMediaBitstreamBuffer *bitstreams,
    NvMediaIDEFrameStats *FrameStatsDump,
    NvMediaDecoderInstanceId instanceId
);

/**
 * \brief This function is intended for use in low-latency decode mode.
 * It is implemented only for H265 decoder. Error will be returned if it is
 * called for any other codec.
 *
 * Each set of buffers should contain exactly 1 slice data. For first slice of
 * every frame, NvMediaIDERender() function should be called. NvMediaIDESliceDecode()
 * should be called for the second slice onwards. Also, the picture
 * information passed to the NvMediaIDERender() function should have
 * sliceDecode.enable = 1 set for low-latency mode.
 *
 * \pre NvMediaIDERegisterNvSciSyncObj()
 * \pre NvMediaIDESetNvSciSyncObjforEOF()
 * \pre NvMediaIDEInsertPreNvSciSyncFence()
 * \post Decoding task is submitted
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Async
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] decoder      The decoder object that will perform the
 *   decode operation.
 * \param[in] target       The target NvSciBufObj that contains the decoded
 *   content, allocated with a call to NvSciBufObjAlloc.
 * \param[in] sliceDecData SliceDecode data info.
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters is invalid.
 */
NvMediaStatus
NvMediaIDESliceDecode (
    const NvMediaIDE *decoder,
    const NvSciBufObj target,
    const NvMediaSliceDecodeData *sliceDecData
);

/**
 * \brief Retrieves the HW decode status available. This function should be
 * called in decode order once decode is complete for target surface. This can
 * be called from separate thread in decode order before the same index is
 * going to be used. Wait on NvSciSyncFence for the corresponding decode is
 * needed before getting the status from this function call.
 *
 * \pre NvMediaIDERegisterNvSciSyncObj()
 * \pre NvMediaIDESetNvSciSyncObjforEOF()
 * \pre NvMediaIDEInsertPreNvSciSyncFence()
 * \post Decoding task status is returned
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] decoder      The decoder object that will perform the
 *   decode operation.
 * \param[in] ringEntryIdx This is decoder order index for the decode
 *   operation.
 * \param[in] FrameStatus  A pointer to NvMediaIDEFrameStatus structure which
 *   will store current decoded frame status.
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameters is invalid.
 */
NvMediaStatus
NvMediaIDEGetFrameDecodeStatus(
    const NvMediaIDE *decoder,
    uint32_t ringEntryIdx,
    NvMediaIDEFrameStatus *FrameStatus
);

/**
 * \brief Fills the NvMediaIDE specific NvSciBuf attributes which than can be
 * used to allocate an NvSciBufObj that NvMediaIDE can consume.
 *
 * \pre NvMediaIDEGetVersion()
 * \post NvSciBufAttrList populated with NvMediaIDE specific NvSciBuf
 *       attributes
 *
 * <b>Usage considerations</b>
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
 *
 * \param[in]  instanceId The ID of the engine instance.
 *   The following instances are supported:
 *   - NVMEDIA_DECODER_INSTANCE_0
 *   - NVMEDIA_DECODER_INSTANCE_1
 *   - NVMEDIA_DECODER_INSTANCE_AUTO
 * \param[out] attrlist   A pointer to an NvSciBufAttrList where NvMediaIDE
 *   places NvSciBuf attributes.
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if attrlist is NULL.
 */
NvMediaStatus
NvMediaIDEFillNvSciBufAttrList(
    NvMediaDecoderInstanceId  instanceId,
    NvSciBufAttrList          attrlist
);

/**
 * \brief Fills the NvMediaIDE specific NvSciSync attributes.
 *
 * This function assumes that attrlist is a valid NvSciSyncAttrList.
 *
 * This function sets the public attributes:
 * - NvSciSyncAttrKey_RequiredPerm
 *
 * The application must not set this attribute.
 *
 * \pre NvMediaIDECreate()
 * \post NvSciSyncAttrList populated with NvMediaIDE specific NvSciSync
 *        attributes
 *
 * <b>Usage considerations</b>
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
 *
 * \param[in]  decoder    A pointer to the NvMediaIDE object.
 *   Input range: Non-NULL - valid pointer address
 * \param[out] attrlist   A pointer to an NvSciSyncAttrList structure where
 *   NvMedia places NvSciSync attributes.
 * \param[in]  clienttype Indicates whether the NvSciSyncAttrList requested
 *   for an NvMediaIDE signaler or an NvMediaIDE waiter.
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if attrlist is NULL, or any of the public
 *         attributes listed above are already set.
 * - NVMEDIA_STATUS_OUT_OF_MEMORY if there is not enough memory for the
 *         requested operation.
 */
NvMediaStatus
NvMediaIDEFillNvSciSyncAttrList(
    const NvMediaIDE           *decoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * \brief NvMediaIDE get backward updates counters for VP9
 *  adaptive entropy contexts.
 *
 * \pre NvMediaIDECreate() and only for VP9
 * \post Updates VP9 Entropy context
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] decoder     A pointer to the decoder object that will perform the
 *                        decoding operation.
 * \param[in] backupdates A pointer to a structure that holds the backward
 *                        update counters.
 *
 * \return NvMediaStatus The completion status of the operation.
 */
NvMediaStatus
NvMediaIDEGetBackwardUpdates(
    const NvMediaIDE *decoder,
    void *backupdates
);

/**
 * \brief Register an NvSciSyncObj with NvMediaIDE.
 *
 * Every NvSciSyncObj(even duplicate objects) used by NvMediaIDE
 * must be registered by a call to this function before it is used.
 * Only the exact same registered NvSciSyncObj can be passed to
 * NvMediaIDESetNvSciSyncObjforEOF(), NvMediaIDEGetEOFNvSciSyncFence(), or
 * NvMediaIDEUnregisterNvSciSyncObj().
 *
 * For a given NvMediaIDE handle,
 * one NvSciSyncObj can be registered as one NvMediaNvSciSyncObjType only.
 * For each NvMediaNvSciSyncObjType, a maximum of 16 NvSciSyncObjs can
 * be registered.
 *
 * \pre NvMediaIDEFillNvSciSyncAttrList()
 * \post NvSciSyncObj registered with NvMediaIDE
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] decoder     A pointer to the NvMediaIDE object.
 *   Input range: Non-NULL - valid pointer address
 * \param[in] syncobjtype Determines how @a nvscisync is used by @a decoder.
 *   Input range: Entries in NvMediaNvSciSyncObjType enumeration
 * \param[in] nvscisync   The NvSciSyncObj to be registered with @a decoder.
 *   Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is NULL or
 *         @a syncobjtype is not a valid NvMediaNvSciSyncObjType.
 * - NVMEDIA_STATUS_NOT_SUPPORTED if @a nvscisync is not a
 *         compatible NvSciSyncObj which NvMediaIDE can support.
 * - NVMEDIA_STATUS_ERROR if the maximum number of NvSciSyncObjs
 *         are already registered for the given @a syncobjtype, or
 *         if @a nvscisync is already registered with the same @a decoder
 *         handle for a different @a syncobjtype.
 */
NvMediaStatus
NvMediaIDERegisterNvSciSyncObj(
    const NvMediaIDE           *decoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * \brief Unregisters an NvSciSyncObj with NvMediaIDE.
 *
 * Every NvSciSyncObj registered with NvMediaIDE by
 * NvMediaIDERegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIDEUnregisterNvSciBufObj() to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * NvMediaIDERender() operation that uses the NvSciSyncObj has completed.
 * If this function is called while NvSciSyncObj is still in use by any
 * NvMediaIDERender() operation, the API returns NVMEDIA_STATUS_PENDING to
 * indicate the same. NvSciSyncFenceWait() API can be called on the EOF
 * NvSciSyncFence obtained post the last call to NvMediaIDERender() to wait
 * for the associated tasks to complete. The EOF NvSciSyncFence would have
 * been previously obtained via a call to NvMediaIDEGetEOFNvSciSyncFence().
 *
 * \pre NvMediaIDERender()
 * \pre NvMediaIDEGetBits() or NvSciSyncFenceWait() [verify that processing
 *                                                   is complete]
 * \post NvSciSyncObj un-registered with NvMediaIDE
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \param[in] decoder   A pointer to the NvMediaIDE object.
 *   Input range: Non-NULL - valid pointer address
 * \param[in] nvscisync An NvSciSyncObj to be unregistered with @a decoder.
 *   Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if decoder is NULL, or
 *         @a nvscisync is not registered with @a decoder.
 * - NVMEDIA_STATUS_PENDING if the @a nvscisync is still in use, i.e.
 *         the submitted task is still in progress. In this case, the
 *         application can choose to wait for operations to complete on the
 *         output surface using NvSciSyncFenceWait() or re-try the
 *         NvMediaIDEUnregisterNvSciBufObj() API call, until the status
 *         returned is not NVMEDIA_STATUS_PENDING.
 * - NVMEDIA_STATUS_ERROR if @a decoder was destroyed before this function
 *         was called.
 */
NvMediaStatus
NvMediaIDEUnregisterNvSciSyncObj(
    const NvMediaIDE  *decoder,
    NvSciSyncObj      nvscisync
);

/**
 * \brief Specifies the NvSciSyncObj to be used for an EOF
 * NvSciSyncFence.
 *
 * To use NvMediaIDEGetEOFNvSciSyncFence(), the application must call
 * NvMediaIDESetNvSciSyncObjforEOF() before it calls NvMediaIDERender().
 *
 * NvMediaIDESetNvSciSyncObjforEOF() currently may be called only once before
 * each call to NvMediaIDERender(). The application may choose to call this
 * function only once before the first call to NvMediaIDERender().
 *
 * \pre NvMediaIDERegisterNvSciSyncObj()
 * \post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] decoder      A pointer to the NvMediaIDE object.
 *   Input range: Non-NULL - valid pointer address
 * \param[in] nvscisyncEOF A registered NvSciSyncObj which is to be
 *                         associated with EOF NvSciSyncFence.
 *   Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is NULL, or if @a nvscisyncEOF
 *         is not registered with @a decoder as either type
 *         @ref NVMEDIA_EOFSYNCOBJ or type @ref NVMEDIA_EOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIDESetNvSciSyncObjforEOF(
    const NvMediaIDE      *decoder,
    NvSciSyncObj          nvscisyncEOF
);

/**
 * \brief Sets an NvSciSyncFence as a prefence for an
 * NvMediaIDERender() NvSciSyncFence operation.
 *
 * You must call NvMediaIDEInsertPreNvSciSyncFence() before you call
 * NvMediaIDERender(). The NvMediaIDERender() operation is started only
 * after the expiry of the @a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIDEInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIDERender(handle, srcsurf, srcrect, dstrect, picparams, instanceid);
 * \endcode
 * the @a NvMediaIDERender() operation is assured to start only after the
 * expiry of @a prenvscisyncfence.
 *
 * You can set a maximum of @ref NVMEDIA_IDE_MAX_PRENVSCISYNCFENCES prefences
 * by calling NvMediaIDEInsertPreNvSciSyncFence() before NvMediaIDERender().
 * After the call to NvMediaIDERender(), all NvSciSyncFences previously
 * inserted by NvMediaIDEInsertPreNvSciSyncFence() are removed, and they are not
 * reused for the subsequent NvMediaIDERender() calls.
 *
 * \pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * \post Pre-NvSciSync fence is set
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] decoder           A pointer to the NvMediaIDE object.
 *   Input range: Non-NULL - valid pointer address
 * \param[in] prenvscisyncfence A pointer to NvSciSyncFence.
 *   Input range: Non-NULL - valid pointer address
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is not a valid NvMediaIDE
 *     handle, or if @a prenvscisyncfence is NULL, or if @a prenvscisyncfence
 *     was not generated with an NvSciSyncObj that was registered with
 *     @a decoder as either @ref NVMEDIA_PRESYNCOBJ or
 *     @ref NVMEDIA_EOF_PRESYNCOBJ type.
 * - NVMEDIA_STATUS_NOT_SUPPORTED if NvMediaIDEInsertPreNvSciSyncFence()
 *     has already been called at least %NVMEDIA_IDE_MAX_PRENVSCISYNCFENCES
 *     times with the same @a decoder handle before an NvMediaIDERender()
 *     call.
 */
NvMediaStatus
NvMediaIDEInsertPreNvSciSyncFence(
    const NvMediaIDE         *decoder,
    const NvSciSyncFence     *prenvscisyncfence
);

/**
 * \brief Gets EOF NvSciSyncFence for an NvMediaIDERender() operation.
 *
 * The EOF NvSciSyncFence associated with an NvMediaIDERender() operation
 * is an NvSciSyncFence. Its expiry indicates that the corresponding
 * NvMediaIDERender() operation has finished.
 *
 * NvMediaIDEGetEOFNvSciSyncFence() returns the EOF NvSciSyncFence associated
 * with the last NvMediaIDERender() call. NvMediaIDEGetEOFNvSciSyncFence() must
 * be called after an NvMediaIDERender() call.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIDERender(handle, srcsurf, srcrect, dstrect, picparams, instanceid);
 * nvmstatus = NvMediaIDEGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * \endcode
 * expiry of @a eofnvscisyncfence indicates that the preceding
 * NvMediaIDERender() operation has finished.
 *
 * \pre NvMediaIDESetNvSciSyncObjforEOF()
 * \pre NvMediaIDERender()
 * \post EOF NvSciSync fence for a submitted task is obtained
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIDE object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in]  decoder            A pointer to the NvMediaIDE object.
 *   Input range: Non-NULL - valid pointer address
 * \param[in]  eofnvscisyncobj    An EOF NvSciSyncObj associated with the
 *                                NvSciSyncFence which is being requested.
 *   Input range: A valid NvSciSyncObj
 * \param[out] eofnvscisyncfence  A pointer to the EOF NvSciSyncFence.
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if @a decoder is not a valid NvMediaIDE
 *         handle, @a eofnvscisyncfence is NULL, or @a eofnvscisyncobj is not
 *         registered with @a decoder as type @ref NVMEDIA_EOFSYNCOBJ or
 *         @ref NVMEDIA_EOF_PRESYNCOBJ.
 * - NVMEDIA_STATUS_ERROR if the function was called before
 *         NvMediaIDERender() was called.
 */
NvMediaStatus
NvMediaIDEGetEOFNvSciSyncFence(
    const NvMediaIDE        *decoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/*
 * @defgroup 6x_history_nvmedia_ide History
 * Provides change history for the NvMediaIDE API.
 *
 * \section 6x_history_nvmedia_ide Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 */
#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IDE_H */
