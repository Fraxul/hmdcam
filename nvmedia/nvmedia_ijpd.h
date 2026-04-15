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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/group__x__image__jpeg__decode__api.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \defgroup x_image_jpeg_decode_api Image JPEG Decoder
 * \ingroup x_nvmedia_image_top
 *
 * The NvMediaIJPD object takes a JPEG bitstream and decompress it to
 * image data.
 *
 * @{
 */

#ifndef NVMEDIA_IJPD_H
#define NVMEDIA_IJPD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_core.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvmedia_common_encode_decode.h"

/** \brief Major version number. */
#define NVMEDIA_IJPD_VERSION_MAJOR   1

/** \brief Minor version number. */
#define NVMEDIA_IJPD_VERSION_MINOR   0

/** \brief Patch version number. */
#define NVMEDIA_IJPD_VERSION_PATCH   0

/**
 * \brief Specifies the maximum number of times
 * NvMediaIJPDInsertPreNvSciSyncFence() can be called before each call to
 * NvMediaIJPDFeedFrame().
 */
#define NVMEDIA_IJPD_MAX_PRENVSCISYNCFENCES        (16U)

/** \brief JPEG decode set alpha. */
#define NVMEDIA_JPEG_DEC_ATTRIBUTE_ALPHA_VALUE     (1 << 0)

/** \brief JPEG decode set color standard. */
#define NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD  (1 << 1)

/** \brief JPEG decode render flag rotate 0. */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_0          0

/** \brief JPEG decode render flag rotate 90. */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_90         1

/** \brief JPEG decode render flag rotate 180. */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_180        2

/** \brief JPEG decode render flag rotate 270. */
#define NVMEDIA_IJPD_RENDER_FLAG_ROTATE_270        3

/** \brief JPEG decode render flag flip horizontal. */
#define NVMEDIA_IJPD_RENDER_FLAG_FLIP_HORIZONTAL   (1 << 2)

/** \brief JPEG decode render flag flip vertical. */
#define NVMEDIA_IJPD_RENDER_FLAG_FLIP_VERTICAL     (1 << 3)

/** \brief JPEG decode max number of app markers supported. */
#define NVMEDIA_MAX_JPEG_APP_MARKERS               16

/**
 * \brief Defines color standards.
 */
typedef enum {
    /** Specifies ITU BT.601 color standard. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601,
    /** Specifies ITU BT.709 color standard. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709,
    /** Specifies SMTE 240M color standard. */
    NVMEDIA_IJPD_COLOR_STANDARD_SMPTE_240M,
    /** Specifies ITU BT.601 color standard extended range. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_601_ER,
    /** Specifies ITU BT.709 color standard extended range. */
    NVMEDIA_IJPD_COLOR_STANDARD_ITUR_BT_709_ER
} NvMediaIJPDColorStandard;

/**
 * \brief Holds image JPEG decoder attributes.
 */
typedef struct {
    /** Color standard. */
    NvMediaIJPDColorStandard colorStandard;

    /** Alpha value. */
    uint32_t alphaValue;
} NVMEDIAJPEGDecAttributes;

/**
 * \brief Holds image JPEG decoder marker Info.
 */
typedef struct {
  /** Marker identifier. */
  uint16_t marker;
  /** Length of the marker data. */
  uint16_t len;
  /** Pointer to the marker data. */
  void    *pMarker;
} NvMediaJPEGAppMarkerInfo;

/**
 * \brief Holds image JPEG decoder stream information.
 */
typedef struct {
  /** Width of the JPEG image. */
  uint16_t width;
  /** Height of the JPEG image. */
  uint16_t height;
  /** Indicates whether partial acceleration is required. */
  uint8_t  partialAccel;
  /** Number of application markers found. */
  uint8_t  num_app_markers;
  /** Array of application marker info structures. */
  NvMediaJPEGAppMarkerInfo appMarkerInfo[NVMEDIA_MAX_JPEG_APP_MARKERS];
} NVMEDIAJPEGDecInfo;

/**
 * \brief An opaque NvMediaIJPD object created by NvMediaIJPDCreate.
 */
typedef struct NvMediaIJPD NvMediaIJPD;

/**
 * \brief Retrieves the version information for the NvMedia IJPD library.
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
NvMediaIJPDGetVersion(
    NvMediaVersion *version
);

/**
 * \brief Creates a JPEG decoder object capable of decoding a JPEG stream
 * into an image surface.
 *
 * \pre NvMediaIJPDGetVersion()
 * \pre NvMediaIJPDFillNvSciBufAttrList()
 * \post NvMediaIJPD object is created
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
 * \param[in] maxWidth The maximum width of output surface to support.
 *   You can use NvMediaIJPDResize() to enlarge this limit for an existing
 *   decoder.
 * \param[in] maxHeight The maximum height of output surface to support.
 *   You can use NvMediaIJPDResize() to enlarge this limit for an existing
 *   decoder.
 * \param[in] maxBitstreamBytes The maximum JPEG bitstream size in bytes to
 *   support. Use NvMediaIJPDResize() to enlarge this limit for an existing
 *   decoder.
 * \param[in] supportPartialAccel Indicates that the JPEG decode object
 *   supports partial acceleration.
 *   If it does, set this argument to the character '1' (true).
 *   If it does not, set this argument to the character '0' (false).
 * \param[in] instanceId The ID of the engine instance. The following
 *   instances are supported:
 *   - NVMEDIA_JPEG_INSTANCE_0
 *   - NVMEDIA_JPEG_INSTANCE_1 [Supported only on T23X]
 *   - NVMEDIA_JPEG_INSTANCE_AUTO [Supported only on T23X]
 *
 * \retval NvMediaIJPD The new image JPEG decoder handle or NULL if
 *   unsuccessful.
 */
NvMediaIJPD *
NvMediaIJPDCreate(
    uint16_t maxWidth,
    uint16_t maxHeight,
    uint32_t maxBitstreamBytes,
    bool supportPartialAccel,
    NvMediaJPEGInstanceId instanceId
);

/**
 * \brief Destroys an NvMedia image JPEG decoder.
 *
 * \pre NvMediaIJPDUnregisterNvSciBufObj()
 * \pre NvMediaIJPDUnregisterNvSciSyncObj()
 * \post NvMediaIJPD object is destroyed
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \param[in] decoder A pointer to the JPEG decoder to destroy.
 */
void NvMediaIJPDDestroy(NvMediaIJPD *decoder);

/**
 * \brief Resizes an existing image JPEG decoder.
 *
 * \pre NvMediaIJPDCreate()
 * \post NvMediaIJPD object is updated as specified
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] decoder A pointer to the JPEG decoder to use.
 * \param[in] maxWidth The new maximum width of output surface to support.
 * \param[in] maxHeight The new maximum height of output surface to support.
 * \param[in] maxBitstreamBytes The new maximum JPEG bitstream size in bytes
 *   to support.
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is invalid.
 * - NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDResize (
   NvMediaIJPD *decoder,
   uint16_t maxWidth,
   uint16_t maxHeight,
   uint32_t maxBitstreamBytes
);

/**
 * \brief Sets attributes of an existing image JPEG decoder.
 *
 * \pre NvMediaIJPDCreate()
 * \post Specified attribute is set
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] decoder A pointer to the JPEG decoder to use.
 * \param[in] attributeMask An attribute mask. Supported mask are:
 *   - NVMEDIA_JPEG_DEC_ATTRIBUTE_ALPHA_VALUE
 *   - NVMEDIA_JPEG_DEC_ATTRIBUTE_COLOR_STANDARD
 * \param[in] attributes Attributes data. Supported attribute structures:
 *   - NVMEDIAJPEGDecAttributes
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDSetAttributes(
   const NvMediaIJPD *decoder,
   uint32_t attributeMask,
   const void *attributes
);

/**
 * \brief A helper function that determines whether the JPEG decoder HW
 * engine can decode the input JPEG stream.
 *
 * Possible outcomes are:
 * - <b>Decode possible</b>. If JPEG decoder supports decode of this stream,
 *   this function returns NVMEDIA_STATUS_OK and the NVMEDIAJPEGDecInfo info
 *   will be filled out. This function also determines whether you must
 *   allocate the NvMediaIJPD object when you call NvMediaIJPDCreate().
 *   You specify that object with the NvMediaIJPDCreate() \a supportPartialAccel
 *   parameter.
 * - <b>Decode not possible</b>. If JPEG decoder cannot decode this stream,
 *   this function returns NVMEDIA_STATUS_NOT_SUPPORTED.
 *
 * \pre NvMediaIJPDCreate()
 * \post NVMEDIAJPEGDecInfo is populated with required information
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
 * \param[in,out] info A pointer to the information to be filled.
 * \param[in] numBitstreamBuffers The number of bitstream buffers.
 * \param[in] bitstreams The bitstream buffer.
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - NVMEDIA_STATUS_NOT_SUPPORTED if stream not supported
 * - NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDGetInfo (
   NVMEDIAJPEGDecInfo *info,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams
);

/**
 * \brief Decodes a JPEG image.
 *
 * The decode pipeline produces a result equivalent to the following sequence:
 * -# Decodes the full JPEG image.
 * -# Downscales the 8x8 block padded image by the \a downscaleLog2 factor.
 *    That is, a "width" by "height" JPEG is downscaled to:
 *    \code
 *    ((width + 7) & ~7) >> downscaleLog2
 *    \endcode
 *    by
 *    \code
 *    ((height + 7) & ~7) >> downscaleLog2
 *    \endcode
 * -# From the downscaled image, removes the rectangle described by
 *    \a srcRect and optionally (a) mirrors the image horizontally and/or
 *    vertically and/or (b) rotates the image.
 * -# Scales the transformed source rectangle to the \a dstRect on the
 *    output surface.
 *
 * <b>Specifying Dimensions</b>
 *
 * The JPEG decoder object must have \a maxWidth and \a maxHeight values that
 * are greater than or equal to the post-downscale JPEG image. Additionally,
 * it must have a \a maxBitstreamBytes value that is greater than or equal to
 * the total number of bytes in the bitstream buffers. You set these values
 * when you create the JPEG decoder object with NvMediaIJPDCreate().
 * Alternatively, you can use NvMediaIJPDResize() to change the dimensions
 * of an existing JPEG decoder object.
 *
 * If the JPEG decoder object has inadequate dimensions,
 * NvMediaIJPDRender() returns NVMEDIA_STATUS_INSUFFICIENT_BUFFERING.
 *
 * <b>Supporting Partial Acceleration</b>
 *
 * If the JPEG stream requires partial acceleration, created the JPEG decoder
 * object with \a supportPartialAccel set to '1'. Otherwise, the function
 * returns NVMEDIA_STATUS_BAD_PARAMETER.
 *
 * Use NvMediaIJPDGetInfo() to determine whether a stream requires
 * partialAccel.
 *
 * <b>Determining Supported JPEG Streams</b>
 *
 * If the JPEG stream is not supported, the function returns
 * NVMEDIA_STATUS_NOT_SUPPORTED.
 *
 * Use NvMediaIJPDGetInfo() to determine whether a stream is unsupported.
 *
 * \note NvMediaIJPDRender() with the NVJPG 1.0 codec does not support
 *   rotation.
 *
 * \pre NvMediaIJPDRegisterNvSciBufObj()
 * \pre NvMediaIJPDRegisterNvSciSyncObj()
 * \pre NvMediaIJPDSetNvSciSyncObjforEOF()
 * \pre NvMediaIJPDInsertPreNvSciSyncFence()
 * \post JPEG decoding task is submitted
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] decoder A pointer to the JPEG decoder to use.
 * \param[out] target NvSciBufObj that contains the decoded content,
 *   allocated with a call to NvSciBufObjAlloc. Supported surface format
 *   attributes:
 *   Buffer Type: NvSciBufType_Image
 *   Surface Type: RGBA
 *   Bit Depth: 8
 *   Layout: NvSciBufImage_PitchLinearType
 *   Scan Type: NvSciBufScan_ProgressiveType
 *   Plane base address alignment: 256
 * \param[in] srcRect The source rectangle. The rectangle from the
 *   post-downscaled image to be transformed and scaled to the \a dstRect.
 *   You can achieve horizontal and/or vertical mirroring by swapping the
 *   left-right and/or top-bottom coordinates. If NULL, the full
 *   post-downscaled surface is implied.
 * \param[in] dstRect The destination rectangle on the output surface.
 *   If NULL, a rectangle the full size of the output surface is implied.
 * \param[in] downscaleLog2 A value clamped between 0 and 3 inclusive,
 *   gives downscale factors of 1 to 8.
 * \param[in] numBitstreamBuffers The number of bitstream buffers.
 * \param[in] bitstreams The bitstream buffer. NvMediaIJPDRender() copies
 *   the data out of these buffers so the caller is free to reuse them as
 *   soon as NvMediaIJPDRender() returns.
 * \param[in] flags Flags that specify a clockwise rotation of the source
 *   in degrees and horizontal and vertical flipping. If both are specified,
 *   the image is flipped before it is rotated. You can set the \a flags
 *   argument to any one of the following:
 *   - NVMEDIA_RENDER_FLAG_ROTATE_0
 *   - NVMEDIA_RENDER_FLAG_ROTATE_90
 *   - NVMEDIA_RENDER_FLAG_ROTATE_180
 *   - NVMEDIA_RENDER_FLAG_ROTATE_270
 *   Additionally, you can use the bitwise OR operation to apply either or
 *   both of the following:
 *   - NVMEDIA_RENDER_FLAG_FLIP_HORIZONTAL
 *   - NVMEDIA_RENDER_FLAG_FLIP_VERTICAL
 * \param[in] instanceId The ID of the engine instance. The following
 *   instances are supported:
 *   - NVMEDIA_JPEG_INSTANCE_0
 *   - NVMEDIA_JPEG_INSTANCE_1 [Supported only on T23X]
 *   - NVMEDIA_JPEG_INSTANCE_AUTO [Supported only on T23X]
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDRender(
   const NvMediaIJPD *decoder,
   NvSciBufObj target,
   const NvMediaRect *srcRect,
   const NvMediaRect *dstRect,
   uint8_t downscaleLog2,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams,
   uint32_t flags,
   NvMediaJPEGInstanceId instanceId
);

/**
 * \brief Decodes a JPEG image into YUV format.
 *
 * This function is similar to NvMediaIJPDRender() except that the output
 * surface is in YUV format, not RGBA format. Also, clipping and scaling
 * (other than downscaleLog2 scaling) are not supported, so there are no
 * source or destination rectangle parameters.
 *
 * \note NvMediaIJPDRenderYUV() with the NVJPG 1.0 codec has the following
 *   limitations:
 *   - It supports chroma subsample conversion to 420 and 420H from any
 *     input format except 400.
 *   - It does not simultaneously support downscaleLog2 and subsample
 *     conversion.
 *
 * \pre NvMediaIJPDRegisterNvSciBufObj()
 * \pre NvMediaIJPDRegisterNvSciSyncObj()
 * \pre NvMediaIJPDSetNvSciSyncObjforEOF()
 * \pre NvMediaIJPDInsertPreNvSciSyncFence()
 * \post JPEG decoding task is submitted
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \param[in] decoder A pointer to the JPEG decoder to use.
 * \param[out] target NvSciBufObj that contains the decoded content,
 *   allocated with a call to NvSciBufObjAlloc. Supported surface format
 *   attributes:
 *   Buffer Type: NvSciBufType_Image
 *   Sub-sampling type: YUV420, YUV422, YUV444 (planar and semi-planar)
 *   Bit Depth: 8
 *   Layout: NvSciBufImage_PitchLinearType
 *   Scan Type: NvSciBufScan_ProgressiveType
 *   Plane base address alignment: 256
 * \param[in] downscaleLog2 A value between 0 and 3 inclusive that gives
 *   downscale factors of 1 to 8.
 * \param[in] numBitstreamBuffers The number of bitstream buffers.
 * \param[in] bitstreams The bitstream buffer. NvMediaIJPDRenderYUV() copies
 *   the data out of these buffers so the caller is free to reuse them as
 *   soon as NvMediaIJPDRenderYUV() returns.
 * \param[in] flags Flags that specify a clockwise rotation of the source
 *   in degrees and horizontal and vertical flipping. If both are specified,
 *   flipping is performed before rotating. You can set the \a flags
 *   argument to any one of the following:
 *   - NVMEDIA_RENDER_FLAG_ROTATE_0
 *   - NVMEDIA_RENDER_FLAG_ROTATE_90
 *   - NVMEDIA_RENDER_FLAG_ROTATE_180
 *   - NVMEDIA_RENDER_FLAG_ROTATE_270
 *   Additionally, you can use the bitwise OR operation to apply either or
 *   both of the following:
 *   - NVMEDIA_RENDER_FLAG_FLIP_HORIZONTAL
 *   - NVMEDIA_RENDER_FLAG_FLIP_VERTICAL
 * \param[in] instanceId The ID of the engine instance. The following
 *   instances are supported:
 *   - NVMEDIA_JPEG_INSTANCE_0
 *   - NVMEDIA_JPEG_INSTANCE_1 [Supported only on T23X]
 *   - NVMEDIA_JPEG_INSTANCE_AUTO [Supported only on T23X]
 *
 * \return NvMediaStatus The completion status of the operation.
 * Possible values are:
 * - NVMEDIA_STATUS_OK if successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if any of the input parameter is NULL.
 * - NVMEDIA_STATUS_ERROR
 */
NvMediaStatus
NvMediaIJPDRenderYUV(
   const NvMediaIJPD *decoder,
   NvSciBufObj target,
   uint8_t downscaleLog2,
   uint32_t numBitstreamBuffers,
   const NvMediaBitstreamBuffer *bitstreams,
   uint32_t flags,
   NvMediaJPEGInstanceId instanceId
);

/**
 * \brief Registers NvSciBufObj for use with a NvMediaIJPD handle.
 *
 * NvMediaIJPD handle maintains a record of all the objects registered
 * using this API and only the registered NvSciBufObj handles are accepted
 * when submitted for decoding via NvMediaIJPDRender. Even duplicated
 * NvSciBufObj objects need to be registered using this API prior.
 *
 * This needs to be used in tandem with NvMediaIJPDUnregisterNvSciBufObj().
 * The pair of APIs for registering and unregistering NvSciBufObj are
 * optional, but it is highly recommended to use them as they ensure
 * deterministic execution of NvMediaIJPDRender().
 *
 * To ensure deterministic execution time of NvMediaIJPDRender API:
 * - NvMediaIJPDRegisterNvSciBufObj must be called for every input
 *   NvSciBufObj that will be used with NvMediaIJPD
 * - All NvMediaIJPDRegisterNvSciBufObj calls must be made before first
 *   NvMediaIJPDRender API call.
 *
 * Registration of the buffer (output) is always with read-write permissions.
 *
 * Maximum of 32 NvSciBufObj handles can be registered.
 *
 * \note This API is currently not supported and can be ignored
 *
 * \pre NvMediaIJPDCreate()
 * \pre NvMediaIJPDRegisterNvSciSyncObj()
 * \post NvSciBufObj is registered with NvMediaIJPD object
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[in] bufObj NvSciBufObj object
 *   \n Input range: A valid NvSciBufObj
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_NOT_SUPPORTED if API is not functionally supported
 */
NvMediaStatus
NvMediaIJPDRegisterNvSciBufObj(
    const NvMediaIJPD   *decoder,
    NvSciBufObj         bufObj
);

/**
 * \brief Un-registers NvSciBufObj which was previously registered with
 * NvMediaIJPD using NvMediaIJPDRegisterNvSciBufObj().
 *
 * For all NvSciBufObj handles registered with NvMediaIJPD using
 * NvMediaIJPDRegisterNvSciBufObj API, NvMediaIJPDUnregisterNvSciBufObj
 * must be called before calling NvMediaIJPDDestroy API. For unregistration
 * to succeed, it should be ensured that none of the submitted tasks on the
 * bufObj are pending prior to calling NvMediaIJPDUnregisterNvSciBufObj().
 * In order to ensure this, NvMediaIJPDUnregisterNvSciSyncObj() should be
 * called prior to this API on all registered NvSciSyncObj. Post this
 * NvMediaIJPDUnregisterNvSciBufObj() can be successfully called on a valid
 * NvSciBufObj.
 *
 * This needs to be used in tandem with NvMediaIJPDRegisterNvSciBufObj().
 * The pair of APIs for registering and unregistering NvSciBufObj are
 * optional, but it is highly recommended to use them as they ensure
 * deterministic execution of NvMediaIJPDRender().
 *
 * To ensure deterministic execution time of NvMediaIJPDRender API:
 * - NvMediaIJPDUnregisterNvSciBufObj should be called only after the last
 *   NvMediaIJPDRender call
 *
 * \note This API is currently not supported and can be ignored
 *
 * \pre NvMediaIJPDUnregisterNvSciSyncObj() [verify that processing is
 *   complete]
 * \post NvSciBufObj is un-registered from NvMediaIJPD object
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[in] bufObj NvSciBufObj object
 *   \n Input range: A valid NvSciBufObj
 *
 * \return NvMediaStatus, the completion status of operation:
 * - NVMEDIA_STATUS_NOT_SUPPORTED if API is not functionally supported
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIJPDUnregisterNvSciBufObj(
    const NvMediaIJPD    *decoder,
    NvSciBufObj          bufObj
);

/**
 * \brief Fills the NvMediaIJPD specific NvSciBuf attributes which than
 * then be used to allocate an NvSciBufObj that NvMediaIJPD can consume.
 *
 * This function updates the input NvSciBufAttrList with values equivalent
 * to the following public attribute key-values:
 * NvSciBufGeneralAttrKey_PeerHwEngineArray set to
 * - NvSciBufHwEngName: NvSciBufHwEngName_NVJPG
 * - NvSciBufPlatformName: The platform this API is used on
 *
 * This function assumes that \a attrlist is a valid NvSciBufAttrList
 * created by the caller by a call to NvSciBufAttrListCreate.
 *
 * \pre NvMediaIJPDGetVersion()
 * \post NvSciBufAttrList populated with NvMediaIJPD specific NvSciBuf
 *   attributes. The caller can then set attributes specific to the type
 *   of surface, reconcile attribute lists and allocate an NvSciBufObj.
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
 * \param[in] instanceId The ID of the engine instance.
 *   \n Input range: The following instances are supported:
 *   - NVMEDIA_JPEG_INSTANCE_0
 *   - NVMEDIA_JPEG_INSTANCE_1 [Supported only on T23X]
 *   - NVMEDIA_JPEG_INSTANCE_AUTO [Supported only on T23X]
 * \param[out] attrlist An NvSciBufAttrList where NvMediaIJPD places the
 *   NvSciBuf attributes.
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a attrlist is NULL
 */
NvMediaStatus
NvMediaIJPDFillNvSciBufAttrList(
    NvMediaJPEGInstanceId     instanceId,
    NvSciBufAttrList          attrlist
);

/**
 * \brief Fills the NvMediaIJPD specific NvSciSync attributes.
 *
 * This function assumes that \a attrlist is a valid NvSciSyncAttrList.
 *
 * This function updates the input NvSciSyncAttrList with values equivalent
 * to the following public attribute key-values:
 * NvSciSyncAttrKey_RequiredPerm set to
 * - NvSciSyncAccessPerm_WaitOnly for clienttype NVMEDIA_WAITER
 * - NvSciSyncAccessPerm_SignalOnly for clienttype NVMEDIA_SIGNALER
 * - NvSciSyncAccessPerm_WaitSignal for clienttype NVMEDIA_SIGNALER_WAITER
 * NvSciSyncAttrKey_PrimitiveInfo set to
 * - NvSciSyncAttrValPrimitiveType_Syncpoint
 *
 * The application must not set these attributes in the NvSciSyncAttrList
 * passed as an input to this function.
 *
 * \pre NvMediaIJPDCreate()
 * \post NvSciSyncAttrList populated with NvMediaIJPD specific NvSciSync
 *   attributes
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *   to achieve synchronization between the engines
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[out] attrlist A pointer to an NvSciSyncAttrList structure where
 *   NvMedia places NvSciSync attributes.
 * \param[in] clienttype Indicates whether the NvSciSyncAttrList requested
 *   for an NvMediaIJPD signaler or an NvMediaIJPD waiter.
 *   \n Input range: Entries in NvMediaNvSciSyncClientType enumeration
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a attrlist is NULL, or any of the
 *   public attributes listed above are already set.
 * - NVMEDIA_STATUS_OUT_OF_MEMORY if there is not enough memory for the
 *   requested operation.
 */
NvMediaStatus
NvMediaIJPDFillNvSciSyncAttrList(
    const NvMediaIJPD           *decoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);

/**
 * \brief Registers an NvSciSyncObj with NvMediaIJPD.
 *
 * Every NvSciSyncObj (even duplicate objects) used by NvMediaIJPD must be
 * registered by a call to this function before it is used. Only the exact
 * same registered NvSciSyncObj can be passed to
 * NvMediaIJPDSetNvSciSyncObjforEOF(), NvMediaIJPDGetEOFNvSciSyncFence(),
 * or NvMediaIJPDUnregisterNvSciSyncObj().
 *
 * For a given NvMediaIJPD handle, one NvSciSyncObj can be registered as
 * one NvMediaNvSciSyncObjType only. For each NvMediaNvSciSyncObjType,
 * a maximum of 16 NvSciSyncObjs can be registered.
 *
 * \pre NvMediaIJPDFillNvSciSyncAttrList()
 * \post NvSciSyncObj registered with NvMediaIJPD
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *   to achieve synchronization between the engines
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[in] syncobjtype Determines how \a nvscisync is used by
 *   \a decoder.
 *   \n Input range: Entries in NvMediaNvSciSyncObjType enumeration
 * \param[in] nvscisync The NvSciSyncObj to be registered with \a decoder.
 *   \n Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a decoder is NULL or \a syncobjtype
 *   is not a valid NvMediaNvSciSyncObjType.
 * - NVMEDIA_STATUS_NOT_SUPPORTED if \a nvscisync is not a compatible
 *   NvSciSyncObj which NvMediaIJPD can support.
 * - NVMEDIA_STATUS_ERROR if the maximum number of NvSciScynObjs are
 *   already registered for the given \a syncobjtype, or if \a nvscisync
 *   is already registered with the same \a decoder handle for a different
 *   \a syncobjtype.
 */
NvMediaStatus
NvMediaIJPDRegisterNvSciSyncObj(
    const NvMediaIJPD           *decoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               nvscisync
);

/**
 * \brief Unregisters an NvSciSyncObj with NvMediaIJPD.
 *
 * Every NvSciSyncObj registered with NvMediaIJPD by
 * NvMediaIJPDRegisterNvSciSyncObj() must be unregistered before calling
 * NvMediaIJPDUnregisterNvSciBufObj() to unregister the NvSciBufObjs.
 *
 * Before the application calls this function, it must ensure that any
 * NvMediaIJPDRender() operation that uses the NvSciSyncObj has completed.
 * If this function is called while NvSciSyncObj is still in use by any
 * NvMediaIJPDRender() operation, the API returns NVMEDIA_STATUS_PENDING
 * to indicate the same. NvSciSyncFenceWait() API can be called on the EOF
 * NvSciSyncFence obtained post the last call to NvMediaIJPDRender() to
 * wait for the associated tasks to complete. The EOF NvSciSyncFence would
 * have been previously obtained via a call to
 * NvMediaIJPDGetEOFNvSciSyncFence(). The other option would be to call
 * NvMediaIJPDGetBits() till there is no more output to retrieve.
 *
 * \pre NvMediaIJPDRender()
 * \pre NvMediaIJPDGetBits() or NvSciSyncFenceWait() [verify that
 *   processing is complete]
 * \post NvSciSyncObj un-registered with NvMediaIJPD
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: No
 *   - De-Init: Yes
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *   to achieve synchronization between the engines
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[in] nvscisync An NvSciSyncObj to be unregistered with \a decoder.
 *   \n Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if decoder is NULL, or \a nvscisync is
 *   not registered with \a decoder.
 * - NVMEDIA_STATUS_PENDING if the NvSciSyncObj is still in use, i.e.,
 *   the submitted task is still in progress. In this case, the application
 *   can choose to wait for operations to complete on the output surface
 *   using NvSciSyncFenceWait() or re-try the
 *   NvMediaIJPDUnregisterNvSciBufObj() API call, until the status returned
 *   is not NVMEDIA_STATUS_PENDING.
 * - NVMEDIA_STATUS_ERROR if \a decoder was destroyed before this function
 *   was called.
 */
NvMediaStatus
NvMediaIJPDUnregisterNvSciSyncObj(
    const NvMediaIJPD  *decoder,
    NvSciSyncObj      nvscisync
);

/**
 * \brief Specifies the NvSciSyncObj to be used for an EOF NvSciSyncFence.
 *
 * To use NvMediaIJPDGetEOFNvSciSyncFence(), the application must call
 * NvMediaIJPDSetNvSciSyncObjforEOF() before it calls NvMediaIJPDRender().
 *
 * NvMediaIJPDSetNvSciSyncObjforEOF() currently may be called only once
 * before each call to NvMediaIJPDRender(). The application may choose to
 * call this function only once before the first call to
 * NvMediaIJPDRender().
 *
 * \pre NvMediaIJPDRegisterNvSciSyncObj()
 * \post NvSciSyncObj to be used as EOF NvSciSyncFence is set
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *   to achieve synchronization between the engines
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[in] nvscisyncEOF A registered NvSciSyncObj which is to be
 *   associated with EOF NvSciSyncFence.
 *   \n Input range: A valid NvSciSyncObj
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a decoder is NULL, or if
 *   \a nvscisyncEOF is not registered with \a decoder as either type
 *   NVMEDIA_EOFSYNCOBJ or NVMEDIA_EOF_PRESYNCOBJ.
 */
NvMediaStatus
NvMediaIJPDSetNvSciSyncObjforEOF(
    const NvMediaIJPD      *decoder,
    NvSciSyncObj          nvscisyncEOF
);

/**
 * \brief Sets an NvSciSyncFence as a prefence for an NvMediaIJPDRender()
 * NvSciSyncFence operation.
 *
 * You must call NvMediaIJPDInsertPreNvSciSyncFence() before you call
 * NvMediaIJPDRender(). The NvMediaIJPDRender() operation is started only
 * after the expiry of the \a prenvscisyncfence.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIJPDInsertPreNvSciSyncFence(handle, prenvscisyncfence);
 * nvmstatus = NvMediaIJPDRender(handle, arg2, arg3, ...);
 * \endcode
 * the NvMediaIJPDRender() operation is assured to start only after the
 * expiry of \a prenvscisyncfence.
 *
 * You can set a maximum of NVMEDIA_IJPD_MAX_PRENVSCISYNCFENCES prefences
 * by calling NvMediaIJPDInsertPreNvSciSyncFence() before
 * NvMediaIJPDRender(). After the call to NvMediaIJPDRender(), all
 * NvSciSyncFences previously inserted by
 * NvMediaIJPDInsertPreNvSciSyncFence() are removed, and they are not
 * reused for the subsequent NvMediaIJPDRender() calls.
 *
 * \pre Pre-NvSciSync fence obtained from previous engine in the pipeline
 * \post Pre-NvSciSync fence is set
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *   to achieve synchronization between the engines
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[in] prenvscisyncfence A pointer to NvSciSyncFence.
 *   \n Input range: Non-NULL - valid pointer address
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a decoder is not a valid NvMediaIJPD
 *   handle, or \a prenvscisyncfence is NULL, or if \a prenvscisyncfence
 *   was not generated with an NvSciSyncObj that was registered with
 *   \a decoder as either NVMEDIA_PRESYNCOBJ or NVMEDIA_EOF_PRESYNCOBJ
 *   type.
 * - NVMEDIA_STATUS_NOT_SUPPORTED if
 *   NvMediaIJPDInsertPreNvSciSyncFence() has already been called at least
 *   NVMEDIA_IJPD_MAX_PRENVSCISYNCFENCES times with the same \a decoder
 *   handle before an NvMediaIJPDRender() call.
 */
NvMediaStatus
NvMediaIJPDInsertPreNvSciSyncFence(
    const NvMediaIJPD         *decoder,
    const NvSciSyncFence     *prenvscisyncfence
);

/**
 * \brief Gets EOF NvSciSyncFence for an NvMediaIJPDRender() operation.
 *
 * The EOF NvSciSyncFence associated with an NvMediaIJPDRender() operation
 * is an NvSciSyncFence. Its expiry indicates that the corresponding
 * NvMediaIJPDRender() operation has finished.
 *
 * This function returns the EOF NvSciSyncFence associated with the last
 * NvMediaIJPDRender() call. NvMediaIJPDGetEOFNvSciSyncFence() must be
 * called after an NvMediaIJPDRender() call.
 *
 * For example, in this sequence of code:
 * \code
 * nvmstatus = NvMediaIJPDRender(handle, arg2, arg3, ...);
 * nvmstatus = NvMediaIJPDGetEOFNvSciSyncFence(handle, nvscisyncEOF, eofnvscisyncfence);
 * \endcode
 * expiry of \a eofnvscisyncfence indicates that the preceding
 * NvMediaIJPDRender() operation has finished.
 *
 * \pre NvMediaIJPDSetNvSciSyncObjforEOF()
 * \pre NvMediaIJPDRender()
 * \post EOF NvSciSync fence for a submitted task is obtained
 *
 * <b>Usage considerations</b>
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes, with the following conditions
 *     - Every thread should be invoked with relevant NvMediaIJPD object.
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-Init: No
 *
 * \note This API is mandatory when multiple engines are pipelined in order
 *   to achieve synchronization between the engines
 *
 * \param[in] decoder A pointer to the NvMediaIJPD object.
 *   \n Input range: Non-NULL - valid pointer address
 * \param[in] eofnvscisyncobj An EOF NvSciSyncObj associated with the
 *   NvSciSyncFence which is being requested.
 *   \n Input range: A valid NvSciSyncObj
 * \param[out] eofnvscisyncfence A pointer to the EOF NvSciSyncFence.
 *
 * \return NvMediaStatus The status of the operation. Possible values are:
 * - NVMEDIA_STATUS_OK if the function is successful.
 * - NVMEDIA_STATUS_BAD_PARAMETER if \a decoder is not a valid NvMediaIJPD
 *   handle, \a eofnvscisyncfence is NULL, or \a eofnvscisyncobj is not
 *   registered with \a decoder as type NVMEDIA_EOFSYNCOBJ or
 *   NVMEDIA_EOF_PRESYNCOBJ.
 * - NVMEDIA_STATUS_ERROR if the function was called before
 *   NvMediaIJPDRender() was called.
 */
NvMediaStatus
NvMediaIJPDGetEOFNvSciSyncFence(
    const NvMediaIJPD        *decoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);


/** @} <!-- Ends x_image_jpeg_decode_api Image JPEG Decoder --> */

/*
 * @defgroup 6x_history_nvmedia_ijpd History
 * Provides change history for the NvMedia Image Jpeg Decode API.
 *
 * \section 6x_history_nvmedia_ijpd Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IJPD_H */
