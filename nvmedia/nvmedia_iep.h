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
// API reference: https://developer.nvidia.com/docs/drive/drive-os/6.0.10/public/drive-os-linux-sdk/api_reference/nvmedia__iep_8h.html
//
//
// The Doxygen comments were automatically merged back into this file by AI and may not be accurate.
// When in doubt, consult the API reference documentation linked above.
//

/**
 * \file
 * \brief NVIDIA Media Interface: NvMedia Image Encode Processing API
 *
 * This file contains the Image Encode Processing API.
 */

#ifndef NVMEDIA_IEP_H
#define NVMEDIA_IEP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "nvmedia_core.h"
#include "nvscisync.h"
#include "nvscibuf.h"
#include "nvmedia_common_encode.h"

/** \brief Major Version number. */
#define NVMEDIA_IEP_VERSION_MAJOR   1

/** \brief Minor Version number. */
#define NVMEDIA_IEP_VERSION_MINOR   0

/** \brief Patch Version number. */
#define NVMEDIA_IEP_VERSION_PATCH   2

/** \brief Specifies the maximum number of times NvMediaIEPInsertPreNvSciSyncFence()
 * can be called before each call to NvMediaIEPFeedFrame(). */
#define NVMEDIA_IEP_MAX_PRENVSCISYNCFENCES  (16U)

/**
 * \brief Image encode type.
 */
typedef enum {
    NVMEDIA_IMAGE_ENCODE_H264,
    NVMEDIA_IMAGE_ENCODE_HEVC,
    NVMEDIA_IMAGE_ENCODE_VP9,
    NVMEDIA_IMAGE_ENCODE_AV1,
    NVMEDIA_IMAGE_ENCODE_END
} NvMediaIEPType;

/**
 * \brief Opaque NvMediaIEP object created by NvMediaIEPCreate.
 */
typedef struct NvMediaIEP NvMediaIEP;

/**
 * \brief Retrieves the version information for the NvMedia IEP library.
 *
 * \param[out] version  A pointer to an \ref NvMediaVersion structure
 *                      to be filled by the IEP library.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was passed.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPGetVersion(
    NvMediaVersion *version
);

/**
 * \brief Create an NvMediaIEP object instance.
 *
 * Creates an NvMediaIEP object capable of turning a stream of YUV surfaces
 * characterized by NvSciBufAttrList into a compressed bitstream of the
 * specified NvMediaIEPType codec type. Surfaces are fed to the encoder with
 * NvMediaIEPFeedFrame and generated bitstream buffers are retrieved with
 * NvMediaIEPGetBits.
 *
 * \param[in] encodeType         The video compression standard to be used for
 *                               encoding. Input range: Entries in
 *                               \ref NvMediaIEPType enumeration.
 * \param[in] initParams         The encoder initialization parameters. The
 *                               structure type depends on the codec selected
 *                               via \a encodeType.
 * \param[in] bufAttrList        A reconciled \ref NvSciBufAttrList
 *                               characterizing the input surface that needs
 *                               to be encoded.
 * \param[in] maxInOutBuffering  Maximum number of frames outstanding at any
 *                               given point in time that NvMediaIEP can hold
 *                               before its output must be retrieved. Input
 *                               range: values between 4 and 16, in increments
 *                               of 1.
 * \param[in] instanceId         The ID of the NvENC HW engine instance.
 *
 * \return The created NvMediaIEP handle or NULL if unsuccessful.
 *
 * \pre NvMediaIEPGetVersion()
 * \pre NvMediaIEPFillNvSciBufAttrList()
 * \post NvMediaIEP object is created
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaIEP *
NvMediaIEPCreate(
    NvMediaIEPType encodeType,
    const void *initParams,
    NvSciBufAttrList bufAttrList,
    uint8_t maxInOutBuffering,
    NvMediaEncoderInstanceId instanceId
);

/**
 * \brief Create an NvMediaIEP object instance.
 *
 * Creates an NvMediaIEP object capable of turning a stream of YUV surfaces
 * characterized by NvSciBufAttrList into a compressed bitstream of the
 * specified NvMediaIEPType codec type. Surfaces are fed to the encoder with
 * NvMediaIEPFeedFrame and generated bitstream buffers are retrieved with
 * NvMediaIEPGetBits.
 *
 * \param[in] encodeType         The video compression standard to be used for
 *                               encoding. Input range: Entries in
 *                               \ref NvMediaIEPType enumeration.
 * \param[in] initParams         The encoder initialization parameters. The
 *                               structure type depends on the codec selected
 *                               via \a encodeType.
 * \param[in] subsampleType      Subsampling type of the input YUV surface.
 * \param[in] bitdepth           Bitdepth for the input YUV surface.
 * \param[in] maxInOutBuffering  Maximum number of frames outstanding. Input
 *                               range: values between 4 and 16, in increments
 *                               of 1.
 * \param[in] instanceId         The ID of the NvENC HW engine instance.
 *
 * \return The created NvMediaIEP handle or NULL if unsuccessful.
 *
 * \pre NvMediaIEPGetVersion()
 * \post NvMediaIEP object is created
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaIEP *
NvMediaIEPCreateEx (
    NvMediaIEPType encodeType,
    const void *initParams,
    const NvSciBufSurfSampleType subsampleType,
    const NvSciBufSurfBPC bitdepth,
    uint8_t maxInOutBuffering,
    NvMediaEncoderInstanceId instanceId
);

#if !NV_IS_SAFETY

/**
 * \brief Create an NvMediaIEP object instance.
 *
 * \return The new image encoder's handle or NULL if unsuccessful.
 *
 * \pre NvMediaIEPGetVersion()
 * \post NvMediaIEP object is created
 *
 * \note Supported only in non-safety build.
 */
NvMediaIEP *
NvMediaIEPCreateCtx(
    void
);

/**
 * \brief Initialize an NvMediaIEP object instance.
 *
 * \param[in] encoder       A pointer to the NvMediaIEP object.
 * \param[in] encodeType    The video compression standard to be used for
 *                          encoding.
 * \param[in] initParams    The encoder initialization parameters.
 * \param[in] bufAttrList   A reconciled \ref NvSciBufAttrList characterizing
 *                          the input surface.
 * \param[in] maxBuffering  Maximum number of frames outstanding at any given
 *                          point in time.
 * \param[in] instanceId    The ID of the NvENC HW engine instance.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was passed.
 * - \ref NVMEDIA_STATUS_ERROR if there was a general error.
 *
 * \note Supported only in non-safety build.
 */
NvMediaStatus
NvMediaIEPInit(
    const NvMediaIEP *encoder,
    NvMediaIEPType encodeType,
    const void *initParams,
    NvSciBufAttrList bufAttrList,
    uint8_t maxBuffering,
    NvMediaEncoderInstanceId instanceId
);
#endif /* !NV_IS_SAFETY */

/**
 * \brief Destroys an NvMediaIEP object instance.
 *
 * \param[in] encoder  A pointer to the NvMediaIEP object to be destroyed.
 *
 * \pre NvMediaIEPUnregisterNvSciBufObj()
 * \pre NvMediaIEPUnregisterNvSciSyncObj()
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
void NvMediaIEPDestroy(NvMediaIEP *encoder);

/**
 * \brief Submits the specified frame for encoding.
 *
 * \param[in] encoder     A pointer to the NvMediaIEP object.
 * \param[in] frame       NvSciBufObj handle of the input YUV frame to be
 *                        encoded.
 * \param[in] picParams   Picture-specific encoding parameters.
 * \param[in] instanceId  The ID of the NvENC HW engine instance.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was passed.
 * - \ref NVMEDIA_STATUS_ERROR if there was an error during frame submission.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPFeedFrame(
    NvMediaIEP *encoder,
    const NvSciBufObj frame,
    const void *picParams,
    NvMediaEncoderInstanceId instanceId
);

/**
 * \brief Sets the encoder configuration.
 *
 * \param[in] encoder        A pointer to the NvMediaIEP object.
 * \param[in] configuration  Pointer to encoder configuration structure. The
 *                           configuration structure type depends on the
 *                           encode type used during NvMediaIEP creation.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was passed.
 * - \ref NVMEDIA_STATUS_ERROR if a catch-all error occurred.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPSetConfiguration(
    NvMediaIEP *encoder,
    const void *configuration
);

/**
 * \brief Returns the bitstream for a slice or a frame.
 *
 * It is safe to call this function from a separate thread.
 *
 * \param[in]  encoder              A pointer to the NvMediaIEP object.
 * \param[out] numBytes             Number of bytes of encoded bitstream data
 *                                  retrieved.
 * \param[in]  numBitstreamBuffers  Number of bitstream buffers in the
 *                                  \a bitstreams array.
 * \param[in]  bitstreams           Array of bitstream buffer pointers where
 *                                  encoded data is written.
 * \param[out] extradata            Pointer to extradata buffer for encoder
 *                                  statistics or configurations.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was passed.
 * - \ref NVMEDIA_STATUS_ERROR if there was an error during retrieval.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPGetBits(
    const NvMediaIEP *encoder,
    uint32_t *numBytes,
    uint32_t numBitstreamBuffers,
    const NvMediaBitstreamBuffer *bitstreams,
    void *extradata
);

/**
 * \brief Returns the status of an encoding task submitted using
 * NvMediaIEPFeedFrame, whose encoded output is to be retrieved next.
 *
 * The number of bytes of encoded output that is available (if ready), is
 * also retrieved along with the status. The specific behavior depends on the
 * specified \ref NvMediaBlockingType.
 *
 * \param[in]  encoder            A pointer to the NvMediaIEP object.
 * \param[out] numBytesAvailable  The number of bytes of encoded output that
 *                                is available.
 * \param[in]  blockingType       The type of blocking behavior to use when
 *                                checking for availability.
 * \param[in]  millisecondTimeout Timeout in milliseconds, or
 *                                \ref NVMEDIA_ENCODE_TIMEOUT_INFINITE for an
 *                                infinite timeout.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_PENDING if the operation has not finished yet.
 * - \ref NVMEDIA_STATUS_NONE_PENDING if no operation is pending.
 * - \ref NVMEDIA_STATUS_TIMED_OUT if the operation timed out.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was passed.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPBitsAvailable(
    const NvMediaIEP *encoder,
    uint32_t *numBytesAvailable,
    NvMediaBlockingType blockingType,
    uint32_t millisecondTimeout
);


/**
 * \brief Gets the encoder attribute for the current encoding session.
 *
 * This function can be used to retrieve encoder-specific attributes such as
 * SPS, PPS, or VPS data that can be used by the application.
 *
 * \param[in]  encoder        A pointer to the NvMediaIEP object.
 * \param[in]  attrType       The attribute type to retrieve.
 * \param[in]  attrSize       Size of the attribute data buffer in bytes.
 * \param[out] AttributeData  Pointer to buffer where attribute data will be
 *                            stored.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was passed.
 * - \ref NVMEDIA_STATUS_ERROR if there was a generic error during attribute
 *                             retrieval.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPGetAttribute(
    const NvMediaIEP *encoder,
    NvMediaEncAttrType attrType,
    uint32_t attrSize,
    void *AttributeData
);

/**
 * \brief Registers NvSciBufObj for use with a NvMediaIEP handle.
 *
 * \param[in] encoder  A pointer to the NvMediaIEP object.
 * \param[in] bufObj   The \ref NvSciBufObj to register.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the buffer object was registered successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid encoder or buffer object
 *                                     was provided.
 * - \ref NVMEDIA_STATUS_ERROR if registration failed.
 */
NvMediaStatus
NvMediaIEPRegisterNvSciBufObj(
    NvMediaIEP    *encoder,
    const NvSciBufObj   bufObj
);

/**
 * \brief Un-registers NvSciBufObj which was previously registered with
 * NvMediaIEP using NvMediaIEPRegisterNvSciBufObj().
 *
 * \param[in] encoder  A pointer to the NvMediaIEP object.
 * \param[in] bufObj   The \ref NvSciBufObj to unregister.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the buffer object was unregistered successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid encoder or buffer object
 *                                     was provided.
 * - \ref NVMEDIA_STATUS_ERROR if unregistration failed.
 */
// coverity[misra_c_2012_rule_8_7_violation : FALSE]
NvMediaStatus
NvMediaIEPUnregisterNvSciBufObj(
    const NvMediaIEP    *encoder,
    const NvSciBufObj   bufObj
);


/**
 * \brief Fills the NvMediaIEP specific NvSciBuf attributes which than then be
 * used to allocate an NvSciBufObj that NvMediaIEP can consume.
 *
 * \param[in]     instanceId  The ID of the NvENC HW engine instance.
 * \param[in,out] attrlist    Pointer to a list of NvSciBuf attributes to be
 *                            filled with NvMediaIEP specific requirements.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was provided.
 */
NvMediaStatus
NvMediaIEPFillNvSciBufAttrList(
    NvMediaEncoderInstanceId  instanceId,
    NvSciBufAttrList          attrlist
);


/**
 * \brief Fills the NvMediaIEP specific NvSciSync attributes.
 *
 * \param[in]     encoder     A pointer to the NvMediaIEP object.
 * \param[in,out] attrlist    Pointer to NvSciSync attribute list to be filled
 *                            with encoder requirements.
 * \param[in]     clienttype  Type of NvSciSync client (producer or consumer).
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was provided.
 */
NvMediaStatus
NvMediaIEPFillNvSciSyncAttrList(
    const NvMediaIEP           *encoder,
    NvSciSyncAttrList          attrlist,
    NvMediaNvSciSyncClientType clienttype
);


/**
 * \brief Registers an NvSciSyncObj with NvMediaIEP.
 *
 * \param[in] encoder      A pointer to the NvMediaIEP object.
 * \param[in] syncobjtype  The type of NvSciSync object being registered.
 * \param[in] syncObj      The NvSciSyncObj to register with the encoder.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was provided.
 * - \ref NVMEDIA_STATUS_ERROR if there was a general error during
 *                             registration.
 */
NvMediaStatus
NvMediaIEPRegisterNvSciSyncObj(
    const NvMediaIEP           *encoder,
    NvMediaNvSciSyncObjType    syncobjtype,
    NvSciSyncObj               syncObj
);

/**
 * \brief Unregisters an NvSciSyncObj with NvMediaIEP.
 *
 * \param[in] encoder  A pointer to the NvMediaIEP object.
 * \param[in] syncObj  The NvSciSyncObj to unregister.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was provided.
 * - \ref NVMEDIA_STATUS_ERROR if there was a general error during
 *                             unregistration.
 */
NvMediaStatus
NvMediaIEPUnregisterNvSciSyncObj(
    const NvMediaIEP  *encoder,
    NvSciSyncObj      syncObj
);

/**
 * \brief Specifies the NvSciSyncObj to be used for an EOF NvSciSyncFence.
 *
 * \param[in] encoder       A pointer to the NvMediaIEP object.
 * \param[in] nvscisyncEOF  The NvSciSyncObj to be used for the EOF
 *                          NvSciSyncFence.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if an invalid parameter was provided.
 * - \ref NVMEDIA_STATUS_ERROR if there was a general error during
 *                             configuration.
 */
NvMediaStatus
NvMediaIEPSetNvSciSyncObjforEOF(
    const NvMediaIEP      *encoder,
    NvSciSyncObj          nvscisyncEOF
);

/**
 * \brief Sets an NvSciSyncFence as a prefence for an NvMediaIEPFeedFrame()
 * NvSciSyncFence operation.
 *
 * This function can be called up to NVMEDIA_IEP_MAX_PRENVSCISYNCFENCES times
 * before each NvMediaIEPFeedFrame() call.
 *
 * \param[in] encoder            A pointer to the NvMediaIEP object.
 * \param[in] prenvscisyncfence  A pointer to the NvSciSyncFence prefence.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if any input parameters are invalid.
 * - \ref NVMEDIA_STATUS_ERROR if there was an error.
 */
NvMediaStatus
NvMediaIEPInsertPreNvSciSyncFence(
    const NvMediaIEP         *encoder,
    const NvSciSyncFence     *prenvscisyncfence
);

/**
 * \brief Gets EOF NvSciSyncFence for an NvMediaIEPFeedFrame() operation.
 *
 * \param[in]  encoder            A pointer to the NvMediaIEP object.
 * \param[in]  eofnvscisyncobj    The NvSciSyncObj for the EOF fence.
 * \param[out] eofnvscisyncfence  A pointer to the EOF NvSciSyncFence.
 *
 * \return \ref NvMediaStatus, the completion status of the operation:
 * - \ref NVMEDIA_STATUS_OK if the operation completed successfully.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if any input parameters are invalid.
 * - \ref NVMEDIA_STATUS_ERROR if there was an error.
 */
NvMediaStatus
NvMediaIEPGetEOFNvSciSyncFence(
    const NvMediaIEP        *encoder,
    NvSciSyncObj      eofnvscisyncobj,
    NvSciSyncFence    *eofnvscisyncfence
);

/*
 * @defgroup 6x_history_nvmedia_iep History
 * Provides change history for the NvMedia IEP API.
 *
 * \section 6x_history_nvmedia_iep Version History
 *
 * <b> Version 1.0 </b> September 28, 2021
 * - Initial release
 *
 * <b> Version 1.1 </b> August 03, 2022
 * - Added new quality preset API, NvMediaEncPreset
 *
 * <b> Version 1.2 </b> March 07, 2023
 * - Added new create API NvMediaIEPCreateEx which does not depends on reconciled attribute list
 *
 */

#ifdef __cplusplus
}     /* extern "C" */
#endif

#endif /* NVMEDIA_IEP_H */
