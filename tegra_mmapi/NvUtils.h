/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file
 * <b>NVIDIA Multimedia API: Utility Functions</b>
 *
 * @b Description: This file declares the video frame utility functions.
 */

/**
 * @defgroup l4t_mm_nvutils_group Video Read/Write
 * @ingroup l4t_mm_nvvideo_group
 *
 * Utility functions for reading video frames from a file to the buffer structure and
 * for writing from the buffer structure to a file.
 *
 * @{
 */

#ifndef __NV_UTILS_H_
#define __NV_UTILS_H_

#include <fstream>
#include "NvBuffer.h"

/**
 * @brief Reads a video frame from a file to the buffer structure.
 *
 * This function reads data from the file into the buffer plane-by-plane.
 * While taking care of the stride of the plane, for each data plane it reads:
 * @code width * height * byteperpixel @endcode
 *
 * @param[in] stream A pointer to the input file stream.
 * @param[in] buffer A reference to the buffer object into which the data are read.
 * @return 0 for success, -1 otherwise.
 */
int read_video_frame(std::ifstream * stream, NvBuffer & buffer);

/**
 * @brief Writes a video frame from the buffer structure to a file.
 *
 * This function writes data to the file from the buffer plane-by-plane.
 * While taking care of the stride of the plane, for each data plane it reads:
 * @code width * height * byteperpixel @endcode
 *
 * @param[in] stream A pointer to the output file stream.
 * @param[in] buffer A reference to the buffer object from which the data are written.
 * @return 0 for success, -1 otherwise.
 */
int write_video_frame(std::ofstream * stream, NvBuffer & buffer);

/**
 * @brief Writes a plane data of the buffer to a file.
 *
 * This function writes data to the file from a plane of the buffer.
 *
 * @param[in] dmabuf_fd DMABUF FD of buffer.
 * @param[in] plane video frame plane.
 * @param[in] stream A pointer to the output file stream.
 * @return 0 for success, -1 otherwise.
 */
int dump_dmabuf(int dmabuf_fd, unsigned int plane, std::ofstream * stream);
/** @} */
#endif
