#include "readPFM.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include <opencv2/imgproc.hpp>

/***
A PFM file contains is a 1- or 3-channel floating point image.

 The text header of a .pfm file takes the following form:

[type]
[xres] [yres]
[byte_order]

Each of the three lines of text ends with zero or more spaces (0x20), followed by a 1-byte Unix-style carriage return (0x0a).
  (note that the "zero or more spaces" part deviates from the actual PFM spec -- the SceneFlow dataset that this will
  be used on has spaces in its headers, which breaks the OpenCV built-in PFM decoder.)

The "[type]" is one of "PF" for a 3-channel RGB color image, or "Pf" for a monochrome single-channel image.
"[xres] [yres]" indicates the x and y resolutions of the image.
"[byte_order]" is a number used to indicate the byte order within the file.
A positive number (e.g. "1.0") indicates big-endian, with the most significant byte of each 4-byte float first.
If the number is negative (e.g. "-1.0") this indicates little-endian, with the least significant byte first.
For example:

PF
768 512
-1.0

Indicates an RGB color image, 768 by 512 pixels, with little-endian byte order.

After the final carriage return the file proceeds with a series of three 4-byte IEEE 754 single precision floating point numbers for each pixel, specified in left to right, bottom to top order.
*/

namespace {

bool readHeaderLine(FILE* file, char* buffer, size_t bufferSize) {
  size_t offset = 0;
  while (offset + 1 < bufferSize) {
    int ch = fgetc(file);
    if (ch == EOF) {
      return false;
    }
    if (ch == 0x0a) {
      buffer[offset] = '\0';
      // Strip trailing spaces.
      while (offset > 0 && buffer[offset - 1] == 0x20) {
        buffer[--offset] = '\0';
      }
      return true;
    }
    buffer[offset++] = static_cast<char>(ch);
  }
  return false;
}

bool isHostLittleEndian() {
  const uint32_t probe = 1;
  return *reinterpret_cast<const uint8_t*>(&probe) == 1;
}

void byteswap32(uint8_t* data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    uint8_t* p = data + (i * 4);
    std::swap(p[0], p[3]);
    std::swap(p[1], p[2]);
  }
}

} // namespace

bool readPFM(const char* filename, cv::Mat& outImage) {
  std::unique_ptr<FILE, int (*)(FILE*)> file(std::fopen(filename, "rb"), &std::fclose);
  if (!file) {
    fprintf(stderr, "readPFM: failed to open '%s'\n", filename);
    return false;
  }

  char typeLine[16];
  char dimsLine[64];
  char orderLine[64];
  if (!readHeaderLine(file.get(), typeLine, sizeof(typeLine)) ||
    !readHeaderLine(file.get(), dimsLine, sizeof(dimsLine)) ||
    !readHeaderLine(file.get(), orderLine, sizeof(orderLine))) {
    fprintf(stderr, "readPFM: '%s' has malformed header\n", filename);
    return false;
  }

  int channels;
  if (std::strcmp(typeLine, "PF") == 0) {
    channels = 3;
  } else if (std::strcmp(typeLine, "Pf") == 0) {
    channels = 1;
  } else {
    fprintf(stderr, "readPFM: '%s' has unrecognized type '%s'\n", filename, typeLine);
    return false;
  }

  int width = 0;
  int height = 0;
  if (std::sscanf(dimsLine, "%d %d", &width, &height) != 2 || width <= 0 || height <= 0) {
    fprintf(stderr, "readPFM: '%s' has invalid dimensions '%s'\n", filename, dimsLine);
    return false;
  }

  float byteOrder = 0.0f;
  if (std::sscanf(orderLine, "%f", &byteOrder) != 1 || byteOrder == 0.0f) {
    fprintf(stderr, "readPFM: '%s' has invalid byte order '%s'\n", filename, orderLine);
    return false;
  }
  const bool fileLittleEndian = (byteOrder < 0.0f);

  const int matType = (channels == 3) ? CV_32FC3 : CV_32FC1;
  cv::Mat image(height, width, matType);

  const size_t rowBytes = static_cast<size_t>(width) * channels * sizeof(float);
  // PFM pixels are stored bottom-to-top; read each scanline directly into the
  // mirrored row of the destination so the result is top-to-bottom.
  for (int row = 0; row < height; ++row) {
    uint8_t* destRow = image.ptr<uint8_t>(height - 1 - row);
    if (std::fread(destRow, 1, rowBytes, file.get()) != rowBytes) {
      fprintf(stderr, "readPFM: '%s' truncated at row %d\n", filename, row);
      return false;
    }
  }

  if (fileLittleEndian != isHostLittleEndian()) {
    byteswap32(image.ptr<uint8_t>(0), static_cast<size_t>(width) * height * channels);
  }

  // PFM stores 3-channel data as RGB; convert to OpenCV's BGR convention.
  if (channels == 3) {
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  }

  outImage = image;
  return true;
}
