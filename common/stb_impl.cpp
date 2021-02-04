#include <stdlib.h>
#include <stddef.h>

#include <zlib.h>
static unsigned char* compress_for_stbiw(unsigned char *data, int data_len, int *out_len, int quality) {
  uLongf bufSize = compressBound(data_len);
  // note that buf will be free'd by stb_image_write.h with STBIW_FREE() (plain free() by default)
  unsigned char* buf = (unsigned char*) malloc(bufSize);
  if(buf == NULL)  return NULL;
  if(compress2(buf, &bufSize, data, data_len, quality) != Z_OK) {
    free(buf);
    return NULL;
  }
  *out_len = bufSize;
  return buf;
}

#define STBIW_ZLIB_COMPRESS compress_for_stbiw
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"


