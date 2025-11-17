#include "rtw_stb_image.h"
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>



static int clamp(int x, int low, int high);
static void convert_to_bytes(rtw_image* ri);

// public functions
// Checks if the image exists before calling rtw_image_load
rtw_image* rtw_image_create(char* filename){
  rtw_image* ri = malloc(sizeof(rtw_image));

  ri->bytes_per_pixel = 3;
  ri->fdata = NULL;
  ri->bdata = NULL;
  
  char* imagedir = getenv("RTW_IMAGES");

  if (!imagedir) {
    fprintf(stderr, "ERROR: RTW_IMAGES not set\n");
    free(ri);
    return NULL;
  }

  // i love c
  char path[512];

  snprintf(path, sizeof(path), "%s%s", imagedir, filename);
  // calls to load image in if condition
  if (!rtw_image_load(ri, "./resources/earthmap.jpg")) {
    fprintf(stderr, "ERROR: rtw_image EMPTY, check if RTW_IMAGES is specified\nRTW_IMAGES: %s\n", path);
    return NULL;
  }

  return ri;
}

bool rtw_image_load(rtw_image* ri, char* filename){
  int n = ri->bytes_per_pixel;
  ri->fdata = stbi_loadf(filename, &ri->image_width, &ri->image_height, &n, ri->bytes_per_pixel);
  fprintf(stderr, "ri->fdata: %f", *ri->fdata);
  
  if (!ri->fdata){
    fprintf(stderr, "ri->fdata empty!");
    return false;
  }

  ri->bytes_per_scanline = ri->image_width * ri->bytes_per_pixel;
  convert_to_bytes(ri);
  return true;
}

int rtw_image_width(rtw_image* ri) {
  return ri->fdata == NULL ? 0 : ri->image_width;
}
int rtw_image_height(rtw_image* ri) {
  return ri->fdata == NULL ? 0 : ri->image_height;
}

unsigned char* pixel_data(rtw_image* ri, int x, int y) {
  // returns the address of the three RGB bytes of the pizel at x,y
  // if no data, returns magenta
  // the color is unsigned becuase RGB is valued from 0 to 255
  // if you put 255 in a normal(unsigned) char, it will overflow to -1
  static unsigned char magenta[] = { 255, 0, 255};
  //fprintf(stderr, "checking ri->bdata...\n");
  if (ri->bdata == NULL)
    return magenta;

  x = clamp(x, 0, ri->image_width);
  y = clamp(y, 0, ri->image_height);

  return ri->bdata + y*ri->bytes_per_scanline + x*ri->bytes_per_pixel;
}

void rtw_image_delete(rtw_image* self);

// private functions

static int clamp(int x, int low, int high) {
  if (x < low) return low;
  if (x < high) return x;
  return high - 1;
}

static unsigned char float_to_byte(float value) {
  if (value <= 0.0)
    return 0;
  if (1.0 <= value)
    return 255;
  return (unsigned char)256.0 * value;
}

static void convert_to_bytes(rtw_image* ri) {
  int total_bytes = ri->image_width * ri->image_height * ri->bytes_per_pixel;
  ri->bdata = malloc(total_bytes);

  // Iterate through all pixel components, converting from [0.0, 1.0] float values to
  // unsigned[0, 255] byte values;
  
  unsigned char* bptr = ri->bdata;
  float* fptr = ri->fdata;
  for (int i = 0; i < total_bytes; i++, fptr++, bptr++)
    *bptr = float_to_byte(*fptr);
} 

// Restore MSVC compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif
