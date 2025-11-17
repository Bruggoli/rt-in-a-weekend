#ifndef RTW_STB_IMAGE_H
#define RTW_STB_IMAGE_H

// Disable strict warnings for this header from the Microsoft Visual C++ compiler.
#ifdef _MSC_VER
    #pragma warning (push, 0)
#endif


#include <stdlib.h>
#include <stdbool.h>

typedef struct rtw_image {
  int             bytes_per_pixel;
  float          *fdata;
  unsigned char  *bdata;
  int             image_width,
                  image_height,
                  bytes_per_scanline;
} rtw_image;

rtw_image* rtw_image_create(char* image_filename);

bool rtw_image_load(rtw_image* ri, char* filename);

int rtw_image_width(rtw_image* ri);
int rtw_image_height(rtw_image* ri);

unsigned char* pixel_data(rtw_image* ri, int x, int y);

void rtw_image_delete(rtw_image* self);


#endif
