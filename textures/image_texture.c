#include "image_texture.h"
#include "../core/interval.h"


texture* image_texture_create(char* filename) {
  image_texture* it = malloc(sizeof(image_texture));
  texture* tx = malloc(sizeof(texture));
  it->image = *rtw_image_create(filename);

  tx->data = it;
  tx->value = image_texture_value_fn;

  return tx;
}

color image_texture_value_fn(texture* tx, double u, double v, point3 p) {
  image_texture* it = tx->data;
  // returns cyan here if no data present
  if (rtw_image_height(&it->image) <= 0){
    fprintf(stderr, "ERROR: texture image data missing, showing cyan: %d\n", rtw_image_height(&it->image));
    return color_create(0, 1, 1);
  }

  // clamp input texture coordinates to [0, 1] x [1, 0]
  u = clamp(interval_create(0.0, 1.0), u);
  v = 1.0 - clamp(interval_create(0.0, 1.0), v);   // Flip V to image coordinates

  int i = (int)(u * rtw_image_width(&it->image));
  int j = (int)(v * rtw_image_height(&it->image));

  unsigned char* pixel = pixel_data(&it->image, i, j);

  // Debug: print first few pixels
  static int count = 0;
  if (count < 500 && count < 510) {
    fprintf(stderr, "Pixel %d at (%d,%d): R=%d G=%d B=%d\n", 
            count++, i, j, pixel[0], pixel[1], pixel[2]);
  }

  double color_scale = 1.0 / 255.0;
  return color_create(color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]);
}
