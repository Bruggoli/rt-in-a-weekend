#include "../utils/rtw_stb_image.h"
#include "texture.h"

typedef struct {
  rtw_image image;
}image_texture;

texture* image_texture_create(char* filename);

color image_texture_value_fn(texture* tx, double u, double v, point3 p);
