#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "vec3.h"

typedef struct {
  int   image_height,
        image_width,
        samples_per_pixel,
        max_depth;
  point3 pixel00_loc, 
         center;
  vec3   pixel_delta_u, 
         pixel_delta_v;
  double aspect_ratio,
         pixel_samples_scale;
} camera;

void render(hittable* world);

camera camera_initialize();

vec3 ray_color(ray r, int max_depth, hittable* world);

ray camera_get_ray(camera* c, double i, double j);

vec3 ray_sample_square();

#endif
