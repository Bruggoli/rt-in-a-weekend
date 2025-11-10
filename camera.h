#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "vec3.h"

typedef struct {
  int   image_height, 
        image_width;
  vec3 pixel00_loc, 
       pixel_delta_u, 
       pixel_delta_v, 
       camera_center;
  double aspect_ratio;
} camera;

void render(hittable* world);

camera camera_initialize();

vec3 ray_color(ray r, hittable* world);

#endif
