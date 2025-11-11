#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "vec3.h"

typedef struct {
  int   image_height,           // rendered image height
        image_width,            // rendered image width
        samples_per_pixel,      // count of random samples per pixel
        max_depth;              // Max number of ray bounces into scene
  point3  pixel00_loc,          // location of pixel at 0, 0
          center,               // camera center
          lookfrom,             // where camera is placed in scene
          lookat;               // what the camera is looking at
  vec3  pixel_delta_u,        // offset to pixel to the right
        pixel_delta_v,        // offset to pixel below
        vup,                  // camera relative "up" direction
        u, v, w;              // camera frame basis vectors
  double  aspect_ratio,         // ratio of image width over height
          pixel_samples_scale,  // color scale factor for a sum of pixel samples (anti-aliasing)
          vfov;                 // vertical FOV of camera
} camera;

void render(hittable* world);

camera camera_initialize();

vec3 ray_color(ray r, int max_depth, hittable* world);

ray camera_get_ray(camera* c, double i, double j);

vec3 ray_sample_square();

#endif
