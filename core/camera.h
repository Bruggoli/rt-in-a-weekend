#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "vec3.h"
#include "color.h"

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
        u, v, w,              // camera frame basis vectors
        defocus_disk_u,       // Defocus disk horizontal radius
        defocus_disk_v;       // Defocus disk vertical radius
  double  aspect_ratio,         // ratio of image width over height
          pixel_samples_scale,  // color scale factor for a sum of pixel samples (anti-aliasing)
          vfov,                 // vertical FOV of camera
          defocus_angle,  // Variation angle of rays through each pixel
          focus_dist;     // Distance from camera lookfrom point to plane of perfect focus
  color background;     // Scene background color
} camera;

void render(hittable* world, camera* cam);

void camera_initialize(camera* cam);

color ray_color(ray r, int depth, hittable* world, color background);

ray camera_get_ray(camera* c, double i, double j);

vec3 ray_sample_square();

void camera_diagnostics(camera* c);
#endif
