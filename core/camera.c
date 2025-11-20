#include "camera.h"
#include "../materials/material.h"
#include "hittable.h"
#include "color.h"
#include "ray.h"
#include "../utils/rtweekend.h"
#include "vec3.h"
#include <stdio.h>
#include <omp.h>


// Sets default values in case they are not set where they are used
double  aspect_ratio      = 16.0 / 9.0;
int     image_width       = 20;
int     samples_per_pixel = 10;
int     max_depth         = 10;

double  vfov              = 90;
double  defocus_angle     = 0;
double  focus_dist        = 10;

point3  lookfrom          = (vec3){0, 0, 0};
point3  lookat            = (vec3){0, 0,-1};
point3  vup               = (vec3){0, 1, 0};


void render(hittable* world, camera* cam) {

  // allocate buffer for all pixels
  color* image_buffer = malloc(cam->image_width*cam->image_height*sizeof(color));
  
  printf("P3\n%d %d\n255\n", cam->image_width, cam->image_height);

  #pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < cam->image_height; j++) {
    #pragma omp critical 
    {
    fprintf(stderr, "\rScanlines remaining: %d/%d", (cam->image_height - j), cam->image_height);
    fflush(stderr);
    }
    for (int i = 0; i < cam->image_width; i++) {
      color pixel_color = vec3_create(0, 0, 0);
      for (int sample = 0; sample < cam->samples_per_pixel; sample++) {
        unsigned int seed = omp_get_thread_num() + j;
        ray r = camera_get_ray(cam, i, j);
        pixel_color = vec3_add(pixel_color, ray_color(r, cam->max_depth, world, cam->background));
      }
    image_buffer[j * cam->image_width + i] = vec3_scale( pixel_color, cam->pixel_samples_scale);
    }
  }

  
  // Sequential write to maintain PPM format
  for (int j = 0; j < cam->image_height; j++) {
    for (int i = 0; i < cam->image_width; i++) {
      write_color(stdout, image_buffer[j * cam->image_width + i]);
    }
  }

  free(image_buffer);
  fprintf(stderr, "\r                                \n");
  fprintf(stderr, "\rDone.\n");
}


void camera_initialize(camera* c) {

  // Calculate image height and ensure > 1
  int image_height = (int)(c->image_width / c->aspect_ratio);
  c->image_height = (image_height < 1) ? 1 : image_height;
  c->pixel_samples_scale = 1.0 / c->samples_per_pixel;
  c->center = c->lookfrom;
  // Determine viewport dimensions.

  double theta = degrees_to_radians(c->vfov);  // <-- c->vfov
  double h = tan(theta / 2);
  double viewport_height = 2.0 * h * c->focus_dist;
  double viewport_width = viewport_height * (double)c->image_width / c->image_height;  // <-- c->image_width
  //point3 camera_center = c->center;

  // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
  c->w = unit_vector(vec3_sub(c->lookfrom, c->lookat));
  c->u = unit_vector(vec3_cross(c->vup, c->w));
  c->v = vec3_cross(c->w, c->u);
  vec3 viewport_u = vec3_scale(c->u, viewport_width);
  vec3 viewport_v = vec3_scale(vec3_negate(c->v), viewport_height);
  c->pixel_delta_u = vec3_div(viewport_u, c->image_width); 
  c->pixel_delta_v = vec3_div(viewport_v, c->image_height);
  
  // auto viewport_upper_left = center - (focal_length * w) - viewport_u/2 - viewport_v/2;
  point3 p0 = vec3_sub(c->center, vec3_scale(c->w, focus_dist));
  p0 = vec3_sub(p0, vec3_div(viewport_u, 2.0));
  p0 = vec3_sub(p0, vec3_div(viewport_v, 2.0));
  point3 viewport_upper_left = p0;
  
  
  vec3 pixel_delta_scaled = vec3_scale(vec3_add(c->pixel_delta_u, c->pixel_delta_v), 0.5);
  c->pixel00_loc = vec3_add(viewport_upper_left, pixel_delta_scaled);

  double defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
  c->defocus_disk_u = vec3_scale(c->u, defocus_radius);
  c->defocus_disk_v = vec3_scale(c->v, defocus_radius);
  fprintf(stderr, "Camera initialised!");

}

color ray_color(ray r, int depth, hittable* world, color background) {

  if (depth <= 0)
    return vec3_create(0, 0, 0);

  hit_record rec;

  if (!world->hit(world, r, interval_create(0.001, INFINITY), &rec))
    return background;

  ray scattered;
  color attenuation;
  color color_from_emission = rec.mat->emit(rec.mat, rec.u, rec.v, rec.p);


  if (!rec.mat->scatter(rec.mat, r, &rec, &attenuation, &scattered)){
    return color_from_emission;
  }

  color color_from_scatter = vec3_mul(attenuation, ray_color(scattered, depth - 1, world, background));

  return vec3_add(color_from_emission, color_from_scatter);
}

point3 defocus_disk_sample(camera* c) {
  vec3 p = random_in_unit_disk();
  // center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
  vec3 d1 = vec3_scale(c->defocus_disk_u, p.e[0]);
  vec3 d2 = vec3_scale(c->defocus_disk_v, p.e[1]);
  return vec3_add(vec3_add(d1, d2), c->center);
}

ray camera_get_ray(camera* c, double i, double j) {
  vec3 offset = ray_sample_square();
  vec3 pixel_sample = vec3_add(
    c->pixel00_loc, 
    vec3_add(
      vec3_scale(c->pixel_delta_u, (i + offset.e[0])),
      vec3_scale(c->pixel_delta_v, (j + offset.e[1]))
      )
  );
  
  vec3 ray_origin = (defocus_angle <= 0) ? c->center : (vec3)defocus_disk_sample(c);
  vec3 ray_direction = vec3_sub(pixel_sample, c->center);
  double ray_time = random_double();

  return ray_create(ray_origin, ray_direction, ray_time);
}

vec3 ray_sample_square() {
  return vec3_create(random_double() - 0.5, random_double() - 0.5, 0);
}

void camera_diagnostics(camera* c) {
  fprintf(stderr, "\nImage dimensions:\nWidth: %d\tHeight: %d", c->image_width, c->image_height);
  fprintf(stderr, "\nCamera coordinates:\nLookfrom:\t");
  vec3_print(stderr, c->lookfrom);
  fprintf(stderr, "Lookat:\t\t");
  vec3_print(stderr, c->lookat);
  fprintf(stderr, "vup:\t\t");
  vec3_print(stderr, c->vup);
}
