#include "hittable.h"
#include "hittable_list.h"
#include "lambertian.h"
#include "open_rendered_image.h"
#include "sphere.h"
#include "material.h"
#include "metal.h"
#include "lambertian.h"
#include "vec3.h"
#include "camera.h"

int main() {
  
  hittable* world = hittable_list_create();

  material* material_ground = mat_lambertian(vec3_create(0.8, 0.8, 0.0));
  material* material_center = mat_lambertian(vec3_create(0.1, 0.2, 0.5));
  material* material_left = mat_metal(vec3_create(0.8, 0.8, 0.8));
  material* material_right = mat_metal(vec3_create(0.8, 0.6, 0.2));

  hittable_list_add(world, sphere_create(vec3_create( 0.0,-100.5, -1.0), 100, material_ground));
  hittable_list_add(world, sphere_create(vec3_create( 0.0,   0.0, -1.2), 0.5, material_center));
  hittable_list_add(world, sphere_create(vec3_create(-1.0,   0.0, -1.0), 0.5, material_left));
  hittable_list_add(world, sphere_create(vec3_create( 1.0,   0.0, -1.0), 0.5, material_right));

  camera cam;

  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 400;
  cam.samples_per_pixel = 500;
  cam.max_depth         = 50;

  render(world);

  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }
}
