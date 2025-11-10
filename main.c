#include "hittable.h"
#include "hittable_list.h"
#include "lambertian.h"
#include "sphere.h"
#include "vec3.h"
#include "camera.h"

int main() {
  
  hittable* world = hittable_list_create();

  hittable_list_add(world, sphere_create(vec3_create(0, 0, -1), 0.5, mat_lambertian(vec3_create(1, 1, 1))));
  hittable_list_add(world, sphere_create(vec3_create(0, -100.5, -1), 100, mat_lambertian(vec3_create(1, 1, 1))));

  camera cam;
  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 400;
  cam.samples_per_pixel = 500;
  cam.max_depth         = 50;

  render(world);

}
