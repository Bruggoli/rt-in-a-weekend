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
#include "dielectric.h"
#include "rtweekend.h"


int main() {
  
  hittable* world = hittable_list_create();

  material* ground_material = mat_lambertian(vec3_create(0.5, 0.5, 0.5));
  hittable_list_add(world, sphere_create(vec3_create(0, -1000, 0), 1000, ground_material));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      double choose_mat = random_double();
      point3 center = vec3_create(a + 0.9*random_double(), 0.2, b + 0.9*random_double());
      
      if (vec3_length(vec3_sub(center, vec3_create(4, 0.2, 0))) > 0.9) {
        material* sphere_material;
        
        if (choose_mat < 0.8) {
          // diffuse
          color albedo = vec3_mul(vec3_random(), vec3_random());
          sphere_material = mat_lambertian(albedo);
          hittable_list_add(world, sphere_create(center, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // metal
          color albedo = vec3_random_range(0.5, 1);
          double fuzz = random_double_range(0, 0.5);
          sphere_material = mat_metal(albedo, fuzz);
          hittable_list_add(world, sphere_create(center, 0.2, sphere_material));
        } else {
          // glass
          sphere_material = mat_dielectric(1.5);
          hittable_list_add(world, sphere_create(center, 0.2, sphere_material));
        }
      }
    }
  }

  material* material1 = mat_dielectric(1.5);
  hittable_list_add(world, sphere_create(vec3_create(0, 1, 0), 1.0, material1));

  material* material2 = mat_lambertian(vec3_create(0.4, 0.2, 0.1));
  hittable_list_add(world, sphere_create(vec3_create(-4, 1, 0), 1.0, material2));

  material* material3 = mat_metal(vec3_create(0.7, 0.6, 0.5), 0.0);
  hittable_list_add(world, sphere_create(vec3_create(4, 1, 0), 1.0, material3));

  camera cam;
  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 1200;
  cam.samples_per_pixel = 50;
  cam.max_depth         = 10;
  cam.vfov              = 20;
  cam.lookfrom          = vec3_create(13, 2, 3);
  cam.lookat            = vec3_create(0, 0, 0);
  cam.vup               = vec3_create(0, 1, 0);
  cam.defocus_angle     = 0.6;
  cam.focus_dist        = 10.0;

  camera_diagnostics(&cam);

  camera_initialize(&cam);

  camera_diagnostics(&cam);
  render(world, &cam);  
  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }
}
