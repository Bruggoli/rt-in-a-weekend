#include "core/color.h"
#include "core/hittable.h"
#include "core/hittable_list.h"
#include "core/vec3.h"
#include "core/camera.h"
#include "materials/lambertian.h"
#include "materials/material.h"
#include "materials/metal.h"
#include "materials/dielectric.h"
#include "textures/solid_color.h"
#include "utils/rtweekend.h"
#include "utils/open_rendered_image.h"
#include "geometries/sphere.h"
#include "textures/texture.h"
#include "textures/checker_texture.h"
#include "textures/noise_texture.h"
#include "textures/perlin.h"
#include "accel/bvh.h"
#include "textures/image_texture.h"

void bouncing_spheres() {
  
  hittable* world = hittable_list_create();

  material* ground_material = mat_lambertian(vec3_create(0.5, 0.5, 0.5));
  texture* checker = checker_texture_create_color(0.32, color_create(0.2, 0.3, 0.1), color_create(0.9, 0.9, 0.9));
  hittable_list_add(world, sphere_create(vec3_create(0, -1000, 0), 1000, mat_lambertian_tx(checker)));

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
          point3 center2 = vec3_add(center, vec3_create(0, random_double_range(0, 0.5), 0));
          hittable_list_add(world, sphere_create_moving(center, center2, 0.2, sphere_material));
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
  
  fprintf(stderr, "Finished adding objects to world\n");

// set up bvh
  hittable_list* world_data = (hittable_list*)world->data;

  hittable* bvh_root = bvh_node_create_range(world_data);
  // We need to overwrite the old world instead of just adding the bvh root to the old world
  // BVH allegedly already contains all the objects PLUS the bounding boxes
  hittable* new_world = hittable_list_create();
  hittable_list_add(new_world, bvh_root);
  world = new_world;

  camera cam;
  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 1200;
  cam.samples_per_pixel = 20;
  cam.max_depth         = 50;
  cam.vfov              = 20;
  cam.lookfrom          = vec3_create(13, 2, 3);
  cam.lookat            = vec3_create(0, 0, 0);
  cam.vup               = vec3_create(0, 1, 0);
  cam.defocus_angle     = 0.6;
  cam.focus_dist        = 10.0;


  camera_initialize(&cam);

  camera_diagnostics(&cam);
  render(world, &cam);  
  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }
}


void checkered_spheres() {
  
  hittable* world = hittable_list_create();

  texture* checker = checker_texture_create_color(0.32, color_create(0.2, 0.3, 0.1), color_create(0.9, 0.9, 0.9));
  hittable_list_add(world, sphere_create(vec3_create(0, -1, 0), 1, mat_lambertian_tx(checker)));
  hittable_list_add(world, sphere_create(vec3_create(0, 1, 0), 1, mat_lambertian_tx(checker)));

  fprintf(stderr, "Finished adding objects to world\n");

// set up bvh
  hittable_list* world_data = (hittable_list*)world->data;

  hittable* bvh_root = bvh_node_create_range(world_data);
  // We need to overwrite the old world instead of just adding the bvh root to the old world
  // BVH allegedly already contains all the objects PLUS the bounding boxes
  hittable* new_world = hittable_list_create();
  hittable_list_add(new_world, bvh_root);
  world = new_world;

  camera cam;
  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 800;
  cam.samples_per_pixel = 100;
  cam.max_depth         = 50;

  cam.vfov              = 20;
  cam.lookfrom          = vec3_create(13, 2, 3);
  cam.lookat            = vec3_create(0, 0, 0);
  cam.vup               = vec3_create(0, 1, 0);
  
  cam.defocus_angle     = 0.6;
  cam.focus_dist        = 10.0;


  camera_initialize(&cam);

  camera_diagnostics(&cam);
  render(world, &cam);  
  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }
}

void earth() {
  hittable* world = hittable_list_create();

  texture* earth_texture = image_texture_create("earthmap.jpg");
  material* earth_surface = mat_lambertian_tx(earth_texture);
  hittable_list_add(world, sphere_create(vec3_create(0, 0, 0), 1, earth_surface));

  camera cam;
  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 800;
  cam.samples_per_pixel = 100;
  cam.max_depth         = 50;

  cam.vfov              = 20;
  cam.lookfrom          = vec3_create(0, 0, 12);
  cam.lookat            = vec3_create(0, 0, 0);
  cam.vup               = vec3_create(0, 1, 0);
  
  cam.defocus_angle     = 0.6;
  cam.focus_dist        = 10.0;


  camera_initialize(&cam);

  camera_diagnostics(&cam);
  render(world, &cam);  
  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }

}

void perlin_spheres() {
  render_noise_map("noise.ppm");
  texture* pertext = noise_texture_create(4);
  material* perlin_mat = mat_lambertian_tx(pertext);
  
  hittable* world = hittable_list_create();

  hittable_list_add(world, sphere_create(vec3_create(0, -1000, 0), 1000, perlin_mat));
  hittable_list_add(world, sphere_create(vec3_create(0, 2, 0), 2, perlin_mat));

  camera cam;
  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 800;
  cam.samples_per_pixel = 100;
  cam.max_depth         = 50;

  cam.vfov              = 20;
  cam.lookfrom          = vec3_create(13,2,3);
  cam.lookat            = vec3_create(0, 0, 0);
  cam.vup               = vec3_create(0, 1, 0);
  
  cam.defocus_angle     = 0.6;
  cam.focus_dist        = 10.0;


  camera_initialize(&cam);

  camera_diagnostics(&cam);
  render(world, &cam);  
  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }}

int main() {
  switch (4) {
    case 1: bouncing_spheres(); break;
    case 2: checkered_spheres(); break;
    case 3: earth(); break;
    case 4: perlin_spheres(); break;
  }
}
