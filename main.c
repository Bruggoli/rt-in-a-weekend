#include "core/color.h"
#include "core/hittable.h"
#include "core/hittable_list.h"
#include "core/vec3.h"
#include "core/camera.h"
#include "core/translate.h"
#include "materials/diffuse_light.h"
#include "materials/lambertian.h"
#include "materials/material.h"
#include "materials/metal.h"
#include "materials/dielectric.h"
#include "textures/solid_color.h"
#include "utils/rtweekend.h"
#include "utils/open_rendered_image.h"
#include "geometries/sphere.h"
#include "geometries/quad.h"
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
  cam.background        = color_create(0.70, 0.80, 1.00);

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
  cam.background        = color_create(0.70, 0.80, 1.00);

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
  cam.background        = color_create(0.70, 0.80, 1.00);

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
  cam.background        = color_create(0.70, 0.80, 1.00);

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
  }
}

void quads() {
  hittable* world = hittable_list_create();

  material* left_red     = mat_lambertian(color_create(1.0, 0.2, 0.2));
  material* back_green   = mat_lambertian(color_create(0.2, 1.0, 0.2));
  material* right_blue   = mat_lambertian(color_create(0.2, 0.2, 1.0));
  material* upper_orange = mat_lambertian(color_create(1.0, 0.5, 0.0));
  material* lower_teal   = mat_lambertian(color_create(0.2, 0.8, 0.8));

  hittable_list_add(world, quad_create((point3){-3, -2, 5}, vec3_create(0, 0, -4), vec3_create(0, 4, 0), left_red));
  hittable_list_add(world, quad_create(vec3_create(-2,-2, 0), vec3_create(4, 0, 0), vec3_create(0, 4, 0), back_green));
  hittable_list_add(world, quad_create(vec3_create( 3,-2, 1), vec3_create(0, 0, 4), vec3_create(0, 4, 0), right_blue));
  hittable_list_add(world, quad_create(vec3_create(-2, 3, 1), vec3_create(4, 0, 0), vec3_create(0, 0, 4), upper_orange));
  hittable_list_add(world, quad_create(vec3_create(-2,-3, 5), vec3_create(4, 0, 0), vec3_create(0, 0,-4), lower_teal));

  camera cam;
  cam.aspect_ratio      = 16.0 / 9.0;
  cam.image_width       = 800;
  cam.samples_per_pixel = 100;
  cam.max_depth         = 50;
  cam.background        = color_create(0.70, 0.80, 1.00);

  cam.vfov              = 80;
  cam.lookfrom          = vec3_create(0,0,9);
  cam.lookat            = vec3_create(0, 0, 0);
  cam.vup               = vec3_create(0, 1, 0);
  
  cam.defocus_angle     = 0.0;
  cam.focus_dist        = 10.0;

  camera_initialize(&cam);
  render(world, &cam);  

  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }
}

void simple_light() {
  hittable* world = hittable_list_create();

  texture* pertext = noise_texture_create(4);
  material* perlin_mat = mat_lambertian_tx(pertext);
  
  hittable_list_add(world, sphere_create(vec3_create(0, -1000, 0), 1000, perlin_mat));
  hittable_list_add(world, sphere_create(vec3_create(0, 2, 0), 2, perlin_mat));

  material* difflight = diffuse_light_create_color(color_create(4, 4, 4));
  hittable_list_add(world, quad_create((point3){3, 1, -2}, vec3_create(2, 0, 0), vec3_create(0, 2, 0), difflight));
  hittable_list_add(world, sphere_create(vec3_create(-3, 12, 0), 3, difflight));

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
  cam.background        = color_create(0, 0, 0);

  cam.vfov              = 20;
  cam.lookfrom          = vec3_create(26,3,6);
  cam.lookat            = vec3_create(0, 2, 0);
  cam.vup               = vec3_create(0, 1, 0);
  
  cam.defocus_angle     = 0.0;
  cam.focus_dist        = 10.0;


  camera_initialize(&cam);

  camera_diagnostics(&cam);
  render(world, &cam);  
  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }
}

void cornell_box() {
  hittable* world = hittable_list_create();

  material* red   = mat_lambertian(color_create(.65, .05, .05));
  material* white = mat_lambertian(color_create(.73, .73, .73));
  material* green = mat_lambertian(color_create(.12, .45, .15));
  material* light = diffuse_light_create_color(color_create(15, 15, 15));

  hittable_list_add(world, quad_create(vec3_create(555,0,0), vec3_create(0,555,0), vec3_create(0,0,555), green));
  hittable_list_add(world, quad_create(vec3_create(0,0,0), vec3_create(0,555,0), vec3_create(0,0,555), red));
  hittable_list_add(world, quad_create(vec3_create(343, 554, 332), vec3_create(-130,0,0), vec3_create(0,0,-105), light));
  hittable_list_add(world, quad_create(vec3_create(0,0,0), vec3_create(555,0,0), vec3_create(0,0,555), white));
  hittable_list_add(world, quad_create(vec3_create(555,555,555), vec3_create(-555,0,0), vec3_create(0,0,-555), white));
  hittable_list_add(world, quad_create(vec3_create(0,0,555), vec3_create(555,0,0), vec3_create(0,555,0), white));

  hittable* box1 = create_box(vec3_create(0,0,0), vec3_create(165,330,165), white);
  box1 = translate_obj(box1, vec3_create(265, 0, 295));
  hittable_list_add(world, box1);

  hittable* box2 = create_box(vec3_create(0,0,0), vec3_create(165,165,165), white);  
  box2 = translate_obj(box2, vec3_create(130, 0, 65));
  hittable_list_add(world, box2);

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
  cam.aspect_ratio      = 1;
  cam.image_width       = 600;
  cam.samples_per_pixel = 200;
  cam.max_depth         = 50;
  cam.background        = color_create(0, 0, 0);

  cam.vfov              = 40;
  cam.lookfrom          = vec3_create(278, 278, -800);
  cam.lookat            = vec3_create(278, 278, 0);
  cam.vup               = vec3_create(0, 1, 0);
  
  cam.defocus_angle     = 0.0;
  cam.focus_dist        = 10.0;


  camera_initialize(&cam);

  camera_diagnostics(&cam);
  render(world, &cam);  
  if (open_file("./image.ppm") == 1) {
    fprintf(stderr, "Could not find picture in path");
  }
}

int main() {
  switch (7) {
    case 1: bouncing_spheres(); break;
    case 2: checkered_spheres(); break;
    case 3: earth(); break;
    case 4: perlin_spheres(); break;
    case 5: quads(); break;
    case 6: simple_light(); break;
    case 7: cornell_box(); break;
  }
}
