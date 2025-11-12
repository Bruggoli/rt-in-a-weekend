#include "camera.h"
#include "material.h"
#include "hittable.h"
#include "color.h"
#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"
#include "camera_cuda.h"
#include "scene_converter.h"
#include "bvh.h"
#include "denoiser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function prototypes for optimized rendering
extern void cuda_render_optimized(
    vec3* host_image_buffer,
    float3* host_albedo_buffer,
    float3* host_normal_buffer,
    void* host_world,
    camera* cam,
    CudaSphere* host_spheres,
    int num_spheres,
    CudaMaterial* host_materials,
    int num_materials,
    BVHNode* host_bvh_nodes,
    int num_bvh_nodes
);


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

  printf("P3\n%d %d\n255\n", cam->image_width, cam->image_height);

  int num_pixels = cam->image_width * cam->image_height;

  // Allocate buffers for rendering
  color* image_buffer = malloc(num_pixels * sizeof(color));

  // Allocate denoiser buffers if OIDN is available
  typedef struct { float e[3]; } float3;
  float3* albedo_buffer = NULL;
  float3* normal_buffer = NULL;

  int use_denoiser = is_oidn_available();
  if (use_denoiser) {
    albedo_buffer = (float3*)malloc(num_pixels * sizeof(float3));
    normal_buffer = (float3*)malloc(num_pixels * sizeof(float3));
    fprintf(stderr, "AI Denoising: ENABLED\n");
  } else {
    fprintf(stderr, "AI Denoising: Not available (render with more samples for quality)\n");
  }

  // Convert scene to CUDA-compatible format
  CudaSphere* spheres;
  CudaMaterial* materials;
  int num_spheres, num_materials;

  fprintf(stderr, "Converting scene to CUDA format...\n");
  convert_scene_to_cuda(world, &spheres, &num_spheres, &materials, &num_materials);

  // Build BVH acceleration structure
  int num_bvh_nodes;
  BVHNode* bvh_nodes = build_bvh(spheres, num_spheres, &num_bvh_nodes);

  // Run optimized CUDA rendering with BVH
  fprintf(stderr, "Launching ultra-optimized CUDA renderer...\n");
  cuda_render_optimized(image_buffer, albedo_buffer, normal_buffer, world, cam,
                        spheres, num_spheres, materials, num_materials,
                        bvh_nodes, num_bvh_nodes);

  // Apply denoising if available
  if (use_denoiser) {
    // Convert float3 buffers to vec3 for denoiser
    vec3* albedo_vec3 = (vec3*)malloc(num_pixels * sizeof(vec3));
    vec3* normal_vec3 = (vec3*)malloc(num_pixels * sizeof(vec3));

    for (int i = 0; i < num_pixels; i++) {
      albedo_vec3[i].e[0] = albedo_buffer[i].e[0];
      albedo_vec3[i].e[1] = albedo_buffer[i].e[1];
      albedo_vec3[i].e[2] = albedo_buffer[i].e[2];

      normal_vec3[i].e[0] = normal_buffer[i].e[0];
      normal_vec3[i].e[1] = normal_buffer[i].e[1];
      normal_vec3[i].e[2] = normal_buffer[i].e[2];
    }

    denoise_image(image_buffer, albedo_vec3, normal_vec3, cam->image_width, cam->image_height);

    free(albedo_vec3);
    free(normal_vec3);
  }

  // Free BVH and converted scene data
  free_bvh(bvh_nodes);
  free_cuda_scene(spheres, materials);

  if (albedo_buffer) free(albedo_buffer);
  if (normal_buffer) free(normal_buffer);

  // Sequential write to maintain PPM format
  fprintf(stderr, "Writing image...\n");
  for (int j = 0; j < cam->image_height; j++) {
    for (int i = 0; i < cam->image_width; i++) {
      write_color(stdout, image_buffer[j * cam->image_width + i]);
    }
  }

  free(image_buffer);
  fprintf(stderr, "\rDone.\n");
}


void camera_initialize(camera* c) {

  // Calculate image height and ensure > 1
  fprintf(stderr, "%d\n", c->image_height);
  fprintf(stderr, "%d\n", c->image_width);
  int image_height = (int)(c->image_width / c->aspect_ratio);
  c->image_height = (image_height < 1) ? 1 : image_height;
  c->pixel_samples_scale = 1.0 / c->samples_per_pixel;
  c->center = c->lookfrom;
  fprintf(stderr, "%d\n", c->image_height);
  fprintf(stderr, "%d\n", c->image_width);
  // Determine viewport dimensions.
  double focal_length = vec3_length(vec3_sub(c->lookfrom, c->lookat));
  double theta = degrees_to_radians(c->vfov);  // <-- c->vfov
  double h = tan(theta / 2);
  double viewport_height = 2.0 * h * c->focus_dist;
  double viewport_width = viewport_height * (double)c->image_width / c->image_height;  // <-- c->image_width
  point3 camera_center = c->center;

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
  
  // calculate the position of the top left pixel
  // temp calc to avoid nesting 1
  vec3 upper_left_calc_1= vec3_sub(camera_center, vec3_create(0, 0, focus_dist));
  // temp calc 2
  vec3 upper_left_calc_2 = vec3_add(vec3_div(viewport_u, 2), vec3_div(viewport_v, 2));
  
  
  vec3 pixel_delta_scaled = vec3_scale(vec3_add(c->pixel_delta_u, c->pixel_delta_v), 0.5);
  c->pixel00_loc = vec3_add(viewport_upper_left, pixel_delta_scaled);

  double defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
  c->defocus_disk_u = vec3_scale(c->u, defocus_radius);
  c->defocus_disk_v = vec3_scale(c->v, defocus_radius);
  fprintf(stderr, "Camera initialised!");

}

color ray_color(ray r, int depth, hittable* world) {
  if (depth <= 0)
    return vec3_create(0, 0, 0);

  hit_record rec;

  if (world->hit(world, r, interval_create(0.001, INFINITY), &rec)) {
    ray scattered;
    color attenuation;
    if (rec.mat->scatter(rec.mat, r, &rec, &attenuation, &scattered)){
      return vec3_mul(attenuation, ray_color(scattered, depth - 1, world));
    }
    return vec3_create(0, 0, 0);
  }
  

  // Sky gradient
  vec3 unit_direction = unit_vector(r.dir);
  double a = 0.5 * (unit_direction.e[1] + 1.0);
  return 
    vec3_add(
      vec3_scale(vec3_create(1.0, 1.0, 1.0), (1.0-a)), 
      vec3_scale(vec3_create(0.5, 0.7, 1.0), a)
    );
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

  return ray_create(c->center, vec3_sub(pixel_sample, c->center));
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
