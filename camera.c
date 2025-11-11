#include "camera.h"
#include "material.h"
#include "hittable.h"
#include "color.h"
#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"

double  aspect_ratio      = 16.0 / 9.0;
int     image_width       = 400;
int     samples_per_pixel = 10;
int     max_depth         = 10;

double  vfov              = 90;
point3  lookfrom          = (vec3){0, 0, 0};
point3  lookat            = (vec3){0, 0, -1};
point3  vup               = (vec3){0, 1, 0};


void render(hittable* world) {
  
  camera camera = camera_initialize();
  
  printf("P3\n%d %d\n255\n", image_width, camera.image_height);

  for (int j = 0; j < camera.image_height; j++) {
    fprintf(stderr, "\rScanlines remaining: %d", (camera.image_height - j));
    fflush(stderr);
    for (int i = 0; i < image_width; i++) {
      color pixel_color = vec3_create(0, 0, 0);
      for (int sample = 0; sample < samples_per_pixel; sample++) {
        ray r = camera_get_ray(&camera, i, j);
        pixel_color = vec3_add(pixel_color, ray_color(r, max_depth, world));
      }
    write_color(stdout, vec3_scale( pixel_color, camera.pixel_samples_scale));
    }
  }

  fprintf(stderr, "\r                                \n");
  fprintf(stderr, "\rDone.\n");
}

camera camera_initialize() {
  
  camera c;
  
  c.aspect_ratio = aspect_ratio;

  // Calculate image height and ensure > 1
  int image_height = (int)(image_width / aspect_ratio);
  c.image_height = (image_height < 1) ? 1 : image_height;

  c.pixel_samples_scale = 1.0 / samples_per_pixel;

  c.center = c.lookfrom;
  
  // Determine viewport dimensions.
  double focal_length = vec3_length(vec3_sub(c.lookfrom, c.lookat));
  double theta = degrees_to_radians(vfov);
  double h = tan(theta / 2);
  double viewport_height = 2.0 * h * focal_length;
  double viewport_width = viewport_height * (double)image_width/image_height;
  point3 camera_center = vec3_create(0, 0, 0);

  // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
  c.w = unit_vector(vec3_sub(c.lookfrom, c.lookat));
  c.u = unit_vector(vec3_cross(c.vup, c.w));
  c.v = vec3_cross(c.w, c.u);

  vec3 viewport_u = vec3_scale(c.u, viewport_width);
  vec3 viewport_v = vec3_scale(vec3_negate(c.v), viewport_height);

  c.pixel_delta_u = vec3_div(viewport_u, image_width); 
  c.pixel_delta_v = vec3_div(viewport_v, image_height);

  // calculate the position of the top left pixel
  // temp calc to avoid nesting 1
  vec3 upper_left_calc_1= vec3_sub(camera_center, vec3_create(0, 0, focal_length));
  // temp calc 2
  vec3 upper_left_calc_2 = vec3_add(vec3_div(viewport_u, 2), vec3_div(viewport_v, 2));

  // auto viewport_upper_left = center - (focal_length * w) - viewport_u/2 - viewport_v/2;
  point3 p0 = vec3_sub(c.center, vec3_scale(c.w, focal_length));
  p0 = vec3_sub(p0, vec3_div(viewport_u, 2.0));
  p0 = vec3_sub(p0, vec3_div(viewport_u, 2.0));
  point3 viewport_upper_left = p0;

  vec3 pixel_delta_scaled = vec3_scale(vec3_add(c.pixel_delta_u, c.pixel_delta_v), 0.5);
  c.pixel00_loc = vec3_add(viewport_upper_left, pixel_delta_scaled);

  return c;
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

ray camera_get_ray(camera* c, double i, double j) {
  vec3 offset = ray_sample_square();
  vec3 pixel_sample = vec3_add(
    c->pixel00_loc, 
    vec3_add(
      vec3_scale(c->pixel_delta_u, (i + offset.e[0])),
      vec3_scale(c->pixel_delta_v, (j + offset.e[1]))
      )
  );
  return ray_create(c->center, vec3_sub(pixel_sample, c->center));
  
}

vec3 ray_sample_square() {
  return vec3_create(random_double() - 0.5, random_double() - 0.5, 0);
}
