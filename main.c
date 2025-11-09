#include <stdio.h>

#include "vec3.h"
#include "color.h"
#include "ray.h"

color ray_color(ray r) {
  vec3 unit_direction = unit_vector(r.dir);
  fprintf(stderr,"unit_direction: %f\n", unit_direction.e[1]);
  double a = 0.5 * (unit_direction.e[1] + 1.0);
  return 
    vec3_add(
      vec3_scale(vec3_create(1.0, 1.0, 1.0), (1.0-a)), 
      vec3_scale(vec3_create(0.5, 0.7, 1.0), a)
    );
}

int main() {

  // Image

  double aspect_ratio = 16.0 / 9.0;
  int image_width = 400;

  // Calculate image height and ensure > 1
  int image_height = (int)(image_width / aspect_ratio);
  image_height = (image_height < 1) ? 1 : image_height;

  // Camera

  double focal_length = 1.0;
  double viewport_height = 2.0;
  double viewport_width = viewport_height * (double)image_width/image_height;
  point3 camera_center = vec3_create(0, 0, 0);

  vec3 viewport_u = vec3_create(viewport_width, 0, 0);
  vec3 viewport_v = vec3_create(0, -viewport_height, 0);

  vec3 pixel_delta_u = vec3_div(viewport_u, image_width); 
  vec3 pixel_delta_v = vec3_div(viewport_v, image_height);

  // calculate the position of the top left pixel
  // temp calc to avoid nesting 1
  vec3 upper_left_calc_1= vec3_sub(camera_center, vec3_create(0, 0, focal_length));
  // temp calc 2
  vec3 upper_left_calc_2 = vec3_add(vec3_div(viewport_u, 2), vec3_div(viewport_v, 2));
  point3 viewport_upper_left = vec3_sub(upper_left_calc_1, upper_left_calc_2);

  vec3 pixel_delta_scaled = vec3_scale(vec3_add(pixel_delta_u, pixel_delta_v), 0.5);
  point3 pixel00_loc = vec3_add(viewport_upper_left, pixel_delta_scaled);


  printf("P3\n%d %d\n255\n", image_width, image_height);

  for (int j = 0; j < image_height; j++) {
    fprintf(stderr, "\rScanlines remaining: %d", (image_height - j));
    fflush(stderr);
    for (int i = 0; i < image_width; i++) {
      point3 pixel_center = vec3_add(
                              vec3_add(pixel00_loc, vec3_scale(pixel_delta_u, i)), 
                              vec3_scale(pixel_delta_v, j));
      vec3 ray_direction = vec3_sub(pixel_center, camera_center);
      ray ray = ray_create(camera_center, ray_direction);

      color pixel_color = ray_color(ray);
      write_color(stdout, pixel_color);
    }
  }
  fprintf(stderr, "\rDone.\n");
}
