#include <stdio.h>

#include "vec3.h"
#include "color.h"
#include "ray.h"

color ray_color(ray r) {
  return vec3_create(0, 0, 0);
}

int main() {
  double aspect_ratio = 16.0 / 9.0;
  int image_width = 400;

  int image_height = (int)(image_width / aspect_ratio);
  image_height = (image_height < 1) ? 1 : image_height;

  double viewport_height = 2.0;
  double viewport_width = viewport_height * (double)image_width/image_height;

  printf("P3\n%d %d\n255\n", image_width, image_height);

  for (int j = 0; j < image_height; j++) {
    fprintf(stderr, "\rScanlines remaining: %d", (image_height - j));
    fflush(stderr);
    for (int i = 0; i < image_width; i++) {
      color pixel_color = vec3_create((double)i / (image_width), (double)j / image_height, 0); 
      write_color(stdout, pixel_color);
    }
  }
  fprintf(stderr, "\rDone.\n");
}
