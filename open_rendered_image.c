#include "open_rendered_image.h"
#include <stdio.h>
#include <stdlib.h>

int open_file(char *path) {
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "xdg-open %s", path);
  return system(cmd) == 1;
}
