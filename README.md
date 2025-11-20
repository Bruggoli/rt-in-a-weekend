Hats off to [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

The instructions are clear enough for me (a complete C beginner) to understand.

I used Claude.ai to start with to help me with the concept of polymorphic C code, translating the book scenes, and some obscure(to me!) pointer bugs.

# C Ray Tracer

Path tracer implementation in C, based on "Ray Tracing in One Weekend" series.

## Features

- BVH acceleration structure for scene traversal
- Polymorphic materials via function pointers
- Lambertian, metal, and dielectric materials
- Positionable camera with depth of field
- PPM output format
- openMP multithreading

## Build
```bash
clang -O3 -o raytracer main.c ./*/*.c -lm -fopenmp
```

## Usage
```bash
./raytracer > output.ppm
```

Sports a makefile, but the current version is made for linux, not adapted to other systems.

Renders default scene (spheres with varied materials). Adjust parameters in `main()`.

**Key parameters:**
- `image_width` / `aspect_ratio` - output resolution
- `samples_per_pixel` - quality vs render time
- `max_depth` - ray bounce limit

## Architecture

**Object system:** Function pointer vtable pattern for polymorphism without C++

**Acceleration:** BVH with surface area heuristic

**Memory:** Stack allocation where possible, manual management for scene graph

## Performance

~11.5 hours on M1 Mac (800x800, 500 samples, max_depth=50)

Potential optimizations:
- SIMD vectorization for ray-AABB tests
- CUDA implementation
- Actually freeing unused memory
- Most likely many more that im not aware of!

## Notes
The project uses free() very rarely, since its a run-once-and-quit program, but looking into doing it properly down the road. The final scene uses around 1gb RAM during the process, but the cpu is pinned while running.

## Project takeaways
Just use c++ for this kind of stuff. The C version feels incredibly verbose compared to the code presented in the ray tracing series.


## Dependencies

Standard C library (`math.h`, `stdlib.h`, `stdio.h`), OpenMP, [stb_image](https://github.com/nothings/stb)
