#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "vec3.h"
#include "ray.h"
#include "camera.h"

// CUDA-compatible structures
struct CudaSphere {
    vec3 center;
    double radius;
    int material_idx;
};

enum MaterialType {
    LAMBERTIAN = 0,
    METAL = 1,
    DIELECTRIC = 2
};

struct CudaMaterial {
    MaterialType type;
    vec3 albedo;
    double fuzz;      // for metal
    double ref_idx;   // for dielectric
};

// Device functions for vector operations
__device__ vec3 d_vec3_create(double x, double y, double z) {
    vec3 v;
    v.e[0] = x; v.e[1] = y; v.e[2] = z;
    return v;
}

__device__ vec3 d_vec3_add(vec3 u, vec3 v) {
    return d_vec3_create(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ vec3 d_vec3_sub(vec3 u, vec3 v) {
    return d_vec3_create(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ vec3 d_vec3_mul(vec3 u, vec3 v) {
    return d_vec3_create(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ vec3 d_vec3_scale(vec3 u, double t) {
    return d_vec3_create(u.e[0] * t, u.e[1] * t, u.e[2] * t);
}

__device__ vec3 d_vec3_div(vec3 u, double t) {
    return d_vec3_scale(u, 1.0 / t);
}

__device__ double d_vec3_dot(vec3 u, vec3 v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__device__ double d_vec3_length_squared(vec3 v) {
    return v.e[0] * v.e[0] + v.e[1] * v.e[1] + v.e[2] * v.e[2];
}

__device__ double d_vec3_length(vec3 v) {
    return sqrt(d_vec3_length_squared(v));
}

__device__ vec3 d_unit_vector(vec3 v) {
    return d_vec3_div(v, d_vec3_length(v));
}

__device__ vec3 d_vec3_cross(vec3 u, vec3 v) {
    return d_vec3_create(
        u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]
    );
}

__device__ vec3 d_reflect(vec3 v, vec3 n) {
    return d_vec3_sub(v, d_vec3_scale(n, 2.0 * d_vec3_dot(v, n)));
}

__device__ vec3 d_refract(vec3 uv, vec3 n, double etai_over_etat) {
    double cos_theta = fmin(d_vec3_dot(d_vec3_scale(uv, -1.0), n), 1.0);
    vec3 r_out_perp = d_vec3_scale(d_vec3_add(uv, d_vec3_scale(n, cos_theta)), etai_over_etat);
    vec3 r_out_parallel = d_vec3_scale(n, -sqrt(fabs(1.0 - d_vec3_length_squared(r_out_perp))));
    return d_vec3_add(r_out_perp, r_out_parallel);
}

__device__ bool d_near_zero(vec3 v) {
    const double s = 1e-8;
    return (fabs(v.e[0]) < s) && (fabs(v.e[1]) < s) && (fabs(v.e[2]) < s);
}

// Random vector generation
__device__ vec3 d_random_vec3(curandState* state) {
    return d_vec3_create(
        curand_uniform_double(state),
        curand_uniform_double(state),
        curand_uniform_double(state)
    );
}

__device__ vec3 d_random_vec3_range(curandState* state, double min, double max) {
    double range = max - min;
    return d_vec3_create(
        min + range * curand_uniform_double(state),
        min + range * curand_uniform_double(state),
        min + range * curand_uniform_double(state)
    );
}

__device__ vec3 d_random_in_unit_sphere(curandState* state) {
    while (true) {
        vec3 p = d_random_vec3_range(state, -1.0, 1.0);
        if (d_vec3_length_squared(p) < 1.0)
            return p;
    }
}

__device__ vec3 d_random_unit_vector(curandState* state) {
    return d_unit_vector(d_random_in_unit_sphere(state));
}

__device__ vec3 d_random_in_unit_disk(curandState* state) {
    while (true) {
        vec3 p = d_vec3_create(
            2.0 * curand_uniform_double(state) - 1.0,
            2.0 * curand_uniform_double(state) - 1.0,
            0.0
        );
        if (d_vec3_length_squared(p) < 1.0)
            return p;
    }
}

// Ray operations
__device__ ray d_ray_create(vec3 origin, vec3 direction) {
    ray r;
    r.orig = origin;
    r.dir = direction;
    return r;
}

__device__ vec3 d_ray_at(ray r, double t) {
    return d_vec3_add(r.orig, d_vec3_scale(r.dir, t));
}

// Hit record
struct HitRecord {
    vec3 p;
    vec3 normal;
    double t;
    bool front_face;
    int material_idx;
};

__device__ void d_set_face_normal(HitRecord* rec, ray r, vec3 outward_normal) {
    rec->front_face = d_vec3_dot(r.dir, outward_normal) < 0;
    rec->normal = rec->front_face ? outward_normal : d_vec3_scale(outward_normal, -1.0);
}

// Sphere intersection
__device__ bool d_sphere_hit(
    CudaSphere* sphere,
    ray r,
    double t_min,
    double t_max,
    HitRecord* rec
) {
    vec3 oc = d_vec3_sub(sphere->center, r.orig);
    double a = d_vec3_length_squared(r.dir);
    double h = d_vec3_dot(r.dir, oc);
    double c = d_vec3_length_squared(oc) - sphere->radius * sphere->radius;

    double discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    double sqrtd = sqrt(discriminant);
    double root = (h - sqrtd) / a;

    if (root <= t_min || t_max <= root) {
        root = (h + sqrtd) / a;
        if (root <= t_min || t_max <= root)
            return false;
    }

    rec->t = root;
    rec->p = d_ray_at(r, root);
    vec3 outward_normal = d_vec3_div(d_vec3_sub(rec->p, sphere->center), sphere->radius);
    d_set_face_normal(rec, r, outward_normal);
    rec->material_idx = sphere->material_idx;

    return true;
}

// Scene intersection
__device__ bool d_hit_scene(
    CudaSphere* spheres,
    int num_spheres,
    ray r,
    double t_min,
    double t_max,
    HitRecord* rec
) {
    HitRecord temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (int i = 0; i < num_spheres; i++) {
        if (d_sphere_hit(&spheres[i], r, t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            *rec = temp_rec;
        }
    }

    return hit_anything;
}

// Material scattering
__device__ bool d_scatter(
    CudaMaterial* materials,
    int mat_idx,
    ray r_in,
    HitRecord* rec,
    vec3* attenuation,
    ray* scattered,
    curandState* state
) {
    CudaMaterial mat = materials[mat_idx];

    if (mat.type == LAMBERTIAN) {
        vec3 scatter_direction = d_vec3_add(rec->normal, d_random_unit_vector(state));
        if (d_near_zero(scatter_direction))
            scatter_direction = rec->normal;
        *scattered = d_ray_create(rec->p, scatter_direction);
        *attenuation = mat.albedo;
        return true;
    }
    else if (mat.type == METAL) {
        vec3 reflected = d_reflect(d_unit_vector(r_in.dir), rec->normal);
        vec3 fuzzed = d_vec3_add(reflected, d_vec3_scale(d_random_unit_vector(state), mat.fuzz));
        *scattered = d_ray_create(rec->p, fuzzed);
        *attenuation = mat.albedo;
        return d_vec3_dot(scattered->dir, rec->normal) > 0;
    }
    else if (mat.type == DIELECTRIC) {
        *attenuation = d_vec3_create(1.0, 1.0, 1.0);
        double ri = rec->front_face ? (1.0 / mat.ref_idx) : mat.ref_idx;

        vec3 unit_direction = d_unit_vector(r_in.dir);
        double cos_theta = fmin(d_vec3_dot(d_vec3_scale(unit_direction, -1.0), rec->normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;

        // Schlick approximation
        double r0 = (1.0 - ri) / (1.0 + ri);
        r0 = r0 * r0;
        double reflectance = r0 + (1.0 - r0) * pow((1.0 - cos_theta), 5.0);

        vec3 direction;
        if (cannot_refract || reflectance > curand_uniform_double(state))
            direction = d_reflect(unit_direction, rec->normal);
        else
            direction = d_refract(unit_direction, rec->normal, ri);

        *scattered = d_ray_create(rec->p, direction);
        return true;
    }

    return false;
}

// Ray color computation
__device__ vec3 d_ray_color(
    ray r,
    CudaSphere* spheres,
    int num_spheres,
    CudaMaterial* materials,
    int max_depth,
    curandState* state
) {
    ray current_ray = r;
    vec3 current_attenuation = d_vec3_create(1.0, 1.0, 1.0);

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (d_hit_scene(spheres, num_spheres, current_ray, 0.001, INFINITY, &rec)) {
            ray scattered;
            vec3 attenuation;

            if (d_scatter(materials, rec.material_idx, current_ray, &rec, &attenuation, &scattered, state)) {
                current_attenuation = d_vec3_mul(current_attenuation, attenuation);
                current_ray = scattered;
            } else {
                return d_vec3_create(0.0, 0.0, 0.0);
            }
        } else {
            // Sky gradient
            vec3 unit_direction = d_unit_vector(current_ray.dir);
            double a = 0.5 * (unit_direction.e[1] + 1.0);
            vec3 sky_color = d_vec3_add(
                d_vec3_scale(d_vec3_create(1.0, 1.0, 1.0), 1.0 - a),
                d_vec3_scale(d_vec3_create(0.5, 0.7, 1.0), a)
            );
            return d_vec3_mul(current_attenuation, sky_color);
        }
    }

    // Exceeded recursion depth
    return d_vec3_create(0.0, 0.0, 0.0);
}

// Initialize random states
__global__ void init_random_states(curandState* states, unsigned long seed, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &states[pixel_index]);
}

// Main render kernel
__global__ void render_kernel(
    vec3* image_buffer,
    CudaSphere* spheres,
    int num_spheres,
    CudaMaterial* materials,
    curandState* rand_states,
    camera cam,
    int samples_per_pixel,
    int max_depth
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= cam.image_width || j >= cam.image_height) return;

    int pixel_index = j * cam.image_width + i;
    curandState local_state = rand_states[pixel_index];

    vec3 pixel_color = d_vec3_create(0.0, 0.0, 0.0);

    for (int sample = 0; sample < samples_per_pixel; sample++) {
        // Get ray with random offset
        double offset_x = curand_uniform_double(&local_state) - 0.5;
        double offset_y = curand_uniform_double(&local_state) - 0.5;

        vec3 pixel_sample = d_vec3_add(
            cam.pixel00_loc,
            d_vec3_add(
                d_vec3_scale(cam.pixel_delta_u, (double)i + offset_x),
                d_vec3_scale(cam.pixel_delta_v, (double)j + offset_y)
            )
        );

        vec3 ray_origin = cam.center;

        // Defocus blur
        if (cam.defocus_angle > 0) {
            vec3 rd = d_vec3_scale(d_random_in_unit_disk(&local_state), 1.0);
            vec3 offset = d_vec3_add(
                d_vec3_scale(cam.defocus_disk_u, rd.e[0]),
                d_vec3_scale(cam.defocus_disk_v, rd.e[1])
            );
            ray_origin = d_vec3_add(cam.center, offset);
        }

        ray r = d_ray_create(ray_origin, d_vec3_sub(pixel_sample, ray_origin));
        pixel_color = d_vec3_add(pixel_color, d_ray_color(r, spheres, num_spheres, materials, max_depth, &local_state));
    }

    rand_states[pixel_index] = local_state;
    image_buffer[pixel_index] = d_vec3_scale(pixel_color, cam.pixel_samples_scale);
}

// C interface for calling CUDA kernel
extern "C" void cuda_render(
    vec3* host_image_buffer,
    void* host_world,
    camera* cam,
    CudaSphere* host_spheres,
    int num_spheres,
    CudaMaterial* host_materials,
    int num_materials
) {
    int width = cam->image_width;
    int height = cam->image_height;
    int num_pixels = width * height;

    // Allocate device memory
    vec3* d_image_buffer;
    CudaSphere* d_spheres;
    CudaMaterial* d_materials;
    curandState* d_rand_states;

    cudaMalloc(&d_image_buffer, num_pixels * sizeof(vec3));
    cudaMalloc(&d_spheres, num_spheres * sizeof(CudaSphere));
    cudaMalloc(&d_materials, num_materials * sizeof(CudaMaterial));
    cudaMalloc(&d_rand_states, num_pixels * sizeof(curandState));

    // Copy data to device
    cudaMemcpy(d_spheres, host_spheres, num_spheres * sizeof(CudaSphere), cudaMemcpyHostToDevice);
    cudaMemcpy(d_materials, host_materials, num_materials * sizeof(CudaMaterial), cudaMemcpyHostToDevice);

    // Initialize random states
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    init_random_states<<<grid_size, block_size>>>(d_rand_states, time(NULL), width, height);
    cudaDeviceSynchronize();

    // Render
    fprintf(stderr, "Launching CUDA kernel with grid (%d, %d) and blocks (%d, %d)\n",
            grid_size.x, grid_size.y, block_size.x, block_size.y);

    render_kernel<<<grid_size, block_size>>>(
        d_image_buffer,
        d_spheres,
        num_spheres,
        d_materials,
        d_rand_states,
        *cam,
        cam->samples_per_pixel,
        cam->max_depth
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(host_image_buffer, d_image_buffer, num_pixels * sizeof(vec3), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image_buffer);
    cudaFree(d_spheres);
    cudaFree(d_materials);
    cudaFree(d_rand_states);
}
