#ifndef DENOISER_H
#define DENOISER_H

#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

// Denoise rendered image using Intel Open Image Denoise
// Requires: color buffer (required)
// Optional: albedo buffer, normal buffer (improve quality significantly)
// Returns: 0 on success, non-zero on error
int denoise_image(
    vec3* color_buffer,      // Input/output: noisy image (will be replaced with denoised)
    vec3* albedo_buffer,     // Optional: albedo/basecolor buffer (can be NULL)
    vec3* normal_buffer,     // Optional: normal buffer (can be NULL)
    int width,
    int height
);

// Check if OIDN is available
int is_oidn_available();

#ifdef __cplusplus
}
#endif

#endif // DENOISER_H
