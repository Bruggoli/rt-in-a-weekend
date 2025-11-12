#include "denoiser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Intel OIDN support (optional - only if library is available)
#ifdef HAVE_OIDN
#include <OpenImageDenoise/oidn.hpp>
#endif

extern "C" {

int is_oidn_available() {
#ifdef HAVE_OIDN
    return 1;
#else
    return 0;
#endif
}

int denoise_image(
    vec3* color_buffer,
    vec3* albedo_buffer,
    vec3* normal_buffer,
    int width,
    int height
) {
#ifdef HAVE_OIDN
    fprintf(stderr, "\n=== INTEL OIDN DENOISER ===\n");
    fprintf(stderr, "Image: %dx%d\n", width, height);
    fprintf(stderr, "Using albedo buffer: %s\n", albedo_buffer ? "yes" : "no");
    fprintf(stderr, "Using normal buffer: %s\n", normal_buffer ? "yes" : "no");

    try {
        // Create OIDN device
        oidn::DeviceRef device = oidn::newDevice();
        device.commit();

        const char* errorMessage;
        if (device.getError(errorMessage) != oidn::Error::None) {
            fprintf(stderr, "OIDN Error: %s\n", errorMessage);
            return 1;
        }

        int num_pixels = width * height;

        // Convert vec3 (double) to float for OIDN
        float* color_float = (float*)malloc(num_pixels * 3 * sizeof(float));
        float* albedo_float = albedo_buffer ? (float*)malloc(num_pixels * 3 * sizeof(float)) : nullptr;
        float* normal_float = normal_buffer ? (float*)malloc(num_pixels * 3 * sizeof(float)) : nullptr;
        float* output_float = (float*)malloc(num_pixels * 3 * sizeof(float));

        // Convert color buffer
        for (int i = 0; i < num_pixels; i++) {
            color_float[i * 3 + 0] = (float)color_buffer[i].e[0];
            color_float[i * 3 + 1] = (float)color_buffer[i].e[1];
            color_float[i * 3 + 2] = (float)color_buffer[i].e[2];
        }

        // Convert albedo buffer (if provided)
        if (albedo_buffer && albedo_float) {
            for (int i = 0; i < num_pixels; i++) {
                albedo_float[i * 3 + 0] = (float)albedo_buffer[i].e[0];
                albedo_float[i * 3 + 1] = (float)albedo_buffer[i].e[1];
                albedo_float[i * 3 + 2] = (float)albedo_buffer[i].e[2];
            }
        }

        // Convert normal buffer (if provided)
        if (normal_buffer && normal_float) {
            for (int i = 0; i < num_pixels; i++) {
                normal_float[i * 3 + 0] = (float)normal_buffer[i].e[0];
                normal_float[i * 3 + 1] = (float)normal_buffer[i].e[1];
                normal_float[i * 3 + 2] = (float)normal_buffer[i].e[2];
            }
        }

        // Create filter
        oidn::FilterRef filter = device.newFilter("RT"); // RTLightmap for path tracing

        // Set filter parameters
        filter.setImage("color", color_float, oidn::Format::Float3, width, height);
        filter.setImage("output", output_float, oidn::Format::Float3, width, height);

        if (albedo_float) {
            filter.setImage("albedo", albedo_float, oidn::Format::Float3, width, height);
        }

        if (normal_float) {
            filter.setImage("normal", normal_float, oidn::Format::Float3, width, height);
        }

        filter.set("hdr", true); // HDR images
        filter.commit();

        // Execute denoising
        fprintf(stderr, "Denoising...\n");
        filter.execute();

        // Check for errors
        if (device.getError(errorMessage) != oidn::Error::None) {
            fprintf(stderr, "OIDN Error: %s\n", errorMessage);
            free(color_float);
            free(output_float);
            if (albedo_float) free(albedo_float);
            if (normal_float) free(normal_float);
            return 1;
        }

        // Convert back to vec3
        for (int i = 0; i < num_pixels; i++) {
            color_buffer[i].e[0] = (double)output_float[i * 3 + 0];
            color_buffer[i].e[1] = (double)output_float[i * 3 + 1];
            color_buffer[i].e[2] = (double)output_float[i * 3 + 2];
        }

        fprintf(stderr, "✓ Denoising complete\n");
        fprintf(stderr, "=========================\n\n");

        // Cleanup
        free(color_float);
        free(output_float);
        if (albedo_float) free(albedo_float);
        if (normal_float) free(normal_float);

        return 0;

    } catch (const std::exception& e) {
        fprintf(stderr, "OIDN Exception: %s\n", e.what());
        return 1;
    }

#else
    fprintf(stderr, "WARNING: OIDN not available - skipping denoising\n");
    fprintf(stderr, "To enable: Install Intel OIDN and rebuild with -DHAVE_OIDN\n");
    return 0; // Not an error, just not available
#endif
}

} // extern "C"
