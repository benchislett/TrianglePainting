#include "evaluate_naive.h"

#include "misc_math.h"

float* tri_render_cpu(const DNATri50& dna) {
    float* image_out = (float*) malloc (resolution * resolution * sizeof(float) * 3);

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            float u = (float) i / (float) resolution;
            float v = (float) j / (float) resolution;

            float3 rgb = color_pixel_blend(u, v, dna);

            image_out[3 * (i * resolution + j) + 0] = rgb.x;
            image_out[3 * (i * resolution + j) + 1] = rgb.y;
            image_out[3 * (i * resolution + j) + 2] = rgb.z;
        }
    }

    return image_out;
}

float tri_loss_cpu(const DNATri50& dna, const float *target_image) {
    float total = 0.f;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            int idx = i * resolution + j;

            float u = (float) i / (float) resolution;
            float v = (float) j / (float) resolution;

            float3 rgb = color_pixel_blend(u, v, dna);
            float3 target_rgb = make_float3(target_image[3 * idx], target_image[3 * idx + 1], target_image[3 * idx + 2]);

            total += abs_error(rgb, target_rgb);
        }
    }

    return total;
}
