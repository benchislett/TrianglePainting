#include "evaluate_naive.h"

#include "misc_math.h"

float* tri_render_cpu(const DNATri50& dna) {
    float* image_out = (float*) malloc (resolution * resolution * sizeof(float) * 3);

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            float u = (float) i / (float) resolution;
            float v = (float) j / (float) resolution;

            float pixel_r = 0.f;
            float pixel_g = 0.f;
            float pixel_b = 0.f;

            for (int poly = 0; poly < 50; poly++) {
                auto [v1, v2, v3] = dna.polys[poly].vertices;
                if (point_in_tri(u, v, v1.first, v1.second, v2.first, v2.second, v3.first, v3.second)) {
                    pixel_r = dna.polys[poly].r;
                    pixel_g = dna.polys[poly].g;
                    pixel_b = dna.polys[poly].b;
                }
            }

            image_out[3 * (i * resolution + j) + 0] = pixel_r;
            image_out[3 * (i * resolution + j) + 1] = pixel_g;
            image_out[3 * (i * resolution + j) + 2] = pixel_b;
        }
    }

    return image_out;
}

float tri_loss_cpu(const DNATri50& dna, const float *target_image) {
    float tse = 0.f;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            int idx = i * resolution + j;

            float u = (float) i / (float) resolution;
            float v = (float) j / (float) resolution;

            float pixel_r = 0.f;
            float pixel_g = 0.f;
            float pixel_b = 0.f;

            for (int poly = 0; poly < 50; poly++) {
                auto [v1, v2, v3] = dna.polys[poly].vertices;
                if (point_in_tri(u, v, v1.first, v1.second, v2.first, v2.second, v3.first, v3.second)) {
                    pixel_r = dna.polys[poly].r;
                    pixel_g = dna.polys[poly].g;
                    pixel_b = dna.polys[poly].b;
                }
            }

            float delta_r = pixel_r - target_image[3 * idx + 0];
            float delta_g = pixel_g - target_image[3 * idx + 1];
            float delta_b = pixel_b - target_image[3 * idx + 2];

            tse += delta_r * delta_r + delta_g * delta_g + delta_b * delta_b;
        }
    }

    return tse;
}
