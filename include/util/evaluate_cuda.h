#pragma once

#include "dna.h"

struct LossState {
    float* target_image;
    float* error_values;

    void init();
};

float* tri_render_gpu(const DNATri50&);

float tri_loss_gpu(const DNATri50&, const float *target_image, LossState&);
