#pragma once

#include "dna.h"

float* tri_render_gpu(const DNAT&);

struct LossStateGPU {
    LossStateGPU(const float* target_image);
    float loss(const DNAT&);

    float* target;
    float* error_values;
};
