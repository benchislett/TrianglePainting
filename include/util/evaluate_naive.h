#pragma once

#include "dna.h"

float* tri_render_cpu(const DNAT&);

struct LossStateCPU {
    LossStateCPU(const float* target_image);
    float loss(const DNAT&);

    float* target;
};
