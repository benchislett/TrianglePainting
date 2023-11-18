#pragma once

#include "dna.h"

#include <cstdint>
#include <vector>

float* tri_render_cpu(const DNATri50& dna);

float tri_loss_cpu(const DNATri50& dna, const float *target_image);
