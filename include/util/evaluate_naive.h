#pragma once

#include "dna.h"

#include <cstdint>
#include <vector>

float* tri_render_cpu(const DNATri50&);

float tri_loss_cpu(const DNATri50&, const float *target_image);
