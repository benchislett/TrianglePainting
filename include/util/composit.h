#pragma once

#include "dna.h"

MISC_SYMS float3 color_pixel_blend(float u, float v, const DNAT& dna) {
    float4 rgba = make_float4(1.f, 1.f, 1.f, 1.f);

    for (int pidx = 0; pidx < NPoly; pidx++) {
        PrimT primitive = dna.primitives[pidx];
        if (primitive.poly.test(u, v)) {
            rgba = over(make_float4(primitive.r, primitive.g, primitive.b, 0.5f), rgba);
        }
    }

    return make_float3(rgba.x, rgba.y, rgba.z);
}

MISC_SYMS float3 color_pixel_layer(float u, float v, const DNAT& dna) {
    float3 rgb;

    for (int pidx = 0; pidx < NPoly; pidx++) {
        PrimT primitive = dna.primitives[pidx];
        if (primitive.poly.test(u, v)) {
            rgb = make_float3(primitive.r, primitive.g, primitive.b);
        }
    }

    return rgb;
}
