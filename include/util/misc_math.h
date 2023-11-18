#pragma once

#ifdef __CUDACC__
#define MISC_SYMS __host__ __device__ inline
#else
#define MISC_SYMS inline

struct float3 {
    float x, y, z;
};
struct float4 {
    float x, y, z, w;
};

float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

#include <cmath>
#endif

MISC_SYMS float sign(float p1x, float p1y, float p2x, float p2y, float p3x, float p3y)
{
    return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y);
}

MISC_SYMS bool point_in_tri(float u, float v, float p1x, float p1y, float p2x, float p2y, float p3x, float p3y)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(u, v, p1x, p1y, p2x, p2y);
    d2 = sign(u, v, p2x, p2y, p3x, p3y);
    d3 = sign(u, v, p3x, p3y, p1x, p1y);

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0);
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0);

    return !(has_neg & has_pos);
}

MISC_SYMS float4 over(float4 src, float4 background) {
    float a_new = src.w + background.w * (1.f - src.w);
    float pixel_r = (src.x * src.w + background.w * background.x * (1.f - src.w)) / a_new;
    float pixel_g = (src.y * src.w + background.w * background.y * (1.f - src.w)) / a_new;
    float pixel_b = (src.z * src.w + background.w * background.z * (1.f - src.w)) / a_new;
    return make_float4(pixel_r, pixel_g, pixel_b, a_new);
}

MISC_SYMS float3 color_pixel_blend(float u, float v, auto dna) {
    float4 pixel_rgba = make_float4(1.f, 1.f, 1.f, 1.f);

    for (int poly = 0; poly < 50; poly++) {
        auto [v1, v2, v3] = dna.polys[poly].vertices;
        if (v1.first == v2.first && v1.second == v2.second) continue;

        if (point_in_tri(u, v, v1.first, v1.second, v2.first, v2.second, v3.first, v3.second)) {
            pixel_rgba = over(make_float4(dna.polys[poly].r, dna.polys[poly].g, dna.polys[poly].b, 0.5f), pixel_rgba);
        }
    }

    return make_float3(pixel_rgba.x, pixel_rgba.y, pixel_rgba.z);
}

MISC_SYMS float3 color_pixel_layer(float u, float v, auto dna) {
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

    return make_float3(pixel_r, pixel_g, pixel_b);
}

MISC_SYMS float abs_error(float actual, float target) {
    float diff = target - actual;
    return fabsf(diff);
}

MISC_SYMS float abs_error(float3 actual, float3 target) {
    return abs_error(actual.x, target.x) + abs_error(actual.y, target.y) + abs_error(actual.z, target.z);
}