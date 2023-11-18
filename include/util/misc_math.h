#pragma once

#ifdef __CUDACC__
#define MISC_SYMS __host__ __device__ inline
#else
#define MISC_SYMS inline
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