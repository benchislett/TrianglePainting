#pragma once

#include "util_cuda.h"

MISC_SYMS float sign(float p1x, float p1y, float p2x, float p2y, float p3x, float p3y)
{
    return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y);
}

/* https://wrfranklin.org/Research/Short_Notes/pnpoly.html */
template<int nvert>
MISC_SYMS bool pnpoly(float testx, float testy, float *vertx, float *verty)
{
  int i, j, c = 0;
  for (i = 0, j = nvert-1; i < nvert; j = i++) {
    if ( ((verty[i]>testy) != (verty[j]>testy)) &&
	 (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]) )
       c = !c;
  }
  return c;
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

MISC_SYMS float abs_error(float actual, float target) {
    float diff = target - actual;
    return fabsf(diff);
}

MISC_SYMS float abs_error(float3 actual, float3 target) {
    return abs_error(actual.x, target.x) + abs_error(actual.y, target.y) + abs_error(actual.z, target.z);
}