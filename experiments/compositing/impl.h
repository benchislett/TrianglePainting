#pragma once

#include <immintrin.h>

inline void blend_RGBAF32_over_RGBF32_premultiplied_scalar(
    float fg_r, float fg_g, float fg_b, float fg_a,
    float bg_r, float bg_g, float bg_b,
    float& out_r, float& out_g, float& out_b
) {
    out_r = (fg_r + (bg_r * (1 - fg_a)));
    out_g = (fg_g + (bg_g * (1 - fg_a)));
    out_b = (fg_b + (bg_b * (1 - fg_a)));
}

inline void blend_RGBAU8_over_RGBU8_premultiplied_scalar(
    unsigned char fg_r, unsigned char fg_g, unsigned char fg_b, unsigned char fg_a,
    unsigned char bg_r, unsigned char bg_g, unsigned char bg_b,
    unsigned char& out_r, unsigned char& out_g, unsigned char& out_b
) {
    out_r = (fg_r + (bg_r * (255 - fg_a)) / 255);
    out_g = (fg_g + (bg_g * (255 - fg_a)) / 255);
    out_b = (fg_b + (bg_b * (255 - fg_a)) / 255);
}

#ifndef INT_PRELERP
#define INT_MULT(a,b,t) ( (t) = (a) * (b) + 0x80, ( ( ( (t)>>8 ) + (t) )>>8 ) )
#define INT_PRELERP(p, q, a, t) ( (p) + (q) - INT_MULT( a, p, t) )
#endif

inline void blend_RGBAU8_over_RGBU8_premultiplied_approx(
    unsigned char fg_r, unsigned char fg_g, unsigned char fg_b, unsigned char fg_a,
    unsigned char bg_r, unsigned char bg_g, unsigned char bg_b,
    unsigned char& out_r, unsigned char& out_g, unsigned char& out_b
) {
    unsigned int t;
    out_r = INT_PRELERP(bg_r, fg_r, fg_a, t);
    out_g = INT_PRELERP(bg_g, fg_g, fg_a, t);
    out_b = INT_PRELERP(bg_b, fg_b, fg_a, t);
}

inline __m128i DivideI32x4By255Approx(__m128i value)
{
    return _mm_srli_epi32(_mm_add_epi32(
        _mm_add_epi32(value, _mm_set1_epi32(1)), _mm_srli_epi32(value, 8)), 8);
}

// Source: https://arxiv.org/pdf/2202.02864
inline __m128i DivideI16x8By255Approx(__m128i value)
{
    value = _mm_add_epi16(value, _mm_set1_epi16(0x80U));
    value = _mm_add_epi16(value, _mm_srli_epi16(value, 8));
    return _mm_srli_epi16(value, 8);
}

inline __m256i DivideI16x16By255Approx(__m256i value)
{
    value = _mm256_add_epi16(value, _mm256_set1_epi16(0x80U));
    value = _mm256_add_epi16(value, _mm256_srli_epi16(value, 8));
    return _mm256_srli_epi16(value, 8);
}

inline void blend_RGBAU8_over_RGBU8_premultiplied_sse(
    unsigned char fg_r, unsigned char fg_g, unsigned char fg_b, unsigned char fg_a,
    unsigned char bg_r, unsigned char bg_g, unsigned char bg_b,
    unsigned char& out_r, unsigned char& out_g, unsigned char& out_b
) {
    __m128i fg = _mm_set_epi32(0, fg_r, fg_g, fg_b);
    __m128i bg = _mm_set_epi32(0, bg_r, bg_g, bg_b);
    __m128i fg_1minus_a = _mm_set1_epi32(255 - fg_a);

    __m128i x = _mm_add_epi32(fg, DivideI32x4By255Approx(_mm_mullo_epi32(bg, fg_1minus_a)));

    out_r = (unsigned char)_mm_extract_epi32(x, 2);
    out_g = (unsigned char)_mm_extract_epi32(x, 1);
    out_b = (unsigned char)_mm_extract_epi32(x, 0);
}

inline void blend_RGBAU32x4_over_RGBU32x4_premultiplied_sse(
    __m128i fg_r, __m128i fg_g, __m128i fg_b, __m128i fg_a,
    __m128i bg_r, __m128i bg_g, __m128i bg_b,
    __m128i& out_r, __m128i& out_g, __m128i& out_b
) {
    __m128i fg_1minus_a = _mm_sub_epi32(_mm_set1_epi32(255), fg_a);
    out_r = _mm_add_epi32(fg_r, DivideI32x4By255Approx(_mm_mullo_epi32(bg_r, fg_1minus_a)));
    out_g = _mm_add_epi32(fg_g, DivideI32x4By255Approx(_mm_mullo_epi32(bg_g, fg_1minus_a)));
    out_b = _mm_add_epi32(fg_b, DivideI32x4By255Approx(_mm_mullo_epi32(bg_b, fg_1minus_a)));
}

inline void blend_RGBAU16x8_over_RGBU16x8_premultiplied_sse(
    __m128i fg_r, __m128i fg_g, __m128i fg_b, __m128i fg_a,
    __m128i bg_r, __m128i bg_g, __m128i bg_b,
    __m128i& out_r, __m128i& out_g, __m128i& out_b
) {
    __m128i fg_1minus_a = _mm_sub_epi16(_mm_set1_epi16(255), fg_a);
    out_r = _mm_add_epi16(fg_r, DivideI16x8By255Approx(_mm_mullo_epi16(bg_r, fg_1minus_a)));
    out_g = _mm_add_epi16(fg_g, DivideI16x8By255Approx(_mm_mullo_epi16(bg_g, fg_1minus_a)));
    out_b = _mm_add_epi16(fg_b, DivideI16x8By255Approx(_mm_mullo_epi16(bg_b, fg_1minus_a)));
}

inline void blend_RGBAU16x16_over_RGBU16x16_premultiplied_avx(
    __m256i fg_r, __m256i fg_g, __m256i fg_b, __m256i fg_a,
    __m256i bg_r, __m256i bg_g, __m256i bg_b,
    __m256i& out_r, __m256i& out_g, __m256i& out_b
) {
    __m256i fg_1minus_a = _mm256_sub_epi16(_mm256_set1_epi16(255), fg_a);
    out_r = _mm256_add_epi16(fg_r, DivideI16x16By255Approx(_mm256_mullo_epi16(bg_r, fg_1minus_a)));
    out_g = _mm256_add_epi16(fg_g, DivideI16x16By255Approx(_mm256_mullo_epi16(bg_g, fg_1minus_a)));
    out_b = _mm256_add_epi16(fg_b, DivideI16x16By255Approx(_mm256_mullo_epi16(bg_b, fg_1minus_a)));
}
