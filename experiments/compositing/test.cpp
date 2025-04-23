#include <gtest/gtest.h>
#include "impl.h"
#include "utils.h"

using BlendFuncU8 = void (*)(
    unsigned char, unsigned char, unsigned char, unsigned char,   // fg
    unsigned char, unsigned char, unsigned char,                  // bg
    unsigned char&, unsigned char&, unsigned char&);              // out

class PremultipliedBlendU8Test : public ::testing::TestWithParam<BlendFuncU8> {};

#define EXPECT_EQ_INT_OFFBY1(a, b) \
    EXPECT_TRUE(int(a) == int(b) || int(a) == int(b) + 1 || int(a) + 1 == int(b))

#define ASSERT_EQ_INT_OFFBY1(a, b) \
    ASSERT_TRUE(int(a) == int(b) || int(a) == int(b) + 1 || int(a) + 1 == int(b))

TEST_P(PremultipliedBlendU8Test, OffByOneBehaviour)
{
    auto blend = GetParam();

    unsigned char fg_r = 82,  fg_g = 61,  fg_b = 0,   fg_a = 120;
    unsigned char bg_r = 0,   bg_g = 201, bg_b = 68;

    float ref_r, ref_g, ref_b;
    blend_RGBAF32_over_RGBF32_premultiplied_scalar(
        fg_r / 255.0f, fg_g / 255.0f, fg_b / 255.0f, fg_a / 255.0f,
        bg_r / 255.0f, bg_g / 255.0f, bg_b / 255.0f,
        ref_r, ref_g, ref_b);

    unsigned char out_r, out_g, out_b;
    blend(fg_r, fg_g, fg_b, fg_a, bg_r, bg_g, bg_b, out_r, out_g, out_b);

    EXPECT_EQ_INT_OFFBY1(out_r, static_cast<unsigned char>(ref_r * 255));
    EXPECT_EQ_INT_OFFBY1(out_g, static_cast<unsigned char>(ref_g * 255));
    EXPECT_EQ_INT_OFFBY1(out_b, static_cast<unsigned char>(ref_b * 255));

    for (int fg_c = 0; fg_c <= 255; ++fg_c) {
        for (int fg_ca = 0; fg_ca <= 255; ++fg_ca) {
            int fc_c_premultiplied = int(float(fg_c) * float(fg_ca) / 255.0f);
            for (int bg_c = 0; bg_c <= 255; ++bg_c) {
                blend_RGBAF32_over_RGBF32_premultiplied_scalar(
                    fc_c_premultiplied / 255.0f,
                    fc_c_premultiplied / 255.0f,
                    fc_c_premultiplied / 255.0f,
                    fg_ca / 255.0f,
                    bg_c / 255.0f,
                    bg_c / 255.0f,
                    bg_c / 255.0f,
                    ref_r, ref_g, ref_b);

                blend(fc_c_premultiplied, fc_c_premultiplied, fc_c_premultiplied,
                      fg_ca,
                      bg_c, bg_c, bg_c,
                      out_r, out_g, out_b);

                ASSERT_EQ_INT_OFFBY1(out_r, static_cast<unsigned char>(ref_r * 255));
                ASSERT_EQ_INT_OFFBY1(out_g, static_cast<unsigned char>(ref_g * 255));
                ASSERT_EQ_INT_OFFBY1(out_b, static_cast<unsigned char>(ref_b * 255));
            }
        }
    }
}

void blend_RGBAU8_over_RGBU8_premultiplied_via_32bitSSEx4(
    unsigned char fg_r, unsigned char fg_g, unsigned char fg_b, unsigned char fg_a,
    unsigned char bg_r, unsigned char bg_g, unsigned char bg_b,
    unsigned char& out_r, unsigned char& out_g, unsigned char& out_b) {
    __m128i out_r_x4, out_g_x4, out_b_x4;
    blend_RGBAU32x4_over_RGBU32x4_premultiplied_sse(
        _mm_set1_epi32(fg_r), _mm_set1_epi32(fg_g), _mm_set1_epi32(fg_b), _mm_set1_epi32(fg_a),
        _mm_set1_epi32(bg_r), _mm_set1_epi32(bg_g), _mm_set1_epi32(bg_b),
        out_r_x4, out_g_x4, out_b_x4);
    out_r = _mm_extract_epi32(out_r_x4, 0);
    out_g = _mm_extract_epi32(out_g_x4, 0);
    out_b = _mm_extract_epi32(out_b_x4, 0);
}

void blend_RGBAU8_over_RGBU8_premultiplied_via_16bitSSEx8(
    unsigned char fg_r, unsigned char fg_g, unsigned char fg_b, unsigned char fg_a,
    unsigned char bg_r, unsigned char bg_g, unsigned char bg_b,
    unsigned char& out_r, unsigned char& out_g, unsigned char& out_b) {
    __m128i out_r_x8, out_g_x8, out_b_x8;
    blend_RGBAU16x8_over_RGBU16x8_premultiplied_sse(
        _mm_set1_epi32(fg_r), _mm_set1_epi32(fg_g), _mm_set1_epi32(fg_b), _mm_set1_epi32(fg_a),
        _mm_set1_epi32(bg_r), _mm_set1_epi32(bg_g), _mm_set1_epi32(bg_b),
        out_r_x8, out_g_x8, out_b_x8);
    out_r = _mm_extract_epi32(out_r_x8, 0);
    out_g = _mm_extract_epi32(out_g_x8, 0);
    out_b = _mm_extract_epi32(out_b_x8, 0);
}

void blend_RGBAU8_over_RGBU8_premultiplied_via_16bitAVXx16(
    unsigned char fg_r, unsigned char fg_g, unsigned char fg_b, unsigned char fg_a,
    unsigned char bg_r, unsigned char bg_g, unsigned char bg_b,
    unsigned char& out_r, unsigned char& out_g, unsigned char& out_b) {
    __m256i out_r_x16, out_g_x16, out_b_x16;
    blend_RGBAU16x16_over_RGBU16x16_premultiplied_avx(
        _mm256_set1_epi32(fg_r), _mm256_set1_epi32(fg_g), _mm256_set1_epi32(fg_b), _mm256_set1_epi32(fg_a),
        _mm256_set1_epi32(bg_r), _mm256_set1_epi32(bg_g), _mm256_set1_epi32(bg_b),
        out_r_x16, out_g_x16, out_b_x16);
    out_r = _mm256_extract_epi32(out_r_x16, 0);
    out_g = _mm256_extract_epi32(out_g_x16, 0);
    out_b = _mm256_extract_epi32(out_b_x16, 0);
}

INSTANTIATE_TEST_SUITE_P(
    CompositingTests,
    PremultipliedBlendU8Test,
    ::testing::Values(
        &blend_RGBAU8_over_RGBU8_premultiplied_scalar,
        &blend_RGBAU8_over_RGBU8_premultiplied_approx,
        &blend_RGBAU8_over_RGBU8_premultiplied_sse,
        &blend_RGBAU8_over_RGBU8_premultiplied_via_32bitSSEx4,
        &blend_RGBAU8_over_RGBU8_premultiplied_via_16bitSSEx8,
        &blend_RGBAU8_over_RGBU8_premultiplied_via_16bitAVXx16),
    [](const ::testing::TestParamInfo<BlendFuncU8>& info) {
        if (info.param == &blend_RGBAU8_over_RGBU8_premultiplied_scalar)
            return "Scalar";
        if (info.param == &blend_RGBAU8_over_RGBU8_premultiplied_approx)
            return "Approx";
        if (info.param == &blend_RGBAU8_over_RGBU8_premultiplied_sse) 
            return "SSE";
        if (info.param == &blend_RGBAU8_over_RGBU8_premultiplied_via_32bitSSEx4)
            return "32bitSSEx4";
        if (info.param == &blend_RGBAU8_over_RGBU8_premultiplied_via_16bitSSEx8)
            return "16bitSSEx8";
        if (info.param == &blend_RGBAU8_over_RGBU8_premultiplied_via_16bitAVXx16)
            return "16bitAVXx16";
        return "Unknown";
    });