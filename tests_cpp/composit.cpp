#include <gtest/gtest.h>
#include "composit.h"

TEST(CompositTest, RGBAF32OverRGBAF32StraightBranchless) {
    // Foreground alpha = 0
    {
        RGBA_F32 fg{1.f, 0.f, 0.f, 0.f};
        RGBA_F32 bg{0.f, 1.f, 0.f, 1.f};
        RGBA_F32 ref = blend_RGBAF32_over_RGBAF32_straight_reference(fg, bg);
        RGBA_F32 out = blend_RGBAF32_over_RGBAF32_straight_branchless(fg, bg);
        EXPECT_NEAR(out.r, ref.r, 1e-6);
        EXPECT_NEAR(out.g, ref.g, 1e-6);
        EXPECT_NEAR(out.b, ref.b, 1e-6);
        EXPECT_NEAR(out.a, ref.a, 1e-6);
    }
    // Foreground alpha = 1
    {
        RGBA_F32 fg{0.f, 0.f, 1.f, 1.f};
        RGBA_F32 bg{0.f, 1.f, 0.f, 1.f};
        RGBA_F32 ref = blend_RGBAF32_over_RGBAF32_straight_reference(fg, bg);
        RGBA_F32 out = blend_RGBAF32_over_RGBAF32_straight_branchless(fg, bg);
        EXPECT_NEAR(out.r, ref.r, 1e-6);
        EXPECT_NEAR(out.g, ref.g, 1e-6);
        EXPECT_NEAR(out.b, ref.b, 1e-6);
        EXPECT_NEAR(out.a, ref.a, 1e-6);
    }
}

TEST(CompositTest, RGBAF32OverRGBF32StraightBranchless) {
    RGBA_F32 fg{0.25f, 0.75f, 0.5f, 0.5f};
    RGB_F32 bg{0.25f, 0.25f, 0.25f};
    RGB_F32 ref = blend_RGBAF32_over_RGBF32_straight_reference(fg, bg);
    RGB_F32 out = blend_RGBAF32_over_RGBF32_straight_branchless(fg, bg);
    EXPECT_NEAR(out.r, ref.r, 1e-6);
    EXPECT_NEAR(out.g, ref.g, 1e-6);
    EXPECT_NEAR(out.b, ref.b, 1e-6);
}

TEST(CompositTest, RGBAF32OverRGBAF32PremultipliedBranchless) {
    RGBA_F32 fg{0.2f, 0.6f, 0.8f, 0.4f};
    RGBA_F32 bg{0.8f, 0.2f, 0.2f, 0.7f};
    // Reference (without manually premultiplying fg, bg)
    RGBA_F32 ref = blend_RGBAF32_over_RGBAF32_premultiplied_reference(fg, bg);
    RGBA_F32 out = blend_RGBAF32_over_RGBAF32_premultiplied_branchless(fg, bg);
    EXPECT_NEAR(out.r, ref.r, 1e-6);
    EXPECT_NEAR(out.g, ref.g, 1e-6);
    EXPECT_NEAR(out.b, ref.b, 1e-6);
    EXPECT_NEAR(out.a, ref.a, 1e-6);
}

TEST(CompositTest, RGBAF32OverRGBF32PremultipliedBranchless) {
    RGBA_F32 fg{1.f, 1.f, 0.f, 0.5f};
    RGB_F32 bg{1.f, 0.f, 1.f};
    RGB_F32 ref = blend_RGBAF32_over_RGBF32_premultiplied_reference(fg, bg);
    RGB_F32 out = blend_RGBAF32_over_RGBF32_premultiplied_branchless(fg, bg);
    EXPECT_NEAR(out.r, ref.r, 1e-6);
    EXPECT_NEAR(out.g, ref.g, 1e-6);
    EXPECT_NEAR(out.b, ref.b, 1e-6);
}

TEST(CompositTest, RGBAU8OverRGBAU8PremultipliedScalarDivide) {
    RGBA_U8 fg{32, 64, 100, 128};
    RGBA_U8 bg{18, 85, 0, 128};
    // Convert to float
    RGBA_F32 fgF = {fg.r/255.f, fg.g/255.f, fg.b/255.f, fg.a/255.f};
    RGBA_F32 bgF = {bg.r/255.f, bg.g/255.f, bg.b/255.f, bg.a/255.f};
    // Reference in float
    RGBA_F32 refF = blend_RGBAF32_over_RGBAF32_premultiplied_reference(fgF, bgF);
    // Convert reference back to U8
    auto clamp255 = [](float x){return (U8)std::min(std::max(int(x*255.f+0.5f),0),255);};
    RGBA_U8 refU8 = { clamp255(refF.r), clamp255(refF.g), clamp255(refF.b), clamp255(refF.a) };

    RGBA_U8 out = blend_RGBAU8_over_RGBAU8_premultiplied_scalar_divide(fg, bg);

    EXPECT_EQ(out.r, refU8.r);
    EXPECT_EQ(out.g, refU8.g);
    EXPECT_EQ(out.b, refU8.b);
    EXPECT_EQ(out.a, refU8.a);
}

TEST(CompositTest, RGBAU8OverRGBU8PremultipliedScalarDivide) {
    RGBA_U8 fg{100, 25, 50, 128};
    RGB_U8 bg{200, 50, 200};
    // Convert to float
    RGBA_F32 fgF = {fg.r/255.f, fg.g/255.f, fg.b/255.f, fg.a/255.f};
    RGB_F32 bgF = {bg.r/255.f, bg.g/255.f, bg.b/255.f};
    // Reference in float
    RGB_F32 refF = blend_RGBAF32_over_RGBF32_premultiplied_reference(fgF, bgF);
    // Convert reference back to U8
    auto clamp255 = [](float x){return (U8)std::min(std::max(int(x*255.f+0.5f),0),255);};
    RGB_U8 refU8 = { clamp255(refF.r), clamp255(refF.g), clamp255(refF.b) };

    RGB_U8 out = blend_RGBAU8_over_RGBU8_premultiplied_scalar_divide(fg, bg);

    EXPECT_EQ(out.r, refU8.r);
    EXPECT_EQ(out.g, refU8.g);
    EXPECT_EQ(out.b, refU8.b);
}
