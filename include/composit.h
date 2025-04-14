#pragma once

#include "common.h"
#include "pixel_types.h"

/* ALPHA BLENDING ROUTINES */

/* STRAIGHT-ALPHA FLOAT32 REFERENCE CODE */

inline RGBA_F32 blend_RGBAF32_over_RGBAF32_straight_reference(
    const RGBA_F32& foreground,
    const RGBA_F32& background
) {
    if (foreground.a == 0.0f) {
        return background;
    }
    if (background.a == 0.0f) {
        return foreground;
    }
    if (foreground.a == 1.0f) {
        return foreground;
    }
    float alpha = foreground.a + background.a * (1 - foreground.a);
    return RGBA_F32{
        (foreground.r * foreground.a + background.r * background.a * (1 - foreground.a)) / alpha,
        (foreground.g * foreground.a + background.g * background.a * (1 - foreground.a)) / alpha,
        (foreground.b * foreground.a + background.b * background.a * (1 - foreground.a)) / alpha,
        alpha
    };
}

inline RGB_F32 blend_RGBAF32_over_RGBF32_straight_reference(
    const RGBA_F32& foreground,
    const RGB_F32& background
) {
    if (foreground.a == 0.0f) {
        return background;
    }
    if (foreground.a == 1.0f) {
        return RGB_F32{
            foreground.r,
            foreground.g,
            foreground.b
        };
    }
    float alpha = foreground.a;
    return RGB_F32{
        (foreground.r * alpha + background.r * (1 - alpha)),
        (foreground.g * alpha + background.g * (1 - alpha)),
        (foreground.b * alpha + background.b * (1 - alpha))
    };
}

/* PREMULTIPLIED-ALPHA FLOAT32 REFERENCE CODE */

inline RGBA_F32 blend_RGBAF32_over_RGBAF32_premultiplied_reference(
    const RGBA_F32& foreground,
    const RGBA_F32& background
) {
    if (foreground.a == 0.0f) {
        return background;
    }
    if (background.a == 0.0f) {
        return foreground;
    }
    if (foreground.a == 1.0f) {
        return foreground;
    }
    float alpha = foreground.a + background.a * (1 - foreground.a);
    return RGBA_F32{
        (foreground.r + background.r * (1 - foreground.a)) / alpha,
        (foreground.g + background.g * (1 - foreground.a)) / alpha,
        (foreground.b + background.b * (1 - foreground.a)) / alpha,
        alpha
    };
}

inline RGB_F32 blend_RGBAF32_over_RGBF32_premultiplied_reference(
    const RGBA_F32& foreground,
    const RGB_F32& background
) {
    if (foreground.a == 0.0f) {
        return background;
    }
    if (foreground.a == 1.0f) {
        return RGB_F32{
            foreground.r,
            foreground.g,
            foreground.b
        };
    }
    float alpha = foreground.a;
    return RGB_F32{
        (foreground.r + background.r * (1 - alpha)),
        (foreground.g + background.g * (1 - alpha)),
        (foreground.b + background.b * (1 - alpha))
    };
}

/* STRAIGHT-ALPHA FLOAT32 BENCHMARK CODE */

inline RGBA_F32 blend_RGBAF32_over_RGBAF32_straight_branchless(
    const RGBA_F32& foreground,
    const RGBA_F32& background
) {
    float alpha = foreground.a + background.a * (1 - foreground.a);
    float invAlpha = 1.0f / alpha;
    return RGBA_F32{
        (foreground.r * foreground.a + background.r * background.a * (1 - foreground.a)) * invAlpha,
        (foreground.g * foreground.a + background.g * background.a * (1 - foreground.a)) * invAlpha,
        (foreground.b * foreground.a + background.b * background.a * (1 - foreground.a)) * invAlpha,
        alpha
    };
}

inline RGB_F32 blend_RGBAF32_over_RGBF32_straight_branchless(
    const RGBA_F32& foreground,
    const RGB_F32& background
) {
    float alpha = foreground.a;
    return RGB_F32{
        (foreground.r * alpha + background.r * (1 - alpha)),
        (foreground.g * alpha + background.g * (1 - alpha)),
        (foreground.b * alpha + background.b * (1 - alpha))
    };
}

/* PREMULTIPLIED-ALPHA FLOAT32 BENCHMARK CODE */

inline RGBA_F32 blend_RGBAF32_over_RGBAF32_premultiplied_branchless(
    const RGBA_F32& foreground,
    const RGBA_F32& background
) {
    float alpha = foreground.a + background.a * (1 - foreground.a);
    float invAlpha = 1.0f / alpha;
    return RGBA_F32{
        (foreground.r + background.r * (1 - foreground.a)) * invAlpha,
        (foreground.g + background.g * (1 - foreground.a)) * invAlpha,
        (foreground.b + background.b * (1 - foreground.a)) * invAlpha,
        alpha
    };
}

inline RGB_F32 blend_RGBAF32_over_RGBF32_premultiplied_branchless(
    const RGBA_F32& foreground,
    const RGB_F32& background
) {
    float oneMinusAlpha = 1.0f - foreground.a;
    return RGB_F32{
        (foreground.r + background.r * oneMinusAlpha),
        (foreground.g + background.g * oneMinusAlpha),
        (foreground.b + background.b * oneMinusAlpha)
    };
}

/* PREMULTIPLIED-ALPHA UINT8 BENCHMARK CODE */

inline RGBA_U8 blend_RGBAU8_over_RGBAU8_premultiplied_scalar_divide(
    const RGBA_U8& foreground,
    const RGBA_U8& background
) {
    int alpha = foreground.a + (background.a * (255 - foreground.a)) / 255;
    return RGBA_U8{
        (U8)((foreground.r + (background.r * (255 - foreground.a)) / 255) * 255 / alpha),
        (U8)((foreground.g + (background.g * (255 - foreground.a)) / 255) * 255 / alpha),
        (U8)((foreground.b + (background.b * (255 - foreground.a)) / 255) * 255 / alpha),
        (U8)(alpha)
    };
}

inline RGB_U8 blend_RGBAU8_over_RGBU8_premultiplied_scalar_divide(
    const RGBA_U8& foreground,
    const RGB_U8& background
) {
    int oneMinusAlpha = 255 - foreground.a;
    return RGB_U8{
        (U8)(foreground.r + (background.r * oneMinusAlpha) / 255),
        (U8)(foreground.g + (background.g * oneMinusAlpha) / 255),
        (U8)(foreground.b + (background.b * oneMinusAlpha) / 255)
    };
}
