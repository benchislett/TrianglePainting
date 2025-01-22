#include "composit.h"

RGBA01 composit_over_premultiplied_01(const RGBA01& background, const RGBA01 &foreground) {
    return RGBA01{
        background.r * (1 - foreground.a) + foreground.r,
        background.g * (1 - foreground.a) + foreground.g,
        background.b * (1 - foreground.a) + foreground.b,
        background.a * (1 - foreground.a) + foreground.a
    };
}

RGBA01 composit_over_straight_01(const RGBA01& background, const RGBA01 &foreground) {
    if (foreground.a == 0) {
        return background;
    }

    RGBA01 background_premultiplied = RGBA01{
        background.r * background.a,
        background.g * background.a,
        background.b * background.a,
        background.a
    };
    RGBA01 foreground_premultiplied = RGBA01{
        foreground.r * foreground.a,
        foreground.g * foreground.a,
        foreground.b * foreground.a,
        foreground.a
    };
    RGBA01 out = composit_over_premultiplied_01(background_premultiplied, foreground_premultiplied);
    return RGBA01{
        out.r / out.a,
        out.g / out.a,
        out.b / out.a,
        out.a
    };
}

// See: http://alvyray.com/Memos/CG/Microsoft/4_comp.pdf
// Image Compositing Fundamentals
// Technical Memo 4
// Alvy Ray Smith
// A: background
// B: foreground

#define INT_MULT(a,b,t) ( (t) = (a) * (b) + 0x80, ( ( ( (t)>>8 ) + (t) )>>8 ) )
#define INT_PRELERP(p, q, a, t) ( (p) + (q) - INT_MULT( a, p, t) )
#define INT_LERP(p, q, a, t) ( (p) + INT_MULT( a, ( (q) - (p) ), t ) )

RGBA255 composit_over_premultiplied_255(const RGBA255& background, const RGBA255 &foreground) {
    if (foreground.a == 0) {
        return background;
    }
    if (foreground.a == 255) {
        return foreground;
    }

    unsigned int t;
    unsigned char r = INT_PRELERP(background.r, foreground.r, foreground.a, t);
    unsigned char g = INT_PRELERP(background.g, foreground.g, foreground.a, t);
    unsigned char b = INT_PRELERP(background.b, foreground.b, foreground.a, t);
    unsigned char a = INT_PRELERP(background.a, foreground.a, foreground.a, t);
    return RGBA255{r, g, b, a};
}

#define PRECISION_BITS 7
#define SHIFTFORDIV255(a) ((((a) >> 8) + a) >> 8)

RGBA255 composit_over_straight_255(const RGBA255& background, const RGBA255 &foreground) {
    if (foreground.a == 0) {
        return background;
    }
    
    unsigned int tmpr, tmpg, tmpb;
    unsigned int blend = background.a * (255 - foreground.a);
    unsigned int outa255 = foreground.a * 255 + blend;

    unsigned int coef1 = foreground.a * 255 * 255 * (1 << PRECISION_BITS) / outa255;
    unsigned int coef2 = 255 * (1 << PRECISION_BITS) - coef1;

    tmpr = foreground.r * coef1 + background.r * coef2;
    tmpg = foreground.g * coef1 + background.g * coef2;
    tmpb = foreground.b * coef1 + background.b * coef2;

    RGBA255 out;
    out.r =
        SHIFTFORDIV255(tmpr + (0x80 << PRECISION_BITS)) >> PRECISION_BITS;
    out.g =
        SHIFTFORDIV255(tmpg + (0x80 << PRECISION_BITS)) >> PRECISION_BITS;
    out.b =
        SHIFTFORDIV255(tmpb + (0x80 << PRECISION_BITS)) >> PRECISION_BITS;
    out.a = SHIFTFORDIV255(outa255 + 0x80);
    return out;
}
