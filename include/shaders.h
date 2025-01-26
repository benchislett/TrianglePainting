#pragma once

#include "colours.h"
#include "composit.h"
#include "image.h"

#include <algorithm>
#include <tuple>

struct Shader {
    virtual void render(float u, float v) = 0;
};

struct ImageShader : Shader {
    virtual void render_pixel(int x, int y) = 0;

    virtual int width() const = 0;
    virtual int height() const = 0;

    void render(float u, float v) override {
        render_pixel(u * width(), v * height());
    }
};

struct CompositOverShader {
    ImageView<RGBA255> background;
    RGBA255 colour;

    CompositOverShader(const ImageView<RGBA255>& bg, const RGBA255& col) : background(bg), colour(col) {}
    
    void render_pixel(int x, int y) {
        RGBA255 bg_pixel = background(x, y);
        RGBA255 out_pixel = composit_over_straight_255(bg_pixel, colour);
        background(x, y) = out_pixel;
    }
};

struct OptimalColourShader {
    long long int ab_r = 0;
    long long int ab_g = 0;
    long long int ab_b = 0;
    long long int a2 = 0;
    long long int b2 = 0;

    long long int error_delta = 0;

    unsigned char current_alpha;

    ImageView<RGBA255> target;
    ImageView<RGBA255> foreground;
    ImageView<RGBA255> background;
    ImageView<int> error_mask;

    OptimalColourShader(unsigned char alpha, const ImageView<RGBA255>& target, const ImageView<RGBA255>& foreground, const ImageView<RGBA255>& background, const ImageView<int>& error_mask) 
        : current_alpha(alpha), target(target), foreground(foreground), background(background), error_mask(error_mask) {}

    void render_pixel(int x, int y) {
        RGBA255 target_pixel = target(x, y);
        RGBA255 background_pixel = background(x, y);
        RGBA255 foreground_pixel = foreground(x, y);

#define INT_MULT(a,b,t) ( (t) = (a) * (b) + 0x80, ( ( ( (t)>>8 ) + (t) )>>8 ) )
#define INT_LERP(p, q, a, t) ( (p) + INT_MULT( a, ( (q) - (p) ), t ) )

        long long int a_0, b_0_r, b_0_g, b_0_b;
        int t1, t2, t3;

        if (foreground_pixel.a == 0) {
            a_0 = current_alpha;
            b_0_r = int(target_pixel.r) - int(background_pixel.r) + INT_MULT(background_pixel.r, current_alpha, t1);
            b_0_g = int(target_pixel.g) - int(background_pixel.g) + INT_MULT(background_pixel.g, current_alpha, t1);
            b_0_b = int(target_pixel.b) - int(background_pixel.b) + INT_MULT(background_pixel.b, current_alpha, t1);
        } else {
            a_0 = int(current_alpha) - INT_MULT(current_alpha, foreground_pixel.a, t1);

            int tmp_r = int(background_pixel.r) - INT_MULT(background_pixel.r, current_alpha, t1);
            int tmp_g = int(background_pixel.g) - INT_MULT(background_pixel.g, current_alpha, t1);
            int tmp_b = int(background_pixel.b) - INT_MULT(background_pixel.b, current_alpha, t1);

            b_0_r = int(target_pixel.r) - INT_LERP(tmp_r, foreground_pixel.r, foreground_pixel.a, t2);
            b_0_g = int(target_pixel.g) - INT_LERP(tmp_g, foreground_pixel.g, foreground_pixel.a, t2);
            b_0_b = int(target_pixel.b) - INT_LERP(tmp_b, foreground_pixel.b, foreground_pixel.a, t2);
        }

        ab_r += a_0 * b_0_r;
        ab_g += a_0 * b_0_g;
        ab_b += a_0 * b_0_b;
        a2 += a_0 * a_0;
        b2 += b_0_r * b_0_r + b_0_g * b_0_g + b_0_b * b_0_b;

        error_delta -= error_mask(x, y);
    }

    std::pair<RGBA255, long long int> final_colour_and_error() const {
        if (a2 == 0) {
            return {RGBA255{0, 0, 0, 0}, 0};
        }

        unsigned char r = std::min(255, std::max(0, int((255 * ab_r) / a2)));
        unsigned char g = std::min(255, std::max(0, int((255 * ab_g) / a2)));
        unsigned char b = std::min(255, std::max(0, int((255 * ab_b) / a2)));
        auto out_colour = RGBA255{r, g, b, current_alpha};

        long long int error = error_delta + b2;
        error += ((int(r) * int(r) + int(g) * int(g) + int(b) * int(b)) * a2) / (255 * 255);
        error -= (2 * (int(r) * ab_r + int(g) * ab_g + int(b) * ab_b)) / 255;

        return {out_colour, error};
    }
};

struct DrawLossFGShader {
    ImageView<RGBA255> target;
    ImageView<RGBA255> foreground;
    ImageView<RGBA255> background;
    ImageView<int> error_mask;

    RGBA255 colour;

    long long int total_error = 0;

    DrawLossFGShader(const ImageView<RGBA255>& target, const ImageView<RGBA255>& foreground, const ImageView<RGBA255>& background, const RGBA255& colour, const ImageView<int>& error_mask) 
        : target(target), foreground(foreground), background(background), colour(colour), error_mask(error_mask) {}

    void render_pixel(int x, int y) {
        RGBA255 target_pixel = target(x, y);
        RGBA255 background_pixel = background(x, y);
        RGBA255 foreground_pixel = foreground(x, y);

        auto middle_pixel = composit_over_straight_255(background_pixel, colour);
        auto final_pixel = composit_over_straight_255(middle_pixel, foreground_pixel);

        total_error += l2_difference(target_pixel, final_pixel);
        total_error -= error_mask(x, y);
    }
};
