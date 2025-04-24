#pragma once

#include <immintrin.h>

inline void blend_RGBAU8_over_RGBU8_premultiplied_scalar(
    unsigned char fg_r, unsigned char fg_g, unsigned char fg_b, unsigned char fg_a,
    unsigned char bg_r, unsigned char bg_g, unsigned char bg_b,
    unsigned char& out_r, unsigned char& out_g, unsigned char& out_b
) {
    out_r = (fg_r + (bg_r * (255 - fg_a)) / 255);
    out_g = (fg_g + (bg_g * (255 - fg_a)) / 255);
    out_b = (fg_b + (bg_b * (255 - fg_a)) / 255);
}

void draw_fill_rgba_over_imrgb_scalar(unsigned char* __restrict buffer_rgb, unsigned char foreground[4], int width) {
    int N = width * width * 3;

    for (int i = 0; i < N; i += 3) {
        unsigned char bg_r = buffer_rgb[i];
        unsigned char bg_g = buffer_rgb[i + 1];
        unsigned char bg_b = buffer_rgb[i + 2];

        unsigned char out_r, out_g, out_b;
        blend_RGBAU8_over_RGBU8_premultiplied_scalar(
            foreground[0], foreground[1], foreground[2], foreground[3],
            bg_r, bg_g, bg_b,
            out_r, out_g, out_b
        );

        buffer_rgb[i] = out_r;
        buffer_rgb[i + 1] = out_g;
        buffer_rgb[i + 2] = out_b;
    }
}

inline __m512i DivideI16x32By255Approx(__m512i v)
{
    v = _mm512_add_epi16(v, _mm512_srli_epi16(v, 8));   // v + (v>>8)
    v = _mm512_add_epi16(v, _mm512_set1_epi16(1));      // +1
    return _mm512_srli_epi16(v, 8);
}

inline void blend_RGBAU16x32_over_RGBU16x32_premultiplied_avx512(
    __m512i fg_r, __m512i fg_g, __m512i fg_b, __m512i fg_a,
    __m512i bg_r, __m512i bg_g, __m512i bg_b,
    __m512i& out_r, __m512i& out_g, __m512i& out_b
) {
    __m512i fg_1minus_a = _mm512_sub_epi16(_mm512_set1_epi16(255), fg_a);
    out_r = _mm512_add_epi16(fg_r, DivideI16x32By255Approx(_mm512_mullo_epi16(bg_r, fg_1minus_a)));
    out_g = _mm512_add_epi16(fg_g, DivideI16x32By255Approx(_mm512_mullo_epi16(bg_g, fg_1minus_a)));
    out_b = _mm512_add_epi16(fg_b, DivideI16x32By255Approx(_mm512_mullo_epi16(bg_b, fg_1minus_a)));
}

void draw_fill_rgba_over_imrgb_avx512(unsigned char* __restrict buffer_rgb, unsigned char foreground[4], int width) {
    int N = width * width * 3;
    __m256i fg_r8 = _mm256_set1_epi8(foreground[0]);
    __m256i fg_g8 = _mm256_set1_epi8(foreground[1]);
    __m256i fg_b8 = _mm256_set1_epi8(foreground[2]);
    __m256i fg_a8 = _mm256_set1_epi8(foreground[3]);
    __m512i fg_r16 = _mm512_cvtepu8_epi16(fg_r8);
    __m512i fg_g16 = _mm512_cvtepu8_epi16(fg_g8);
    __m512i fg_b16 = _mm512_cvtepu8_epi16(fg_b8);
    __m512i fg_a16 = _mm512_cvtepu8_epi16(fg_a8);

    for (size_t i = 0; i < N; i += 96)
    {
        /* ---- 1. 32-byte loads (AVX-256) ------------------------------------ */

        __m256i bg_r8 = _mm256_load_si256((__m256i const*)(buffer_rgb + i +  0));
        __m256i bg_g8 = _mm256_load_si256((__m256i const*)(buffer_rgb + i + 32));
        __m256i bg_b8 = _mm256_load_si256((__m256i const*)(buffer_rgb + i + 64));

        /* ---- 2. widen 32×u8  → 32×u16   (VPMOVZXBW) ------------------------ */

        __m512i bg_r16 = _mm512_cvtepu8_epi16(bg_r8);
        __m512i bg_g16 = _mm512_cvtepu8_epi16(bg_g8);
        __m512i bg_b16 = _mm512_cvtepu8_epi16(bg_b8);

        /* ---- 3.  core blend  ---------------------------------------------- */

        __m512i out_r16, out_g16, out_b16;
        blend_RGBAU16x32_over_RGBU16x32_premultiplied_avx512(
            fg_r16, fg_g16, fg_b16, fg_a16,
            bg_r16, bg_g16, bg_b16,
            out_r16, out_g16, out_b16
        );

        /* ---- 4.  pack-to-u8 *and* store   (VPMOVUSWB) --------------------- */
        _mm512_mask_cvtusepi16_storeu_epi8(buffer_rgb + i +  0, 0xFFFFFFFFu, out_r16); // R
        _mm512_mask_cvtusepi16_storeu_epi8(buffer_rgb + i + 32, 0xFFFFFFFFu, out_g16); // G
        _mm512_mask_cvtusepi16_storeu_epi8(buffer_rgb + i + 64, 0xFFFFFFFFu, out_b16); // B
    }
}

int edgeFunction(
    int ax, int ay,
    int bx, int by,
    int cx, int cy
) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

void draw_triangle_rgba_over_imrgb_scalar(unsigned char* __restrict buffer_rgb, unsigned char foreground[4], int width) {
    int N = width * width * 3;

    int v0[2] = {0, 0};
    int v1[2] = {3 * width, 0};
    int v2[2] = {0, 3 * width};

    int w0_base = edgeFunction(v1[0], v1[1], v2[0], v2[1], 0, 0);
    int w1_base = edgeFunction(v2[0], v2[1], v0[0], v0[1], 0, 0);
    int w2_base = edgeFunction(v0[0], v0[1], v1[0], v1[1], 0, 0);

    int w0_row = w0_base;
    int w1_row = w1_base;
    int w2_row = w2_base;

    int w0_step_x = -(v2[1] - v1[1]);
    int w0_step_y = (v2[0] - v1[0]);
    int w1_step_x = -(v0[1] - v2[1]);
    int w1_step_y = (v0[0] - v2[0]);
    int w2_step_x = -(v1[1] - v0[1]);
    int w2_step_y = (v1[0] - v0[0]);

    for (int y = 0; y < width; ++y) {
        
        int w0 = w0_row;
        int w1 = w1_row;
        int w2 = w2_row;

        for (int x = 0; x < width; x++) {
            int i = (y * width + x) * 3;

            if (x != 0) {
                w0 += w0_step_x;
                w1 += w1_step_x;
                w2 += w2_step_x;
            }

            if (w0 < 0 || w1 < 0 || w2 < 0) {
                continue;
            }

            unsigned char bg_r = buffer_rgb[i];
            unsigned char bg_g = buffer_rgb[i + 1];
            unsigned char bg_b = buffer_rgb[i + 2];
            
            unsigned char out_r, out_g, out_b;
            blend_RGBAU8_over_RGBU8_premultiplied_scalar(
                foreground[0], foreground[1], foreground[2], foreground[3],
                bg_r, bg_g, bg_b,
                out_r, out_g, out_b
            );
            
            buffer_rgb[i] = out_r;
            buffer_rgb[i + 1] = out_g;
            buffer_rgb[i + 2] = out_b;
        }

        w0_row += w0_step_y;
        w1_row += w1_step_y;
        w2_row += w2_step_y;
    }
}

void draw_triangle_rgba_over_imrgb_avx512(unsigned char* __restrict buffer_rgb, unsigned char foreground[4], int width) {
    int N = width * width * 3;

    /* -------- 0.  Triangle setup (scalar) -------------------------------- */
    const int v0[2] = {0, 0};
    const int v1[2] = {3 * width, 0};
    const int v2[2] = {0, 3 * width};

    const int w0_base = edgeFunction(v1[0], v1[1], v2[0], v2[1], 0, 0);
    const int w1_base = edgeFunction(v2[0], v2[1], v0[0], v0[1], 0, 0);
    const int w2_base = edgeFunction(v0[0], v0[1], v1[0], v1[1], 0, 0);

    const int w0_step_x = -(v2[1] - v1[1]);   //   -(y2-y1)
    const int w1_step_x = -(v0[1] - v2[1]);   //   -(y0-y2)
    const int w2_step_x = -(v1[1] - v0[1]);   //   -(y1-y0)

    const int w0_step_y =  (v2[0] - v1[0]);   //    x2-x1
    const int w1_step_y =  (v0[0] - v2[0]);   //    x0-x2
    const int w2_step_y =  (v1[0] - v0[0]);   //    x1-x0

    /* ---- constant vectors: fg colour, offset indices, Δx per lane ------- */
    __m256i fg_r8 = _mm256_set1_epi8(foreground[0]);
    __m256i fg_g8 = _mm256_set1_epi8(foreground[1]);
    __m256i fg_b8 = _mm256_set1_epi8(foreground[2]);
    __m256i fg_a8 = _mm256_set1_epi8(foreground[3]);
    __m512i fg_r16 = _mm512_cvtepu8_epi16(fg_r8);
    __m512i fg_g16 = _mm512_cvtepu8_epi16(fg_g8);
    __m512i fg_b16 = _mm512_cvtepu8_epi16(fg_b8);
    __m512i fg_a16 = _mm512_cvtepu8_epi16(fg_a8);

    const __m512i idx0_15 =
        _mm512_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7,
                        8, 9,10,11,12,13,14,15);

    const __m512i w0_dx = _mm512_mullo_epi32(idx0_15, _mm512_set1_epi32(w0_step_x));
    const __m512i w1_dx = _mm512_mullo_epi32(idx0_15, _mm512_set1_epi32(w1_step_x));
    const __m512i w2_dx = _mm512_mullo_epi32(idx0_15, _mm512_set1_epi32(w2_step_x));

    /* ------------------------------- main raster loop -------------------- */
    for (int y = 0; y < width; ++y)
    {
        const int w0_row = w0_base + y * w0_step_y;
        const int w1_row = w1_base + y * w1_step_y;
        const int w2_row = w2_base + y * w2_step_y;

        for (int x = 0; x < width; x += 32)
        {
            /* --- 1. build edge values for the 32 pixels in this tile ----- */
            const int w0_blk = w0_row + x * w0_step_x;
            const int w1_blk = w1_row + x * w1_step_x;
            const int w2_blk = w2_row + x * w2_step_x;

            /* first 16 pixels (x+0 … x+15) */
            __m512i w0_lo = _mm512_add_epi32(_mm512_set1_epi32(w0_blk), w0_dx);
            __m512i w1_lo = _mm512_add_epi32(_mm512_set1_epi32(w1_blk), w1_dx);
            __m512i w2_lo = _mm512_add_epi32(_mm512_set1_epi32(w2_blk), w2_dx);

            /* next 16 pixels (x+16 … x+31)  → base + 16*Δx */
            const int w0_blk_hi = w0_blk + 16 * w0_step_x;
            const int w1_blk_hi = w1_blk + 16 * w1_step_x;
            const int w2_blk_hi = w2_blk + 16 * w2_step_x;

            __m512i w0_hi = _mm512_add_epi32(_mm512_set1_epi32(w0_blk_hi), w0_dx);
            __m512i w1_hi = _mm512_add_epi32(_mm512_set1_epi32(w1_blk_hi), w1_dx);
            __m512i w2_hi = _mm512_add_epi32(_mm512_set1_epi32(w2_blk_hi), w2_dx);

            /* --- 2. inside-test → __mmask32 (1 bit / pixel) --------------- */
            const __mmask16 neg0_lo = _mm512_cmplt_epi32_mask(w0_lo, _mm512_setzero_si512());
            const __mmask16 neg1_lo = _mm512_cmplt_epi32_mask(w1_lo, _mm512_setzero_si512());
            const __mmask16 neg2_lo = _mm512_cmplt_epi32_mask(w2_lo, _mm512_setzero_si512());

            const __mmask16 neg0_hi = _mm512_cmplt_epi32_mask(w0_hi, _mm512_setzero_si512());
            const __mmask16 neg1_hi = _mm512_cmplt_epi32_mask(w1_hi, _mm512_setzero_si512());
            const __mmask16 neg2_hi = _mm512_cmplt_epi32_mask(w2_hi, _mm512_setzero_si512());

            const __mmask16 inside_lo = ~(neg0_lo | neg1_lo | neg2_lo);
            const __mmask16 inside_hi = ~(neg0_hi | neg1_hi | neg2_hi);
            const __mmask32 inside = (__mmask32)inside_lo | ((__mmask32)inside_hi << 16);

            /* --- 3. load BG, blend, store with the mask ------------------ */
            const size_t base = static_cast<size_t>(y) * width + x;

            __m256i bg_r8 = _mm256_load_si256((__m256i const*)(buffer_rgb + (base*3) +  0));
            __m256i bg_g8 = _mm256_load_si256((__m256i const*)(buffer_rgb + (base*3) + 32));
            __m256i bg_b8 = _mm256_load_si256((__m256i const*)(buffer_rgb + (base*3) + 64));

            __m512i bg_r16 = _mm512_cvtepu8_epi16(bg_r8);
            __m512i bg_g16 = _mm512_cvtepu8_epi16(bg_g8);
            __m512i bg_b16 = _mm512_cvtepu8_epi16(bg_b8);

            __m512i out_r16, out_g16, out_b16;
            blend_RGBAU16x32_over_RGBU16x32_premultiplied_avx512(
                fg_r16, fg_g16, fg_b16, fg_a16,
                bg_r16, bg_g16, bg_b16,
                out_r16, out_g16, out_b16);

            _mm512_mask_cvtusepi16_storeu_epi8(buffer_rgb + (base*3) +  0, inside, out_r16);
            _mm512_mask_cvtusepi16_storeu_epi8(buffer_rgb + (base*3) + 32, inside, out_g16);
            _mm512_mask_cvtusepi16_storeu_epi8(buffer_rgb + (base*3) + 64, inside, out_b16);
        }
    }
}
