#pragma once

#include "shaders.h"
#include "geometry.h"

#include "utils.h"

#include <memory>
#include <cstring>
#include <string>

enum class RasterStrategy {
    Bounded,
    Integer,
    Scanline,
    EdgeTable,
    Binned,
    Default = Bounded
};

std::string raster_strategy_name(RasterStrategy strategy) {
    switch (strategy) {
        case RasterStrategy::Bounded:
            return "Bounded";
        case RasterStrategy::Integer:
            return "Integer";
        case RasterStrategy::Scanline:
            return "Scanline";
        case RasterStrategy::EdgeTable:
            return "EdgeTable";
        case RasterStrategy::Binned:
            return "Binned";
        default:
            return "Unknown";
    }
}

struct RasterConfig {
    RasterStrategy strategy = RasterStrategy::Default;
    int image_width = 0;
    int image_height = 0;
};

template<class Shader>
void rasterize(std::shared_ptr<Shape> shape, Shader& shader, const RasterConfig& config) {
    ShapeType T = shape->type();

    if (config.strategy == RasterStrategy::Bounded) {
        if (T == ShapeType::Triangle) {
            rasterize_shape_bounded<Shader, Triangle>(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
        } else if (T == ShapeType::Circle) {
            rasterize_shape_bounded<Shader, Circle>(dynamic_cast<Circle&>(*shape), config.image_width, config.image_height, shader);
        } else if (T == ShapeType::Polygon) {
            rasterize_shape_bounded<Shader, Polygon>(dynamic_cast<Polygon&>(*shape), config.image_width, config.image_height, shader);
        } else {
            throw std::runtime_error("Unsupported shape type for rasterization.");
        }
    } else if (config.strategy == RasterStrategy::Integer) {
        if (T != ShapeType::Triangle) {
            throw std::runtime_error("Integer rasterization strategy is only supported for triangles.");
        }
        rasterize_triangle_integer(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
    } else if (config.strategy == RasterStrategy::Scanline) {
        if (T == ShapeType::Triangle) {
            rasterize_polygon_scanline(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
        } else if (T == ShapeType::Polygon) {
            rasterize_polygon_scanline(dynamic_cast<Polygon&>(*shape), config.image_width, config.image_height, shader);
        } else {
            throw std::runtime_error("Scanline polygon rasterization is only supported for triangles and polygons.");
        }
    } else if (config.strategy == RasterStrategy::EdgeTable) {
        if (T == ShapeType::Triangle) {
            if constexpr (std::is_same_v<Shader, CompositOverShader>) {
                rasterize_test_polygon(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader); 
            } else {
                throw std::runtime_error("Test polygon rasterization is only supported for CompositOverShader.");
            }
        } else {
            throw std::runtime_error("Test polygon rasterization is only supported for triangles.");
        }
    } else if (config.strategy == RasterStrategy::Binned) {
        if (T == ShapeType::Triangle) {
            if constexpr (std::is_same_v<Shader, CompositOverShader>) {
                rasterize_triangle_binned(dynamic_cast<Triangle&>(*shape), config.image_width, config.image_height, shader);
            } else {
                throw std::runtime_error("Test polygon rasterization is only supported for CompositOverShader.");
            }
        } else {
            throw std::runtime_error("Binned rasterization strategy is only supported for triangles.");
        }
    } else {
        throw std::runtime_error("Unsupported rasterization strategy.");
    }
}

/* Generic 2D Rasterization 
    * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `shape`
    * assuming an image domain of size `width` x `height`.
    * 
    * Method: Bounded loop.
    * - Determine the integer bounding box of the shape within the image
    * - For each pixel in the box, compute the barycentric coordinates at the center of that pixel w.r.t. the `shape`
    * - If the barycentric coordinates are all non-negative, shade the point. */
template<class Shader, typename Shape>
void rasterize_shape_bounded(const Shape& shape, int width, int height, Shader& shader) {
    Point lower = shape.min();
    Point upper = shape.max();

    int lower_x = std::max(0, std::min(int(lower.x * width), width - 1));
    int lower_y = std::max(0, std::min(int(lower.y * height), height - 1));
    int upper_x = std::max(0, std::min(int(upper.x * width), width - 1));
    int upper_y = std::max(0, std::min(int(upper.y * height), height - 1));

    for (int x = lower_x; x <= upper_x; x++) {
        for (int y = lower_y; y <= upper_y; y++) {
            float u = (x + 0.5f) / (float)width;
            float v = (y + 0.5f) / (float)height;
            if (shape.is_inside(Point{u, v})) {
                shader.render_pixel(x, y);
            }
        }
    }
}

/* Generic 2D Triangle Rasterization 
    * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `triangle`
    * assuming an image domain of size `width` x `height`. 
    * 
    * Method: Integer loop.
    * An optimized variant of the bounded loop method that uses exclusively integer arithmetic
    * to compute inclusion by integer edge functions whose values are updated incrementally.
    * 
    * Credit for the implementation goes to https://fgiesen.wordpress.com/2013/02/10/optimizing-the-basic-rasterizer
    */
template<class Shader>
void rasterize_triangle_integer(const Triangle& tri, int width, int height, Shader& shader) {
    int xs[3] = {int(tri[0].x * width), int(tri[1].x * width), int(tri[2].x * width)};
    int ys[3] = {int(tri[0].y * height), int(tri[1].y * height), int(tri[2].y * height)};

    // Orient the triangle correctly
    int w = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (ys[1] - ys[0]) * (xs[2] - xs[0]);
    if (w < 0) {
        std::swap(xs[0], xs[1]);
        std::swap(ys[0], ys[1]);
    }

    auto orient2d = [](int ax, int ay, int bx, int by, int cx, int cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    };

    int lower_x = std::max(0, std::min({xs[0], xs[1], xs[2]}));
    int lower_y = std::max(0, std::min({ys[0], ys[1], ys[2]}));
    int upper_x = std::min(width - 1, std::max({xs[0], xs[1], xs[2]}));
    int upper_y = std::min(height - 1, std::max({ys[0], ys[1], ys[2]}));

    int w0_row = orient2d(xs[1], ys[1], xs[2], ys[2], lower_x, lower_y);
    int w1_row = orient2d(xs[2], ys[2], xs[0], ys[0], lower_x, lower_y);
    int w2_row = orient2d(xs[0], ys[0], xs[1], ys[1], lower_x, lower_y);

    int A01 = ys[0] - ys[1], B01 = xs[1] - xs[0];
    int A12 = ys[1] - ys[2], B12 = xs[2] - xs[1];
    int A20 = ys[2] - ys[0], B20 = xs[0] - xs[2];

    // Rasterize
    for (int y = lower_y; y <= upper_y; y++) {
        int w0 = w0_row;
        int w1 = w1_row;
        int w2 = w2_row;

        for (int x = lower_x; x <= upper_x; x++) {
            // If p is on or inside all edges, render pixel.
            if ((w0 | w1 | w2) >= 0) {
                shader.render_pixel(x, y);
            }

            // One step to the right
            w0 += A12;
            w1 += A20;
            w2 += A01;
        }

        // One row step
        w0_row += B12;
        w1_row += B20;
        w2_row += B01;
    }
}

/* Generic 2D Polygon Rasterization 
    * Given an arbitrary shader object, call `shader.render_pixel(x, y)` for each pixel in the `polygon`
    * assuming an image domain of size `width` x `height`. 
    * 
    * Method: Sorted scanlines.
    * - For each row of the image, build a list of nodes where the polygon intersects the row
    * - Sort the nodes by x-coordinate
    * - Fill the pixels between even-odd pairs of nodes
    * 
    * Credit for the implementation goes to http://alienryderflex.com/polygon_fill/
    */
template<class Shader, typename Shape>
void rasterize_polygon_scanline(const Shape& polygon, int width, int height, Shader& shader) {
    constexpr int MAX_POLY_CORNERS = 10;

    int  nodes, nodeX[MAX_POLY_CORNERS], pixelX, pixelY, i, j, swap;

    if constexpr (!std::is_same_v<Shape, Triangle> && !std::is_same_v<Shape, Polygon>) {
        static_assert(false, "Unsupported shape type for scanline polygon rasterization.");
    }

    int polyCorners = polygon.num_vertices();
    if (polyCorners > MAX_POLY_CORNERS) {
        throw std::runtime_error("Polygon has too many corners.");
    }

    //  Loop through the rows of the image.
    for (pixelY=0; pixelY<height; pixelY++) {

        //  Build a list of nodes.
        nodes=0; j=polyCorners-1;
        for (i=0; i<polyCorners; i++) {
            float p_i_x = polygon[i].x * width;
            float p_j_x = polygon[j].x * width;
            float p_i_y = polygon[i].y * height;
            float p_j_y = polygon[j].y * height;
            if (p_i_y<(float) pixelY && p_j_y>=(float) pixelY
            ||  p_j_y<(float) pixelY && p_i_y>=(float) pixelY) {
            nodeX[nodes++]=(int) (p_i_x+(pixelY-p_i_y)/(p_j_y-p_i_y)
            *(p_j_x-p_i_x)); }
            j=i; }

        //  Sort the nodes, via a simple “Bubble” sort.
        i=0;
        while (i<nodes-1) {
            if (nodeX[i]>nodeX[i+1]) {
            swap=nodeX[i]; nodeX[i]=nodeX[i+1]; nodeX[i+1]=swap; if (i) i--; }
            else {
            i++; }}

        //  Fill the pixels between node pairs.
        for (i=0; i<nodes; i+=2) {
            if   (nodeX[i  ]>=width) break;
            if   (nodeX[i+1]> 0 ) {
            if (nodeX[i  ]< 0 ) nodeX[i  ]=0 ;
            if (nodeX[i+1]> width) nodeX[i+1]=width;
            for (pixelX=nodeX[i]; pixelX<nodeX[i+1]; pixelX++) shader.render_pixel(pixelX,pixelY); }}}
}

template<class Shader>
int ImagingDrawPolygon(int count, int *xy, int width, int height, Shader& shader);

// template<class Shader>
void rasterize_test_polygon(const Triangle& tri, int width, int height, CompositOverShader& shader) {
    int pts[6] = {
        int(tri[0].x * width), int(tri[0].y * height),
        int(tri[1].x * width), int(tri[1].y * height),
        int(tri[2].x * width), int(tri[2].y * height)
    };
    ImagingDrawPolygon(3, &pts[0], width, height, shader);
}

/* TEST CODE IMPORTED FROM libImaging (from pillow's PIL.ImageDraw core) */


#include <math.h>
#include <stdint.h>

#define CEIL(v) (int)ceil(v)
#define FLOOR(v) ((v) >= 0.0 ? (int)(v) : (int)floor(v))

#define SHIFTFORDIV255(a) ((((a) >> 8) + a) >> 8)

/* like (a * b + 127) / 255), but much faster on most platforms */
#define MULDIV255(a, b, tmp) (tmp = (a) * (b) + 128, SHIFTFORDIV255(tmp))

#define DIV255(a, tmp) (tmp = (a) + 128, SHIFTFORDIV255(tmp))

#define BLEND(mask, in1, in2, tmp1) DIV255(in1 * (255 - mask) + in2 * mask, tmp1)

#define PREBLEND(mask, in1, in2, tmp1) (MULDIV255(in1, (255 - mask), tmp1) + in2)

#define CLIP8(v) ((v) <= 0 ? 0 : (v) < 256 ? (v) : 255)

/*
 * Rounds around zero (up=away from zero, down=towards zero)
 * This guarantees that ROUND_UP|DOWN(f) == -ROUND_UP|DOWN(-f)
 */
#define ROUND_UP(f) ((int)((f) >= 0.0 ? floor((f) + 0.5F) : -floor(fabs(f) + 0.5F)))
#define ROUND_DOWN(f) ((int)((f) >= 0.0 ? ceil((f) - 0.5F) : -ceil(fabs(f) - 0.5F)))

typedef struct {
    /* edge descriptor for polygon engine */
    int d;
    int x0, y0;
    int xmin, ymin, xmax, ymax;
    float dx;
} Edge;


template<class Shader>
static inline void
point32rgba(int x, int y, int width, int height, Shader& shader) {
    unsigned int tmp;

    if (x >= 0 && x < width && y >= 0 && y < height) {
        shader.render_pixel(x, y);
    }
}

// get SSE and AVX
#include <immintrin.h>
#include <xmmintrin.h>

__m128i alphaBlendSSE(unsigned char* buf, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
	const int invA = 255 - a;
	const int F_r = r * invA;
	const int F_g = g * invA;
	const int F_b = b * invA;

	const __m128i fg_const = _mm_setr_epi16(
		(short)F_r, (short)F_g, (short)F_b, 0,
		(short)F_r, (short)F_g, (short)F_b, 0);
	
	const __m128i alpha_vec = _mm_set1_epi16((short)a);

	const __m128i add128 = _mm_set1_epi16(128);

	const __m128i alpha_mask = _mm_set1_epi32(0xff000000);
	const __m128i color_mask = _mm_set1_epi32(0x00ffffff);

	__m128i bg = _mm_load_si128((__m128i*)(buf));

    __m128i bg_lo = _mm_unpacklo_epi8(bg, _mm_setzero_si128());
    __m128i bg_hi = _mm_unpackhi_epi8(bg, _mm_setzero_si128());

    __m128i mult_lo = _mm_mullo_epi16(bg_lo, alpha_vec);
    __m128i sum_lo = _mm_add_epi16(mult_lo, fg_const);
    __m128i tmp_lo = _mm_add_epi16(sum_lo, add128);
    __m128i tmp2_lo = _mm_srli_epi16(tmp_lo, 8);
    __m128i blended_lo = _mm_srli_epi16(_mm_add_epi16(tmp_lo, tmp2_lo), 8);

    __m128i mult_hi = _mm_mullo_epi16(bg_hi, alpha_vec);
    __m128i sum_hi = _mm_add_epi16(mult_hi, fg_const);
    __m128i tmp_hi = _mm_add_epi16(sum_hi, add128);
    __m128i tmp2_hi = _mm_srli_epi16(tmp_hi, 8);
    __m128i blended_hi = _mm_srli_epi16(_mm_add_epi16(tmp_hi, tmp2_hi), 8);

    __m128i blended_pixels = _mm_packus_epi16(blended_lo, blended_hi);

    blended_pixels = _mm_and_si128(blended_pixels, color_mask);
    blended_pixels = _mm_or_si128(blended_pixels, alpha_mask);

    return blended_pixels;
}

__m256i alphaBlendAVX(unsigned char* buf, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    // Precompute the inverse alpha and the constant terms for each channel:
    // For each channel, we want: result = DIV255( bg_channel * a + fg_channel * (255 - a) )
    const int invA = 255 - a;
    const int F_r = r * invA;
    const int F_g = g * invA;
    const int F_b = b * invA;
    
    // For processing 4 pixels at a time (each pixel: R, G, B, 0) pattern,
    // we create a 128-bit constant that will be reused for both halves.
    __m128i fg_const_sse = _mm_setr_epi16(
        (short)F_r, (short)F_g, (short)F_b, 0,
        (short)F_r, (short)F_g, (short)F_b, 0);
    
    // Broadcast the foreground alpha (a) into all 16-bit lanes for the 4 pixels.
    __m128i alpha_vec_sse = _mm_set1_epi16((short)a);
    // Constant for our DIV255 approximation.
    __m128i add128_sse = _mm_set1_epi16(128);
    
    // Create 256-bit masks for forcing the output alpha to 255.
    // Each pixel is 32-bit: we want to zero out the alpha and then OR in 0xff.
    __m256i color_mask256 = _mm256_set1_epi32(0x00ffffff);
    __m256i alpha_mask256 = _mm256_set1_epi32(0xff000000);
    
    // Load 8 pixels (8 x 4 bytes = 32 bytes) from the buffer.
    __m256i bg = _mm256_load_si256((__m256i*)buf);
    
    // Split the 256-bit register into its lower and upper 128-bit halves (each holds 4 pixels).
    __m128i bg_lo = _mm256_castsi256_si128(bg);
    __m128i bg_hi = _mm256_extracti128_si256(bg, 1);
    
    // Process the lower 128-bit half.
    __m128i zero128 = _mm_setzero_si128();
    // Unpack the lower 4 pixels from 8-bit to 16-bit (two groups: low and high parts).
    __m128i bg_lo_lo = _mm_unpacklo_epi8(bg_lo, zero128);
    __m128i bg_lo_hi = _mm_unpackhi_epi8(bg_lo, zero128);
    
    // For the lower group: multiply background channels by 'a' and add the constant.
    __m128i mult_lo_lo = _mm_mullo_epi16(bg_lo_lo, alpha_vec_sse);
    __m128i sum_lo_lo = _mm_add_epi16(mult_lo_lo, fg_const_sse);
    // Approximate division by 255: add 128, then add shifted result, then shift by 8.
    __m128i tmp_lo_lo = _mm_add_epi16(sum_lo_lo, add128_sse);
    __m128i tmp2_lo_lo = _mm_srli_epi16(tmp_lo_lo, 8);
    __m128i blended_lo_lo = _mm_srli_epi16(_mm_add_epi16(tmp_lo_lo, tmp2_lo_lo), 8);
    
    // Process the higher group within the lower half.
    __m128i mult_lo_hi = _mm_mullo_epi16(bg_lo_hi, alpha_vec_sse);
    __m128i sum_lo_hi = _mm_add_epi16(mult_lo_hi, fg_const_sse);
    __m128i tmp_lo_hi = _mm_add_epi16(sum_lo_hi, add128_sse);
    __m128i tmp2_lo_hi = _mm_srli_epi16(tmp_lo_hi, 8);
    __m128i blended_lo_hi = _mm_srli_epi16(_mm_add_epi16(tmp_lo_hi, tmp2_lo_hi), 8);
    
    // Pack the two groups (each 8 x 16-bit) back into 8-bit values: now we have 4 blended pixels.
    __m128i blended_lo_128 = _mm_packus_epi16(blended_lo_lo, blended_lo_hi);
    
    // Process the upper 128-bit half (the other 4 pixels) similarly.
    __m128i bg_hi_lo = _mm_unpacklo_epi8(bg_hi, zero128);
    __m128i bg_hi_hi = _mm_unpackhi_epi8(bg_hi, zero128);
    
    __m128i mult_hi_lo = _mm_mullo_epi16(bg_hi_lo, alpha_vec_sse);
    __m128i sum_hi_lo = _mm_add_epi16(mult_hi_lo, fg_const_sse);
    __m128i tmp_hi_lo = _mm_add_epi16(sum_hi_lo, add128_sse);
    __m128i tmp2_hi_lo = _mm_srli_epi16(tmp_hi_lo, 8);
    __m128i blended_hi_lo = _mm_srli_epi16(_mm_add_epi16(tmp_hi_lo, tmp2_hi_lo), 8);
    
    __m128i mult_hi_hi = _mm_mullo_epi16(bg_hi_hi, alpha_vec_sse);
    __m128i sum_hi_hi = _mm_add_epi16(mult_hi_hi, fg_const_sse);
    __m128i tmp_hi_hi = _mm_add_epi16(sum_hi_hi, add128_sse);
    __m128i tmp2_hi_hi = _mm_srli_epi16(tmp_hi_hi, 8);
    __m128i blended_hi_hi = _mm_srli_epi16(_mm_add_epi16(tmp_hi_hi, tmp2_hi_hi), 8);
    
    __m128i blended_hi_128 = _mm_packus_epi16(blended_hi_lo, blended_hi_hi);
    
    // Recombine the two 128-bit halves into one 256-bit register.
    __m256i blended_pixels = _mm256_castsi128_si256(blended_lo_128);
    blended_pixels = _mm256_inserti128_si256(blended_pixels, blended_hi_128, 1);
    
    // Force the alpha channel to 255 for every pixel:
    blended_pixels = _mm256_or_si256(
        _mm256_and_si256(blended_pixels, color_mask256),
        alpha_mask256);
        
    return blended_pixels;
}

__m256i alphaBlendAVX2_SinglePath(unsigned char* buf, 
    unsigned char r, 
    unsigned char g, 
    unsigned char b, 
    unsigned char a) {
    // Precompute the inverse alpha and constant terms.
    // The blend is: result = DIV255( bg_channel * a + fg_channel * (255 - a) ).
    const int invA = 255 - a;
    const int F_r = r * invA;
    const int F_g = g * invA;
    const int F_b = b * invA;

    // Build a 128-bit constant for one group of 4 pixels:
    // Pattern per pixel is: { F_r, F_g, F_b, 0 }.
    // For 4 pixels we need 4*4 = 16 16-bit values.
    __m128i fg_const_128 = _mm_setr_epi16(
    (short)F_r, (short)F_g, (short)F_b, 0,
    (short)F_r, (short)F_g, (short)F_b, 0);
    // Broadcast this to a 256-bit constant (each 256-bit lane now has the same 16 16-bit values).
    __m256i fg_const = _mm256_broadcastsi128_si256(fg_const_128);

    // Broadcast alpha into all 16-bit lanes.
    __m256i alpha_vec = _mm256_set1_epi16((short)a);
    // Constant used for the DIV255 approximation.
    __m256i add128 = _mm256_set1_epi16(128);

    // Masks for forcing the final alpha to 255.
    __m256i color_mask = _mm256_set1_epi32(0x00ffffff);
    __m256i alpha_mask = _mm256_set1_epi32(0xff000000);

    // Load 8 pixels (32 bytes) from the buffer.
    __m256i bg = _mm256_load_si256((__m256i*)buf);

    // Create a 256-bit zero vector.
    __m256i zero = _mm256_setzero_si256();

    // Unpack 8-bit RGBA values to 16-bit integers.
    // _mm256_unpacklo_epi8 and _mm256_unpackhi_epi8 operate on each 128-bit lane.
    // The "lo" unpack yields the lower 8 bytes from each 128-bit lane interleaved with zero,
    // which corresponds to the first 2 pixels from each lane (pixels 0,1 and 4,5).
    // The "hi" unpack yields the upper 8 bytes from each 128-bit lane (pixels 2,3 and 6,7).
    __m256i bg_lo = _mm256_unpacklo_epi8(bg, zero);
    __m256i bg_hi = _mm256_unpackhi_epi8(bg, zero);

    // Process the lower unpacked group.
    __m256i mult_lo = _mm256_mullo_epi16(bg_lo, alpha_vec);
    __m256i sum_lo  = _mm256_add_epi16(mult_lo, fg_const);
    __m256i tmp_lo  = _mm256_add_epi16(sum_lo, add128);
    __m256i tmp2_lo = _mm256_srli_epi16(tmp_lo, 8);
    __m256i blended_lo = _mm256_srli_epi16(_mm256_add_epi16(tmp_lo, tmp2_lo), 8);

    // Process the higher unpacked group.
    __m256i mult_hi = _mm256_mullo_epi16(bg_hi, alpha_vec);
    __m256i sum_hi  = _mm256_add_epi16(mult_hi, fg_const);
    __m256i tmp_hi  = _mm256_add_epi16(sum_hi, add128);
    __m256i tmp2_hi = _mm256_srli_epi16(tmp_hi, 8);
    __m256i blended_hi = _mm256_srli_epi16(_mm256_add_epi16(tmp_hi, tmp2_hi), 8);

    // Pack the two groups of 16-bit values into one 256-bit register of 8-bit values.
    // _mm256_packus_epi16 packs each 128-bit lane separately.
    __m256i blended = _mm256_packus_epi16(blended_lo, blended_hi);

    // Force alpha to 255: clear alpha bytes then OR in 0xff000000 for each pixel.
    blended = _mm256_or_si256(_mm256_and_si256(blended, color_mask), alpha_mask);

    return blended;
}

template<class Shader>
static inline void
hline32rgba(int x0, int y0, int x1, int width, int height, Shader& shader) {
    unsigned int tmp;

    if (y0 >= 0 && y0 < height) {
        if (x0 < 0) {
            x0 = 0;
        } else if (x0 >= width) {
            return;
        }
        if (x1 < 0) {
            return;
        } else if (x1 >= width) {
            x1 = width - 1;
        }
        if (x0 <= x1) {
            int base_ptr_delta = (long int)(shader.background.data()) % 4;
            while ((base_ptr_delta + x0) % 4 != 0 && x0 <= x1) {
                shader.render_pixel(x0, y0);
                x0++;
            }
            // while (x0 + 8 <= x1) {
            //     // avx2 8x shading
            //     unsigned char* ptr = (unsigned char*)(shader.background.data() + y0 * width + x0);
            //     __m256i newVal = alphaBlendAVX2_SinglePath(
            //         ptr,
            //         shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a);
            //     _mm256_store_si256((__m256i*)ptr, newVal);
            //     x0 += 8;
            // }
            while (x0 + 4 <= x1) {
                // sse 4x shading
                unsigned char* ptr = (unsigned char*)(shader.background.data() + y0 * width + x0);
                __m128i newVal = alphaBlendSSE(
                    ptr,
                    shader.colour.r, shader.colour.g, shader.colour.b, shader.colour.a);
                _mm_store_si128((__m128i*)ptr, newVal);
                x0 += 4;
            }

            while (x0 <= x1) {
                shader.render_pixel(x0, y0);
                x0++;
            }
            // while (x0 <= x1) {
            //     shader.render_pixel(x0, y0);
            //     x0++;
            // }
        }
    }
}

template<class Shader>
static inline void
line32rgba(int x0, int y0, int x1, int y1, int width, int height, Shader& shader) {
    int i, n, e;
    int dx, dy;
    int xs, ys;

    /* normalize coordinates */
    dx = x1 - x0;
    if (dx < 0) {
        dx = -dx, xs = -1;
    } else {
        xs = 1;
    }
    dy = y1 - y0;
    if (dy < 0) {
        dy = -dy, ys = -1;
    } else {
        ys = 1;
    }

    n = (dx > dy) ? dx : dy;

    if (dx == 0) {
        /* vertical */
        for (i = 0; i < dy; i++) {
            point32rgba(x0, y0, width, height, shader);
            y0 += ys;
        }
    } else if (dy == 0) {
        /* horizontal */
        for (i = 0; i < dx; i++) {
            point32rgba(x0, y0, width, height, shader);
            x0 += xs;
        }
    } else if (dx > dy) {
        /* bresenham, horizontal slope */
        n = dx;
        dy += dy;
        e = dy - dx;
        dx += dx;

        for (i = 0; i < n; i++) {
            point32rgba(x0, y0, width, height, shader);
            if (e >= 0) {
                y0 += ys;
                e -= dx;
            }
            e += dy;
            x0 += xs;
        }
    } else {
        /* bresenham, vertical slope */
        n = dy;
        dx += dx;
        e = dx - dy;
        dy += dy;

        for (i = 0; i < n; i++) {
            point32rgba(x0, y0, width, height, shader);
            if (e >= 0) {
                x0 += xs;
                e -= dy;
            }
            e += dx;
            y0 += ys;
        }
    }
}

static int
x_cmp(const void *x0, const void *x1) {
    float diff = *((float *)x0) - *((float *)x1);
    if (diff < 0) {
        return -1;
    } else if (diff > 0) {
        return 1;
    } else {
        return 0;
    }
}

template<class Shader>
static void
draw_horizontal_lines(
    int n, Edge *e, int *x_pos, int y, int width, int height, Shader& shader
) {
    int i;
    for (i = 0; i < n; i++) {
        if (e[i].ymin == y && e[i].ymin == e[i].ymax) {
            int xmax;
            int xmin = e[i].xmin;
            if (*x_pos != -1 && *x_pos < xmin) {
                // Line would be after the current position
                continue;
            }

            xmax = e[i].xmax;
            if (*x_pos > xmin) {
                // Line would be partway through x_pos, so increase the starting point
                xmin = *x_pos;
                if (xmax < xmin) {
                    // Line would now end before it started
                    continue;
                }
            }

            hline32rgba(xmin, e[i].ymin, xmax, width, height, shader);
            *x_pos = xmax + 1;
        }
    }
}

/*
 * Filled polygon draw function using scan line algorithm.
 */
template<class Shader>
static inline int polygon32rgba(int n, Edge *e, int width, int height, Shader& shader) {
    Edge **edge_table;
    float *xx;
    int edge_count = 0;
    int ymin = height - 1;
    int ymax = 0;
    int i, j, k;
    float adjacent_line_x, adjacent_line_x_other_edge;

    if (n <= 0) {
        return 0;
    }

    /* Initialize the edge table and find polygon boundaries */
    /* malloc check ok, using calloc */
    edge_table = (Edge**) calloc(n, sizeof(Edge *));
    if (!edge_table) {
        return -1;
    }

    for (i = 0; i < n; i++) {
        if (ymin > e[i].ymin) {
            ymin = e[i].ymin;
        }
        if (ymax < e[i].ymax) {
            ymax = e[i].ymax;
        }
        if (e[i].ymin == e[i].ymax) {
            continue;
        }
        edge_table[edge_count++] = (e + i);
    }
    if (ymin < 0) {
        ymin = 0;
    }
    if (ymax > height) {
        ymax = height;
    }

    /* Process the edge table with a scan line searching for intersections */
    /* malloc check ok, using calloc */
    xx = (float*) calloc(edge_count * 2, sizeof(float));
    if (!xx) {
        free(edge_table);
        return -1;
    }
    for (; ymin <= ymax; ymin++) {
        j = 0;
        for (i = 0; i < edge_count; i++) {
            Edge *current = edge_table[i];
            if (ymin >= current->ymin && ymin <= current->ymax) {
                xx[j++] = (ymin - current->y0) * current->dx + current->x0;

                if (ymin == current->ymax && ymin < ymax) {
                    // Needed to draw consistent polygons
                    xx[j] = xx[j - 1];
                    j++;
                } else if (current->dx != 0 && j % 2 == 1 &&
                           roundf(xx[j - 1]) == xx[j - 1]) {
                    // Connect discontiguous corners
                    for (k = 0; k < i; k++) {
                        Edge *other_edge = edge_table[k];
                        if ((current->dx > 0 && other_edge->dx <= 0) ||
                            (current->dx < 0 && other_edge->dx >= 0)) {
                            continue;
                        }
                        // Check if the two edges join to make a corner
                        if (xx[j - 1] ==
                            (ymin - other_edge->y0) * other_edge->dx + other_edge->x0) {
                            // Determine points from the edges on the next row
                            // Or if this is the last row, check the previous row
                            int offset = ymin == ymax ? -1 : 1;
                            adjacent_line_x =
                                (ymin + offset - current->y0) * current->dx +
                                current->x0;
                            adjacent_line_x_other_edge =
                                (ymin + offset - other_edge->y0) * other_edge->dx +
                                other_edge->x0;
                            if (ymin == current->ymax) {
                                if (current->dx > 0) {
                                    xx[k] =
                                        fmax(
                                            adjacent_line_x, adjacent_line_x_other_edge
                                        ) +
                                        1;
                                } else {
                                    xx[k] =
                                        fmin(
                                            adjacent_line_x, adjacent_line_x_other_edge
                                        ) -
                                        1;
                                }
                            } else {
                                if (current->dx > 0) {
                                    xx[k] = fmin(
                                        adjacent_line_x, adjacent_line_x_other_edge
                                    );
                                } else {
                                    xx[k] =
                                        fmax(
                                            adjacent_line_x, adjacent_line_x_other_edge
                                        ) +
                                        1;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        qsort(xx, j, sizeof(float), x_cmp);
        int x_pos = j == 0 ? -1 : 0;
        for (i = 1; i < j; i += 2) {
            int x_end = ROUND_DOWN(xx[i]);
            if (x_end < x_pos) {
                // Line would be before the current position
                continue;
            }
            draw_horizontal_lines(n, e, &x_pos, ymin, width, height, shader);
            if (x_end < x_pos) {
                // Line would be before the current position
                continue;
            }

            int x_start = ROUND_UP(xx[i - 1]);
            if (x_pos > x_start) {
                // Line would be partway through x_pos, so increase the starting
                // point
                x_start = x_pos;
                if (x_end < x_start) {
                    // Line would now end before it started
                    continue;
                }
            }
            hline32rgba(x_start, ymin, x_end, width, height, shader);
            x_pos = x_end + 1;
        }
        draw_horizontal_lines(n, e, &x_pos, ymin, width, height, shader);
    }

    free(xx);
    free(edge_table);
    return 0;
}

static inline void
add_edge(Edge *e, int x0, int y0, int x1, int y1) {
    /* printf("edge %d %d %d %d\n", x0, y0, x1, y1); */

    if (x0 <= x1) {
        e->xmin = x0, e->xmax = x1;
    } else {
        e->xmin = x1, e->xmax = x0;
    }

    if (y0 <= y1) {
        e->ymin = y0, e->ymax = y1;
    } else {
        e->ymin = y1, e->ymax = y0;
    }

    if (y0 == y1) {
        e->d = 0;
        e->dx = 0.0;
    } else {
        e->dx = ((float)(x1 - x0)) / (y1 - y0);
        if (y0 == e->ymin) {
            e->d = 1;
        } else {
            e->d = -1;
        }
    }

    e->x0 = x0;
    e->y0 = y0;
}

/* -------------------------------------------------------------------- */
/* Interface                                                            */
/* -------------------------------------------------------------------- */

template<class Shader>
int ImagingDrawPolygon(int count, int *xy, int width, int height, Shader& shader) {
    int i, n, x0, y0, x1, y1;

    if (count <= 0) {
        return 0;
    }

    /* Build edge list */
    /* malloc check ok, using calloc */
    Edge *e = (Edge*) calloc(count, sizeof(Edge));
    if (!e) {
        return -1;
    }
    for (i = n = 0; i < count - 1; i++) {
        x0 = xy[i * 2];
        y0 = xy[i * 2 + 1];
        x1 = xy[i * 2 + 2];
        y1 = xy[i * 2 + 3];
        if (y0 == y1 && i != 0 && y0 == xy[i * 2 - 1]) {
            // This is a horizontal line,
            // that immediately follows another horizontal line
            Edge *last_e = &e[n - 1];
            if (x1 > x0 && x0 > xy[i * 2 - 2]) {
                // They are both increasing in x
                last_e->xmax = x1;
                continue;
            } else if (x1 < x0 && x0 < xy[i * 2 - 2]) {
                // They are both decreasing in x
                last_e->xmin = x1;
                continue;
            }
        }
        add_edge(&e[n++], x0, y0, x1, y1);
    }
    if (xy[i * 2] != xy[0] || xy[i * 2 + 1] != xy[1]) {
        add_edge(&e[n++], xy[i * 2], xy[i * 2 + 1], xy[0], xy[1]);
    }
    polygon32rgba(n, e, width, height, shader);
    free(e);

    return 0;
}

void rasterize_triangle_binned(const Triangle& tri, int width, int height, CompositOverShader& shader) {
    float pts[6] = {
        tri[0].x * width, tri[0].y * height,
        tri[1].x * width, tri[1].y * height,
        tri[2].x * width, tri[2].y * height
    };
    int pts_int[6] = {
        int(pts[0]), int(pts[1]),
        int(pts[2]), int(pts[3]),
        int(pts[4]), int(pts[5])
    };
    int xmin = std::min({pts_int[0], pts_int[2], pts_int[4]});
    int xmax = std::max({pts_int[0], pts_int[2], pts_int[4]});
    int ymin = std::min({pts_int[1], pts_int[3], pts_int[5]});
    int ymax = std::max({pts_int[1], pts_int[3], pts_int[5]});

    xmin = clampi(xmin, 0, width - 1);
    xmax = clampi(xmax + 1, 0, width - 1);
    ymin = clampi(ymin, 0, height - 1);
    ymax = clampi(ymax + 1, 0, height - 1);

    int tile_size = 4;

    for (int y_tile_start = ymin; y_tile_start < ymax; y_tile_start += tile_size) {
        for (int x_tile_start = xmin; x_tile_start < xmax; x_tile_start += tile_size) {
            // test four corners
            int hits = 0;

            float u,v;
            int x = x_tile_start;
            int y = y_tile_start;
            u = x / (float)width;
            v = y / (float)height;
            if (tri.is_inside(Point{u, v})) {
                hits++;
            }

            x += tile_size;
            u = x / (float)width;
            if (tri.is_inside(Point{u, v})) {
                hits++;
            }

            y += tile_size;
            v = y / (float)height;
            if (tri.is_inside(Point{u, v})) {
                hits++;
            }

            x -= tile_size;
            u = x / (float)width;
            if (tri.is_inside(Point{u, v})) {
                hits++;
            }

            if (hits == 0) {
                // check for a vertex inside this tile
                for (int i = 0; i < 3; i++) {
                    if (pts_int[2 * i + 0] >= x_tile_start && pts_int[2 * i + 0] <= x_tile_start + tile_size &&
                        pts_int[2 * i + 1] >= y_tile_start && pts_int[2 * i + 1] <= y_tile_start + tile_size) {
                        hits = 1;
                        break;
                    }
                }
                // no vertices inside this tile, skip
                if (hits == 0) {
                    continue;
                }
            } 
            
            if (hits == 4) {
                // shade all pixels inside the tile
                for (int y_idx = 0; y_idx < tile_size; y_idx++) {
                    for (int x_idx = 0; x_idx < tile_size; x_idx++) {
                        int x = x_tile_start + x_idx;
                        int y = y_tile_start + y_idx;
                        shader.render_pixel(x, y);
                    }
                }
            } else {
                // shade pixels one by one
                for (int y_idx = 0; y_idx < tile_size; y_idx++) {
                    for (int x_idx = 0; x_idx < tile_size; x_idx++) {
                        int x = x_tile_start + x_idx;
                        int y = y_tile_start + y_idx;
                        float u = (x + 0.5f) / (float)width;
                        float v = (y + 0.5f) / (float)height;
                        if (tri.is_inside(Point{u, v})) {
                            shader.render_pixel(x, y);
                        }
                    }
                }
            }
        }
    }
}
