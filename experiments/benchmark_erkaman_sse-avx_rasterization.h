#pragma once

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <cassert>

// get SSE and AVX
#include <immintrin.h>
#include <xmmintrin.h>

#include "rasterize.h"
#include "utils.h"

struct vec2 {
public:
	float x;
	float y;

	vec2() : x(0.0f), y(0.0f) { }
	vec2(const float x_, const float y_) : x(x_), y(y_) { }
};

unsigned int rounddownAligned(unsigned int i, unsigned int align) {
	return (unsigned int)floor((float)i / (float)align) * align;
}

unsigned int roundupAligned(unsigned int i, unsigned int align) {
	return (unsigned int)ceil((float)i / (float)align) * align;
}

__m128 edgeFunctionSSE(const vec2 &a, const vec2 &b, __m128 cx, __m128 cy)
{
	return _mm_sub_ps(_mm_mul_ps(_mm_sub_ps(cx, _mm_set1_ps(a.x)), _mm_set1_ps(b.y - a.y)), _mm_mul_ps(_mm_sub_ps(cy, _mm_set1_ps(a.y)), _mm_set1_ps(b.x - a.x)));
}

__m256 edgeFunctionAVX(const vec2 &a, const vec2 &b, __m256 cx, __m256 cy)
{
	return _mm256_sub_ps(_mm256_mul_ps(_mm256_sub_ps(cx, _mm256_set1_ps(a.x)), _mm256_set1_ps(b.y - a.y)), _mm256_mul_ps(_mm256_sub_ps(cy, _mm256_set1_ps(a.y)), _mm256_set1_ps(b.x - a.x)));
}

float edgeFunction(const vec2 &a, const vec2 &b, const vec2 &c)
{
	// we are doing the reversed edge test, compared to the article.
	// we need to do it in this way, since our coordinate system has the origin in the top-left corner.
	return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void alphaBlendInPlace(unsigned char* buf, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
	buf[0] = (r * (255 - a) + (buf[0] * a)) / 255;
	buf[1] = (g * (255 - a) + (buf[1] * a)) / 255;
	buf[2] = (b * (255 - a) + (buf[2] * a)) / 255;
	assert(buf[3] == 255);
}

// __m128i alphaBlendSSE(unsigned char* buf, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
// 	const int invA = 255 - a;
// 	const int F_r = r * invA;
// 	const int F_g = g * invA;
// 	const int F_b = b * invA;

// 	const __m128i fg_const = _mm_setr_epi16(
// 		(short)F_r, (short)F_g, (short)F_b, 0,
// 		(short)F_r, (short)F_g, (short)F_b, 0);
	
// 	const __m128i alpha_vec = _mm_set1_epi16((short)a);

// 	const __m128i add128 = _mm_set1_epi16(128);

// 	const __m128i alpha_mask = _mm_set1_epi32(0xff000000);
// 	const __m128i color_mask = _mm_set1_epi32(0x00ffffff);

// 	__m128i bg = _mm_load_si128((__m128i*)(buf));

//     __m128i bg_lo = _mm_unpacklo_epi8(bg, _mm_setzero_si128());
//     __m128i bg_hi = _mm_unpackhi_epi8(bg, _mm_setzero_si128());

//     __m128i mult_lo = _mm_mullo_epi16(bg_lo, alpha_vec);
//     __m128i sum_lo = _mm_add_epi16(mult_lo, fg_const);
//     __m128i tmp_lo = _mm_add_epi16(sum_lo, add128);
//     __m128i tmp2_lo = _mm_srli_epi16(tmp_lo, 8);
//     __m128i blended_lo = _mm_srli_epi16(_mm_add_epi16(tmp_lo, tmp2_lo), 8);

//     __m128i mult_hi = _mm_mullo_epi16(bg_hi, alpha_vec);
//     __m128i sum_hi = _mm_add_epi16(mult_hi, fg_const);
//     __m128i tmp_hi = _mm_add_epi16(sum_hi, add128);
//     __m128i tmp2_hi = _mm_srli_epi16(tmp_hi, 8);
//     __m128i blended_hi = _mm_srli_epi16(_mm_add_epi16(tmp_hi, tmp2_hi), 8);

//     __m128i blended_pixels = _mm_packus_epi16(blended_lo, blended_hi);

//     blended_pixels = _mm_and_si128(blended_pixels, color_mask);
//     blended_pixels = _mm_or_si128(blended_pixels, alpha_mask);

//     return blended_pixels;
// }

// __m256i alphaBlendAVX(unsigned char* buf, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
//     // Precompute the inverse alpha and the constant terms for each channel:
//     // For each channel, we want: result = DIV255( bg_channel * a + fg_channel * (255 - a) )
//     const int invA = 255 - a;
//     const int F_r = r * invA;
//     const int F_g = g * invA;
//     const int F_b = b * invA;
    
//     // For processing 4 pixels at a time (each pixel: R, G, B, 0) pattern,
//     // we create a 128-bit constant that will be reused for both halves.
//     __m128i fg_const_sse = _mm_setr_epi16(
//         (short)F_r, (short)F_g, (short)F_b, 0,
//         (short)F_r, (short)F_g, (short)F_b, 0);
    
//     // Broadcast the foreground alpha (a) into all 16-bit lanes for the 4 pixels.
//     __m128i alpha_vec_sse = _mm_set1_epi16((short)a);
//     // Constant for our DIV255 approximation.
//     __m128i add128_sse = _mm_set1_epi16(128);
    
//     // Create 256-bit masks for forcing the output alpha to 255.
//     // Each pixel is 32-bit: we want to zero out the alpha and then OR in 0xff.
//     __m256i color_mask256 = _mm256_set1_epi32(0x00ffffff);
//     __m256i alpha_mask256 = _mm256_set1_epi32(0xff000000);
    
//     // Load 8 pixels (8 x 4 bytes = 32 bytes) from the buffer.
//     __m256i bg = _mm256_load_si256((__m256i*)buf);
    
//     // Split the 256-bit register into its lower and upper 128-bit halves (each holds 4 pixels).
//     __m128i bg_lo = _mm256_castsi256_si128(bg);
//     __m128i bg_hi = _mm256_extracti128_si256(bg, 1);
    
//     // Process the lower 128-bit half.
//     __m128i zero128 = _mm_setzero_si128();
//     // Unpack the lower 4 pixels from 8-bit to 16-bit (two groups: low and high parts).
//     __m128i bg_lo_lo = _mm_unpacklo_epi8(bg_lo, zero128);
//     __m128i bg_lo_hi = _mm_unpackhi_epi8(bg_lo, zero128);
    
//     // For the lower group: multiply background channels by 'a' and add the constant.
//     __m128i mult_lo_lo = _mm_mullo_epi16(bg_lo_lo, alpha_vec_sse);
//     __m128i sum_lo_lo = _mm_add_epi16(mult_lo_lo, fg_const_sse);
//     // Approximate division by 255: add 128, then add shifted result, then shift by 8.
//     __m128i tmp_lo_lo = _mm_add_epi16(sum_lo_lo, add128_sse);
//     __m128i tmp2_lo_lo = _mm_srli_epi16(tmp_lo_lo, 8);
//     __m128i blended_lo_lo = _mm_srli_epi16(_mm_add_epi16(tmp_lo_lo, tmp2_lo_lo), 8);
    
//     // Process the higher group within the lower half.
//     __m128i mult_lo_hi = _mm_mullo_epi16(bg_lo_hi, alpha_vec_sse);
//     __m128i sum_lo_hi = _mm_add_epi16(mult_lo_hi, fg_const_sse);
//     __m128i tmp_lo_hi = _mm_add_epi16(sum_lo_hi, add128_sse);
//     __m128i tmp2_lo_hi = _mm_srli_epi16(tmp_lo_hi, 8);
//     __m128i blended_lo_hi = _mm_srli_epi16(_mm_add_epi16(tmp_lo_hi, tmp2_lo_hi), 8);
    
//     // Pack the two groups (each 8 x 16-bit) back into 8-bit values: now we have 4 blended pixels.
//     __m128i blended_lo_128 = _mm_packus_epi16(blended_lo_lo, blended_lo_hi);
    
//     // Process the upper 128-bit half (the other 4 pixels) similarly.
//     __m128i bg_hi_lo = _mm_unpacklo_epi8(bg_hi, zero128);
//     __m128i bg_hi_hi = _mm_unpackhi_epi8(bg_hi, zero128);
    
//     __m128i mult_hi_lo = _mm_mullo_epi16(bg_hi_lo, alpha_vec_sse);
//     __m128i sum_hi_lo = _mm_add_epi16(mult_hi_lo, fg_const_sse);
//     __m128i tmp_hi_lo = _mm_add_epi16(sum_hi_lo, add128_sse);
//     __m128i tmp2_hi_lo = _mm_srli_epi16(tmp_hi_lo, 8);
//     __m128i blended_hi_lo = _mm_srli_epi16(_mm_add_epi16(tmp_hi_lo, tmp2_hi_lo), 8);
    
//     __m128i mult_hi_hi = _mm_mullo_epi16(bg_hi_hi, alpha_vec_sse);
//     __m128i sum_hi_hi = _mm_add_epi16(mult_hi_hi, fg_const_sse);
//     __m128i tmp_hi_hi = _mm_add_epi16(sum_hi_hi, add128_sse);
//     __m128i tmp2_hi_hi = _mm_srli_epi16(tmp_hi_hi, 8);
//     __m128i blended_hi_hi = _mm_srli_epi16(_mm_add_epi16(tmp_hi_hi, tmp2_hi_hi), 8);
    
//     __m128i blended_hi_128 = _mm_packus_epi16(blended_hi_lo, blended_hi_hi);
    
//     // Recombine the two 128-bit halves into one 256-bit register.
//     __m256i blended_pixels = _mm256_castsi128_si256(blended_lo_128);
//     blended_pixels = _mm256_inserti128_si256(blended_pixels, blended_hi_128, 1);
    
//     // Force the alpha channel to 255 for every pixel:
//     blended_pixels = _mm256_or_si256(
//         _mm256_and_si256(blended_pixels, color_mask256),
//         alpha_mask256);
        
//     return blended_pixels;
// }

void rasterizeTriangle(
	const float x0, const float y0,
	const float x1, const float y1,
	const float x2, const float y2,
	const unsigned int fbWidth, const unsigned int fbHeight,
	unsigned char* framebuffer,
	unsigned char r, unsigned char g, unsigned char b,
	unsigned char a
) {
	vec2 vcoords[3];
	vcoords[0] = vec2(x0, y0);
	vcoords[1] = vec2(x1, y1);
	vcoords[2] = vec2(x2, y2);

	// min of triangle AABB
	vec2 amin(+FLT_MAX, +FLT_MAX);
	// max of triangle AABB
	vec2 amax(-FLT_MAX, -FLT_MAX);

	// find triangle AABB.
	{
		for (int i = 0; i < 3; ++i) {
			vec2 p = vcoords[i];

			amin.x = min(p.x, amin.x);
			amin.y = min(p.y, amin.y);

			amax.x = max(p.x, amax.x);
			amax.y = max(p.y, amax.y);
		}

		// make sure we don't rasterize outside the framebuffer..
		amin.x = clamp(amin.x, -1.0f, +1.0f);
		amax.x = clamp(amax.x, -1.0f, +1.0f);

		amin.y = clamp(amin.y, -1.0f, +1.0f);
		amax.y = clamp(amax.y, -1.0f, +1.0f);
	}

	const int intAminx = clampi(float(rounddownAligned((unsigned int)((0.5f + 0.5f * amin.x)* fbWidth), 1)), 0, fbWidth - 1);
	const int intAmaxx = clampi(float(roundupAligned((unsigned int)((0.5f + 0.5f * amax.x)* fbWidth), 1)), 0, fbWidth - 1);
	const int intAminy = clampi(float((0.5f + 0.5f * amin.y)* fbHeight), 0, fbHeight - 1);
	const int intAmaxy = clampi(float((0.5f + 0.5f * amax.y)* fbHeight), 0, fbHeight - 1);

	const float doublePixelWidth = 2.0f / (float)fbWidth;
	const float doublePixelHeight = 2.0f / (float)fbHeight;

	vec2 p;

	for (int iy = intAminy; iy <= intAmaxy; iy += 1) {
		// map from [0,height] to [-1,+1]
		p.y = -1.0f + iy * doublePixelHeight;
		for (int ix = intAminx; ix <= intAmaxx; ix += 1) {
			// map from [0,width] to [-1,+1]
			p.x = -1.0f + ix * doublePixelWidth;

			float w0 = edgeFunction(vcoords[1], vcoords[2], p);
			float w1 = edgeFunction(vcoords[2], vcoords[0], p);
			float w2 = edgeFunction(vcoords[0], vcoords[1], p);

			// is it on the right side of all edges?
			if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
				unsigned int iBuf = (unsigned int)(iy * fbWidth + ix);
				unsigned char* ptr = framebuffer + iBuf * 4;
				ptr[0] = (r * (255 - a) + (ptr[0] * a)) / 255;
				ptr[1] = (g * (255 - a) + (ptr[1] * a)) / 255;
				ptr[2] = (b * (255 - a) + (ptr[2] * a)) / 255;
				ptr[3] = 255;
			}
		}
	}
}

void rasterizeTriangleSSE(
	const float x0, const float y0,
	const float x1, const float y1,
	const float x2, const float y2,
	const unsigned int fbWidth, const unsigned int fbHeight,
	unsigned char* framebuffer,
	unsigned char r, unsigned char g, unsigned char b,
	unsigned char a
) {
	vec2 vcoords[3];
	vcoords[0] = vec2(x0, y0);
	vcoords[1] = vec2(x1, y1);
	vcoords[2] = vec2(x2, y2);

	// min of triangle AABB
	vec2 amin(+FLT_MAX, +FLT_MAX);
	// max of triangle AABB
	vec2 amax(-FLT_MAX, -FLT_MAX);

	// find triangle AABB.
	{
		for (int i = 0; i < 3; ++i) {
			vec2 p = vcoords[i];

			amin.x = min(p.x, amin.x);
			amin.y = min(p.y, amin.y);

			amax.x = max(p.x, amax.x);
			amax.y = max(p.y, amax.y);
		}

		// make sure we don't rasterize outside the framebuffer..
		amin.x = clamp(amin.x, -1.0f, +1.0f);
		amax.x = clamp(amax.x, -1.0f, +1.0f);

		amin.y = clamp(amin.y, -1.0f, +1.0f);
		amax.y = clamp(amax.y, -1.0f, +1.0f);
	}

	// float where are all bits are set.
	float filledbitsfloat;
	{
		unsigned int ii = 0xffffffff;
		memcpy(&filledbitsfloat, &ii, sizeof(float));
	}
	float whitecolorfloat = filledbitsfloat;

	/*
	We'll be looping over all pixels in the AABB, and rasterize the pixels within the triangle. The AABB has been
	extruded on the x-axis, and aligned to 16bytes.
	This is necessary since _mm_store_ps can only write to 16-byte aligned addresses.
	*/
	const int intAminx = clampi(float(rounddownAligned((unsigned int)((0.5f + 0.5f * amin.x)* fbWidth), 1)), 0, fbWidth - 1);
	const int intAmaxx = clampi(float(roundupAligned((unsigned int)((0.5f + 0.5f * amax.x)* fbWidth), 1)), 0, fbWidth - 1);
	const int intAminy = clampi(float((0.5f + 0.5f * amin.y)* fbHeight), 0, fbHeight - 1);
	const int intAmaxy = clampi(float((0.5f + 0.5f * amax.y)* fbHeight), 0, fbHeight - 1);

	const float doublePixelWidth = 2.0f / (float)fbWidth;
	const float doublePixelHeight = 2.0f / (float)fbHeight;

	__m128 minusone = _mm_set1_ps(-1.0f);
	__m128 zero = _mm_setzero_ps();

	for (int iy = intAminy; iy <= intAmaxy; iy += 1) {
		// map from [0,height] to [-1,+1]
		__m128 py = _mm_add_ps(minusone, _mm_mul_ps(_mm_set1_ps(iy), _mm_set1_ps(doublePixelHeight)));

		for (int ix = (intAminx - (intAminx % 16)); ix <= intAmaxx; ix += 4) {
			// this `px` register contains the x-coords of four pixels in a row.
			// we map from [0,width] to [-1,+1]
			__m128 px = _mm_add_ps(minusone, _mm_mul_ps(
				_mm_set_ps(ix + 3.0f, ix + 2.0f, ix + 1.0f, ix + 0.0f), _mm_set1_ps(doublePixelWidth)));

			__m128 w0 = edgeFunctionSSE(vcoords[1], vcoords[2], px, py);
			__m128 w1 = edgeFunctionSSE(vcoords[2], vcoords[0], px, py);
			__m128 w2 = edgeFunctionSSE(vcoords[0], vcoords[1], px, py);

			// the default bitflag, results in all the four pixels being overwritten.
			__m128 writeFlag = _mm_set_ps1(filledbitsfloat);

			// the results of the edge tests are used to modify our bitflag.
			writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w0, zero));
			writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w1, zero));
			writeFlag = _mm_and_ps(writeFlag, _mm_cmpge_ps(w2, zero));

			unsigned int iBuf = (unsigned int)(iy * fbWidth + ix);

			__m128i origBufferVal = _mm_load_si128((__m128i*)(framebuffer + iBuf * 4));
			__m128i newBufferVal = alphaBlendSSE((unsigned char*)(framebuffer + (iBuf * 4)), r, g, b, a);
			/*
			We only want to write to pixels that are inside the triangle.
			However, implementing such a conditional write is tricky when dealing SIMD.

			We implement this by using a bitflag. This bitflag determines which of the four floats in __m128 should
			just write the old value to the buffer(meaning that the pixel is NOT actually rasterized),
			and which should overwrite the current value in the buffer(meaning that the pixel IS rasterized).

			This is implemented by some bitwise manipulation tricks.
			*/
			_mm_store_si128((__m128i*)(framebuffer+(4 * iBuf)),
				_mm_or_si128(
					_mm_and_si128(_mm_castps_si128(writeFlag), newBufferVal),
					_mm_andnot_si128(_mm_castps_si128(writeFlag), origBufferVal)
				));	
		}
	}
}

void rasterizeTriangleAVX(
	const float x0, const float y0,
	const float x1, const float y1,
	const float x2, const float y2,
	const unsigned int fbWidth, const unsigned int fbHeight,
	unsigned char* framebuffer,
	unsigned char r, unsigned char g, unsigned char b,
	unsigned char a
) {
	vec2 vcoords[3];
	vcoords[0] = vec2(x0, y0);
	vcoords[1] = vec2(x1, y1);
	vcoords[2] = vec2(x2, y2);

	// min of triangle AABB
	vec2 amin(+FLT_MAX, +FLT_MAX);
	// max of triangle AABB
	vec2 amax(-FLT_MAX, -FLT_MAX);

	// find triangle AABB.
	{
		for (int i = 0; i < 3; ++i) {
			vec2 p = vcoords[i];

			amin.x = min(p.x, amin.x);
			amin.y = min(p.y, amin.y);

			amax.x = max(p.x, amax.x);
			amax.y = max(p.y, amax.y);
		}

		// make sure we don't rasterize outside the framebuffer..
		amin.x = clamp(amin.x, -1.0f, +1.0f);
		amax.x = clamp(amax.x, -1.0f, +1.0f);

		amin.y = clamp(amin.y, -1.0f, +1.0f);
		amax.y = clamp(amax.y, -1.0f, +1.0f);
	}

	// float where are all bits are set.
	float filledbitsfloat;
	{
		unsigned int ii = 0xffffffff;
		memcpy(&filledbitsfloat, &ii, sizeof(float));
	}
	float whitecolorfloat = filledbitsfloat;

	const int intAminx = clampi(float(rounddownAligned((unsigned int)((0.5f + 0.5f * amin.x)* fbWidth), 1)), 0, fbWidth - 1);
	const int intAmaxx = clampi(float(roundupAligned((unsigned int)((0.5f + 0.5f * amax.x)* fbWidth), 1)), 0, fbWidth - 1);
	const int intAminy = clampi(float((0.5f + 0.5f * amin.y)* fbHeight), 0, fbHeight - 1);
	const int intAmaxy = clampi(float((0.5f + 0.5f * amax.y)* fbHeight), 0, fbHeight - 1);

	const float doublePixelWidth = 2.0f / (float)fbWidth;
	const float doublePixelHeight = 2.0f / (float)fbHeight;

	__m256 minusone = _mm256_set1_ps(-1.0f);
	__m256 zero = _mm256_setzero_ps();

	for (int iy = intAminy; iy <= intAmaxy; iy += 1) {
		// map from [0,height] to [-1,+1]
		__m256 py = _mm256_add_ps(minusone, _mm256_mul_ps(_mm256_set1_ps(iy), _mm256_set1_ps(doublePixelHeight)));

		for (int ix = (intAminx - (intAminx % 32)); ix <= intAmaxx; ix += 8) {
			// we map from [0,width] to [-1,+1]
			__m256 px = _mm256_add_ps(minusone, _mm256_mul_ps(
				_mm256_set_ps(ix + 7.0f, ix + 6.0f, ix + 5.0f, ix + 4.0f, ix + 3.0f, ix + 2.0f, ix + 1.0f, ix + 0.0f), _mm256_set1_ps(doublePixelWidth)));

			__m256 w0 = edgeFunctionAVX(vcoords[1], vcoords[2], px, py);
			__m256 w1 = edgeFunctionAVX(vcoords[2], vcoords[0], px, py);
			__m256 w2 = edgeFunctionAVX(vcoords[0], vcoords[1], px, py);

			__m256 writeFlag = _mm256_set1_ps(filledbitsfloat);

			// the results of the edge tests are used to modify our bitflag.
			writeFlag = _mm256_and_ps(writeFlag, _mm256_cmp_ps(w0, zero, _CMP_NLT_US));
			writeFlag = _mm256_and_ps(writeFlag, _mm256_cmp_ps(w1, zero, _CMP_NLT_US));
			writeFlag = _mm256_and_ps(writeFlag, _mm256_cmp_ps(w2, zero, _CMP_NLT_US));

			unsigned int iBuf = (unsigned int)(iy * fbWidth + ix);

			__m256i origBufferVal = _mm256_load_si256((__m256i*)(framebuffer + iBuf * 4));
			__m256i newBufferVal = alphaBlendAVX((unsigned char*)(framebuffer + (iBuf * 4)), r, g, b, a);

			_mm256_store_si256((__m256i*)(framebuffer+(4 * iBuf)),
				_mm256_or_si256(
					_mm256_and_si256(_mm256_castps_si256(writeFlag), newBufferVal),
					_mm256_andnot_si256(_mm256_castps_si256(writeFlag), origBufferVal)
				));

		}
	}
}

#include "benchmark_triangle_rasterization.h"

struct RasterImplPlain : public RasterImpl {
	ImageView<RGBA255> canvas;
	void set_canvas(ImageView<RGBA255> canvas) override {
		this->canvas = canvas;
	}
	void render(SampleInput sample) override {
		for (int i = 0; i < 6; i++) sample.triangle[i] = 2 * sample.triangle[i] - 1;
		rasterizeTriangle(
			sample.triangle[0], sample.triangle[1],
			sample.triangle[2], sample.triangle[3],
			sample.triangle[4], sample.triangle[5],
			canvas.width(), canvas.height(),
			(unsigned char*)canvas.data(),
			sample.colour_rgba[0], sample.colour_rgba[1],
			sample.colour_rgba[2], sample.colour_rgba[3]);
	}

	std::string name() const override {
        return "Erkaman Reference";
    }
};

struct RasterImplSSE : public RasterImpl {
	ImageView<RGBA255> canvas;
	void set_canvas(ImageView<RGBA255> canvas) override {
		this->canvas = canvas;
	}
	void render(SampleInput sample) override {
		for (int i = 0; i < 6; i++) sample.triangle[i] = 2 * sample.triangle[i] - 1;
		rasterizeTriangleSSE(
			sample.triangle[0], sample.triangle[1],
			sample.triangle[2], sample.triangle[3],
			sample.triangle[4], sample.triangle[5],
			canvas.width(), canvas.height(),
			(unsigned char*)canvas.data(),
			sample.colour_rgba[0], sample.colour_rgba[1],
			sample.colour_rgba[2], sample.colour_rgba[3]);
	}
	std::string name() const override {
        return "Erkaman SSE";
    }
};

struct RasterImplAVX : public RasterImpl {
	ImageView<RGBA255> canvas;
	void set_canvas(ImageView<RGBA255> canvas) override {
		this->canvas = canvas;
	}
	void render(SampleInput sample) override {
		for (int i = 0; i < 6; i++) sample.triangle[i] = 2 * sample.triangle[i] - 1;
		rasterizeTriangleAVX(
			sample.triangle[0], sample.triangle[1],
			sample.triangle[2], sample.triangle[3],
			sample.triangle[4], sample.triangle[5],
			canvas.width(), canvas.height(),
			(unsigned char*)canvas.data(),
			sample.colour_rgba[0], sample.colour_rgba[1],
			sample.colour_rgba[2], sample.colour_rgba[3]);
	}
	std::string name() const override {
        return "Erkaman AVX";
    }
};
