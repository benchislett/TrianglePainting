#pragma once

#include <immintrin.h>
#include <xmmintrin.h>

int clampi(int i, int min, int max) {
	if (i < min) {
		return min;
	}
	else if (i > max) {
		return max;
	}
	else {
		return i;
	}
}

float clamp(float f, float min, float max) {
	if (f < min) {
		return min;
	}
	else if (f > max) {
		return max;
	}
	else {
		return f;
	}
}

float min(float x, float y) {
	return x < y ? x : y;
}

float max(float x, float y) {
	return x > y ? x : y;
}

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