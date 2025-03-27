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