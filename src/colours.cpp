#include "colours.h"

#include <cmath>

unsigned int l2_difference(const RGB255 &a, const RGB255 &b)
{
    return (a.r - b.r) * (a.r - b.r) + (a.g - b.g) * (a.g - b.g) + (a.b - b.b) * (a.b - b.b);
}

unsigned int l2_difference(const RGBA255 &a, const RGBA255 &b)
{
    return (a.r - b.r) * (a.r - b.r) + (a.g - b.g) * (a.g - b.g) + (a.b - b.b) * (a.b - b.b);
}

unsigned int l1_difference(const RGB255 &a, const RGB255 &b)
{
    return abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b);
}

unsigned int l1_difference(const RGBA255 &a, const RGBA255 &b)
{
    return abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b);
}

float l2_difference(const RGB01 &a, const RGB01 &b)
{
    return (a.r - b.r) * (a.r - b.r) + (a.g - b.g) * (a.g - b.g) + (a.b - b.b) * (a.b - b.b);
}

float l2_difference(const RGBA01 &a, const RGBA01 &b)
{
    return (a.r - b.r) * (a.r - b.r) + (a.g - b.g) * (a.g - b.g) + (a.b - b.b) * (a.b - b.b);
}

float l1_difference(const RGB01 &a, const RGB01 &b)
{
    return abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b);
}

float l1_difference(const RGBA01 &a, const RGBA01 &b)
{
    return abs(a.r - b.r) + abs(a.g - b.g) + abs(a.b - b.b);
}

RGB01 to_rgb01(const RGB255 &rgb)
{
    return RGB01{(float)rgb.r / 255, (float)rgb.g / 255, (float)rgb.b / 255};
}

RGBA01 to_rgba01(const RGBA255 &rgba)
{
    return RGBA01{(RGBA01::data_type)rgba.r / 255, (float)rgba.g / 255, (float)rgba.b / 255, (float)rgba.a / 255};
}

RGB255 to_rgb255(const RGB01 &rgb)
{
    return RGB255{(unsigned char)(rgb.r * 255.99), (unsigned char)(rgb.g * 255.99), (unsigned char)(rgb.b * 255.99)};
}

RGBA255 to_rgba255(const RGBA01 &rgba)
{
    return RGBA255{(unsigned char)(rgba.r * 255.99), (unsigned char)(rgba.g * 255.99), (unsigned char)(rgba.b * 255.99), (unsigned char)(rgba.a * 255.99)};
}
