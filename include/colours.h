#pragma once

struct RGB01 {
    constexpr static int channels = 3;
    using data_type = float;

    float r, g, b;
};

struct RGBA01 {
    constexpr static int channels = 4;
    using data_type = float;

    float r, g, b, a;
};

struct RGB255 {
    constexpr static int channels = 3;
    using data_type = unsigned char;

    unsigned char r, g, b;
};

struct RGBA255 {
    constexpr static int channels = 4;
    using data_type = unsigned char;

    unsigned char r, g, b, a;
};

unsigned int l2_difference(const RGB255& a, const RGB255& b);
unsigned int l2_difference(const RGBA255& a, const RGBA255& b);
unsigned int l1_difference(const RGB255& a, const RGB255& b);
unsigned int l1_difference(const RGBA255& a, const RGBA255& b);

float l2_difference(const RGB01& a, const RGB01& b);
float l2_difference(const RGBA01& a, const RGBA01& b);
float l1_difference(const RGB01& a, const RGB01& b);
float l1_difference(const RGBA01& a, const RGBA01& b);

RGB01 to_rgb01(const RGB255& rgb);
RGBA01 to_rgba01(const RGBA255& rgba);
RGB255 to_rgb255(const RGB01& rgb);
RGBA255 to_rgba255(const RGBA01& rgba);
