#pragma once

#include <array>
#include <string>

#include "common.h"

struct Point {
    float x, y;
};

struct Shape {};

struct Circle : public Shape {
    Point center;
    float radius;
};

template<int N>
struct Polygon : public Shape {
    std::array<Point, N> points;

    Point& operator[](int i);
    const Point& operator[](int i) const;
};

struct Triangle : public Polygon<3> {};

struct Barycentric {
    float u, v, w;
};

PURE Barycentric barycentric_coordinates(const Point& p, const Triangle& t);
