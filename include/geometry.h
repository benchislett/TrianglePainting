#pragma once

#include <array>
#include <vector>
#include <string>

#include "common.h"

struct Point {
    float x, y;
};

enum class ShapeType {
    Circle,
    Triangle,
    Polygon
};

struct Shape {
    virtual ShapeType type() const = 0;
    virtual bool is_inside(const Point& p) const = 0;
    virtual Point max() const = 0;
    virtual Point min() const = 0;
};

struct Circle : public Shape {
    Point center;
    float radius;

    ShapeType type() const override { return ShapeType::Circle; }
    bool is_inside(const Point& p) const override;
    Point max() const override { return {center.x + radius, center.y + radius}; }
    Point min() const override { return {center.x - radius, center.y - radius}; }
};

struct Polygon : public Shape {
    std::vector<Point> vertices;

    Point& operator[](int i) { return vertices[i]; }
    const Point& operator[](int i) const { return vertices[i]; }

    ShapeType type() const override { return ShapeType::Polygon; }
    bool is_inside(const Point& p) const override;
    int num_vertices() const { return vertices.size(); }

    Point max() const override;
    Point min() const override;
};

struct Triangle : public Shape {
    std::array<Point, 3> vertices;

    Point& operator[](int i) { return vertices[i]; }
    const Point& operator[](int i) const { return vertices[i]; }

    ShapeType type() const override { return ShapeType::Triangle; }
    bool is_inside(const Point& p) const override;
    int num_vertices() const { return 3; }

    Point max() const override;
    Point min() const override;
};

struct Barycentric {
    float u, v, w;
};

PURE Barycentric barycentric_coordinates(const Point& p, const Triangle& t);
