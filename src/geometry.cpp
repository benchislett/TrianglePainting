#include "geometry.h"

// https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
PURE Barycentric barycentric_coordinates(const Point& p, const Triangle& t) {
    float det = (t[1].y - t[2].y) * (t[0].x - t[2].x)
              + (t[2].x - t[1].x) * (t[0].y - t[2].y);
    float w = ((t[1].y - t[2].y) * (p.x - t[2].x)
             + (t[2].x - t[1].x) * (p.y - t[2].y)) / det;
    float u = ((t[2].y - t[0].y) * (p.x - t[2].x)
             + (t[0].x - t[2].x) * (p.y - t[2].y)) / det;
    float v = 1.0f - w - u;
    return {u, v, w};
}

bool Circle::is_inside(const Point& p) const {
    float dx = p.x - center.x;
    float dy = p.y - center.y;
    return dx * dx + dy * dy <= radius * radius;
}

bool Polygon::is_inside(const Point& p) const {
    bool inside = false;
    for (size_t i = 0, j = vertices.size() - 1; i < vertices.size(); j = i++) {
        // Check if the ray intersects the edge
        if (((vertices[i].y > p.y) != (vertices[j].y > p.y)) &&
            (p.x < (vertices[j].x - vertices[i].x) * (p.y - vertices[i].y) /
            (vertices[j].y - vertices[i].y) + vertices[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}

bool Triangle::is_inside(const Point& p) const {
    auto bary = barycentric_coordinates(p, *this);
    return bary.u >= 0 && bary.v >= 0 && bary.w >= 0;
}

Point Polygon::max() const {
    Point max = vertices[0];
    for (const auto& p : vertices) {
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
    }
    return max;
}

Point Polygon::min() const {
    Point min = vertices[0];
    for (const auto& p : vertices) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
    }
    return min;
}

Point Triangle::max() const {
    Point max = vertices[0];
    for (const auto& p : vertices) {
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
    }
    return max;
}

Point Triangle::min() const {
    Point min = vertices[0];
    for (const auto& p : vertices) {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
    }
    return min;
}
