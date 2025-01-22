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
