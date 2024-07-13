#include "geometry/barycentric.h"

namespace geometry2d {
    // https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    PURE barycentric barycentric_coordinates(const point& p, const triangle& t) {
        float det = (t.b.y - t.c.y) * (t.a.x - t.c.x) + (t.c.x - t.b.x) * (t.a.y - t.c.y);
        float w = ((t.b.y - t.c.y) * (p.x - t.c.x) + (t.c.x - t.b.x) * (p.y - t.c.y)) / det;
        float u = ((t.c.y - t.a.y) * (p.x - t.c.x) + (t.a.x - t.c.x) * (p.y - t.c.y)) / det;
        float v = 1.0f - w - u;
        return {u, v, w};
    }
};
