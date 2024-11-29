#pragma once

#include "common.h"
#include "geometry/types.h"

namespace geometry {
    /*
    Barycentric coordinates in the plane relative to a triangle
    Parameterized such that `P = w * A + u * B + v * C`
    */
    struct barycentric {
        float u, v, w;
    };

    PURE barycentric barycentric_coordinates(const point& p, const triangle& t);
};
