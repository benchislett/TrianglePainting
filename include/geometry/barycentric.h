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

        std::string __repr__() const {
            return "<Barycentric u=" + std::to_string(u) + 
                    ", v=" + std::to_string(v) + 
                    ", w=" + std::to_string(w) + ">";
        }
    };

    PURE barycentric barycentric_coordinates(const point& p, const triangle& t);
};
