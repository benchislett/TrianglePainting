#pragma once

#include <string>

namespace geometry {
    struct point {
        float x, y;

        std::string __repr__() const {
            return "<Point x=" + std::to_string(x) + ", y=" + std::to_string(y) + ">";
        }
    };
    
    struct triangle {
        point a, b, c;

        point& operator[](int i) {
            if (i == 0) {
                return a;
            } else if (i == 1) {
                return b;
            } else {
                return c;
            }
        }

        const point& operator[](int i) const {
            if (i == 0) {
                return a;
            } else if (i == 1) {
                return b;
            } else {
                return c;
            }
        }

        std::string __repr__() const {
            return "<Triangle a=" + a.__repr__() + ", b=" + b.__repr__() + ", c=" + c.__repr__() + ">";
        }
    };

    struct circle {
        point center;
        float radius;

        std::string __repr__() const {
            return "<Circle center=" + center.__repr__() + ", radius=" + std::to_string(radius) + ">";
        }
    };
};
