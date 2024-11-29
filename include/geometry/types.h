#pragma once

namespace geometry {
    struct point {
        float x, y;
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
    };

    struct circle {
        point center;
        float radius;
    };
};

namespace geometry3d {
    struct point {
        float x, y, z;
    };
    
    struct triangle {
        point a, b, c;
    };

    struct sphere {
        point center;
        float radius;
    };
};
