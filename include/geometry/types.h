#pragma once

namespace geometry2d {
    struct point {
        float x, y;
    };
    
    struct triangle {
        point a, b, c;
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
