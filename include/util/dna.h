#pragma once

#include <array>
#include <tuple>
#include <cstdlib>
#include <ctime>

constexpr int resolution = 200;
constexpr float alpha = 0.5f;

inline float randf() {
    static unsigned int seed = 0;
    if (seed == 0) {
        seed = static_cast <unsigned int> (time(0));
        srand(seed);
    }

    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

template<int NumPolys, int PolyVerts>
struct DNA {
    struct Primitive {
        std::array<std::pair<float, float>, PolyVerts> vertices;
        float r, g, b;
    };

    std::array<Primitive, NumPolys> polys;

    static inline DNA gen_rand() {
        DNA out;
        for (int i = 0; i < NumPolys * (sizeof (Primitive) / sizeof (float)); i++) {
            ((float*)(&out))[i] = randf();
        }
        return out;
    }
};

using DNATri50 = DNA<50, 3>;
