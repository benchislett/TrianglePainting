#pragma once

#include "util_cuda.h"
#include "misc_math.h"

constexpr int resolution = 200;
constexpr float alpha = 0.5f;

constexpr int target_image_size = resolution * resolution * 3;

template<int NVertices>
struct Polygon {
    array<float, NVertices> verts_x;
    array<float, NVertices> verts_y;

    MISC_SYMS Polygon() : verts_x{}, verts_y{} {}
    MISC_SYMS Polygon(const array<float, NVertices>& vxs, const array<float, NVertices>& vys) : verts_x(vxs), verts_y(vys) {}
    MISC_SYMS Polygon(const array<pair<float, float>, NVertices> vertices) {
        for (int i = 0; i < NVertices; i++) {
            auto [x, y] = vertices[i];
            verts_x[i] = x;
            verts_y[i] = y;
        }
    }

    MISC_SYMS pair<float, float> getVertex(int i) const {
        return make_pair(verts_x[i], verts_y[i]);
    }

    MISC_SYMS void setVertex(int i, float x, float y) {
        verts_x[i] = x;
        verts_y[i] = y;
    }

    MISC_SYMS void setVertex(int i, pair<float, float> v) {
        setVertex(i, v.first, v.second);
    }

    MISC_SYMS array<pair<float, float>, NVertices> transpose() const {
        array<pair<float, float>, NVertices> vertices;
        for (int i = 0; i < NVertices; i++) {
            vertices[i] = getVertex(i);
        }
        return vertices;
    }

    MISC_SYMS bool test(float u, float v) const {
        return pnpoly<NVertices>(u, v, &verts_x[0], &verts_y[0]);
    }

    static MISC_SYMS constexpr int params() {
        return 2 * NVertices;
    }
};

template<int PolyVerts>
struct Primitive {
    Polygon<PolyVerts> poly;
    float r, g, b;

    static MISC_SYMS constexpr int params() {
        return Polygon<PolyVerts>::params() + 3;
    }
};

template<int NumPolys, int PolyVerts>
struct DNA {
    using DNAPrimT = Primitive<PolyVerts>;

    array<DNAPrimT, NumPolys> primitives;

    MISC_SYMS DNA() {}
    MISC_SYMS DNA(const DNA& d) : primitives(d.primitives) {}

    static MISC_SYMS constexpr int params() {
        return NumPolys * DNAPrimT::params();
    }
};

constexpr int NPoly = 50;
constexpr int NVert = 3;

using PolyT = Polygon<NVert>;
using PrimT = Primitive<NVert>;
using DNAT = DNA<NPoly, NVert>;
