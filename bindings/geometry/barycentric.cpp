#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include "geometry/types.h"
#include "geometry/barycentric.h"

#include <tuple>
#include <utility>

std::tuple<float, float, float> barycentrics(float px, float py, float t1x, float t1y, float t2x, float t2y, float t3x, float t3y) {
    geometry::point p = {px, py};
    geometry::triangle t = {{t1x, t1y}, {t2x, t2y}, {t3x, t3y}};
    auto b = geometry::barycentric_coordinates(p, t);
    return {b.u, b.v, b.w};
}

namespace nb = nanobind;

void init_geometry_barycentrics(nb::module_& m)
{
    m.def("barycentrics", &barycentrics);
}
