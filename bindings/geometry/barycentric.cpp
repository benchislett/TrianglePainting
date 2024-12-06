#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>

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
using namespace nb::literals;

void init_geometry_barycentrics(nb::module_& m)
{
    nb::class_<geometry::barycentric>(m, "Barycentric")
        .def(nb::init<>())  // Default constructor
        .def_rw("u", &geometry::barycentric::u)
        .def_rw("v", &geometry::barycentric::v)
        .def_rw("w", &geometry::barycentric::w)
        .def("__repr__", &geometry::barycentric::__repr__);

    m.def("barycentric_coordinates", &geometry::barycentric_coordinates,
          "sample_point"_a, "triangle"_a);
}
