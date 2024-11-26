#include <nanobind/nanobind.h>

#include "geometry/types.h"

namespace nb = nanobind;

void init_geometry_types(nb::module_& m)
{
    nb::class_<geometry2d::point>(m, "Point")
        .def(nb::init<float, float>())
        .def_rw("x", &geometry2d::point::x)
        .def_rw("y", &geometry2d::point::y);
}
