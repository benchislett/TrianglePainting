#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "geometry/types.h"

namespace nb = nanobind;

void init_geometry_types(nb::module_& m)
{
    nb::class_<geometry::point>(m, "Point")
        .def(nb::init<float, float>())
        .def_rw("x", &geometry::point::x)
        .def_rw("y", &geometry::point::y)
        .def("__repr__", &geometry::point::__repr__);

    nb::class_<geometry::triangle>(m, "Triangle")
        .def(nb::init<geometry::point, geometry::point, geometry::point>())  // Default constructor
        .def_rw("a", &geometry::triangle::a)
        .def_rw("b", &geometry::triangle::b)
        .def_rw("c", &geometry::triangle::c)
        .def("__getitem__", [](geometry::triangle &t, int i) {
            return t[i];
        })
        .def("__setitem__", [](geometry::triangle &t, int i, const geometry::point &p) {
            t[i] = p;
        })
        .def("__repr__", &geometry::triangle::__repr__);

    nb::class_<geometry::circle>(m, "Circle")
        .def(nb::init<>())  // Default constructor
        .def_rw("center", &geometry::circle::center)
        .def_rw("radius", &geometry::circle::radius)
        .def("__repr__", &geometry::circle::__repr__);
}
