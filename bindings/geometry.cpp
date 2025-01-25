#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <format>

#include "geometry.h"

namespace nb = nanobind;

void init_geometry(nb::module_& m)
{
    nb::class_<Point>(m, "Point")
        .def(nb::init<float, float>())
        .def_rw("x", &Point::x)
        .def_rw("y", &Point::y)
        .def("__repr__", [](const Point &p) {
            return std::format("Point(x={:.2f}, y={:.2f})", p.x, p.y);
        });

    nb::class_<Triangle>(m, "Triangle")
        .def("__init__", [](Triangle* t, const Point& a, const Point& b, const Point& c) {
            new (t) Triangle{}; 
            t->points = {a, b, c};
        })
        .def("__getitem__", [](Triangle &t, int i) {
            return t[i];
        })
        .def("__setitem__", [](Triangle &t, int i, const Point &p) {
            t[i] = p;
        })
        .def("__repr__", [](const Triangle &t) {
            return std::format("Triangle(v0=({:.2f}, {:.2f}), v1=({:.2f}, {:.2f}), v2=({:.2f}, {:.2f}))",
                   t[0].x, t[0].y,
                   t[1].x, t[1].y,
                   t[2].x, t[2].y);
        });
    
    nb::class_<Circle>(m, "Circle")
        .def("__init__", [](Circle* t, const Point& center, float radius) {
            new (t) Circle{}; 
            t->center = center;
            t->radius = radius;
        })
        .def_rw("center", &Circle::center)
        .def_rw("radius", &Circle::radius)
        .def("__repr__", [](const Circle &c) {
            return std::format("Circle(center=({:.2f}, {:.2f}), radius={:.2f})", c.center.x, c.center.y, c.radius);
        });
}
