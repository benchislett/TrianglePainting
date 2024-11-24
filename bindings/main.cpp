#include <nanobind/nanobind.h>

#include "geometry/types.h"

namespace nb = nanobind;

NB_MODULE(SciencePy, m) {
    // bind geometry2d::triangle as Triangle
    nb::class_<geometry2d::triangle>(m, "Triangle")
        .def(nb::init<>());
}
