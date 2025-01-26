#include <nanobind/nanobind.h>

#include "nb_image.h"

namespace nb = nanobind;

void init_geometry(nb::module_& m);
void init_rasterization(nb::module_& m);
void init_shaders(nb::module_& m);

NB_MODULE(polypaint, m) {
    init_geometry(m);
    init_rasterization(m);
    init_shaders(m);
}
