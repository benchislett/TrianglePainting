#include <nanobind/nanobind.h>

#include "nb_image.h"

namespace nb = nanobind;

void init_geometry_types(nb::module_& m);
void init_geometry_barycentrics(nb::module_& m);

void init_io_png_rw(nb::module_& m);

void init_raster_composit(nb::module_& m); // TODO

NB_MODULE(polypaint, m) {
    init_geometry_types(m);
    init_geometry_barycentrics(m);

    init_io_png_rw(m);

    // init_raster_composit(m);
}
