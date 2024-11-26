#include <nanobind/nanobind.h>

#include "geometry/types.h"

namespace nb = nanobind;

void init_geometry_types(nb::module_& m);
void init_geometry_barycentrics(nb::module_& m);

void init_image_png(nb::module_& m); // TODO
void init_image_io(nb::module_& m); // TODO

void init_raster_composit(nb::module_& m); // TODO

NB_MODULE(SciencePy, m) {
    init_geometry_types(m);
    init_geometry_barycentrics(m);

//     init_image_png(m);
//     init_image_io(m);

//     init_raster_composit(m);
}
