#include <nanobind/nanobind.h>

#include "geometry/types.h"
#include "io/image.h"
#include "raster/rasterization.h"
#include "nb_image.h"

namespace nb = nanobind;

void init_rasterization(nb::module_& m)
{
    nb::enum_<raster::TriangleRasterizationMode>(m, "TriangleRasterizationMode")
        .value("Bounded", raster::TriangleRasterizationMode::Bounded)
        .value("Integer", raster::TriangleRasterizationMode::Integer)
        .value("ScanlinePolygon", raster::TriangleRasterizationMode::ScanlinePolygon);
    
    m.def("rasterize_triangle_onto_image", [](const geometry::triangle &triangle, PyRGBA py_colour, PyImageRGBA py_image, raster::TriangleRasterizationMode mode = raster::TriangleRasterizationMode::Default) {
        auto colour = depythonize_rgba255(py_colour);
        auto image = depythonize_imageview_rgba255(py_image);
        if (mode == raster::TriangleRasterizationMode::Bounded) {
            raster::rasterize_triangle_onto_image<raster::TriangleRasterizationMode::Bounded>(triangle, colour, image);
        } else if (mode == raster::TriangleRasterizationMode::Integer) {
            raster::rasterize_triangle_onto_image<raster::TriangleRasterizationMode::Integer>(triangle, colour, image);
        } else if (mode == raster::TriangleRasterizationMode::ScanlinePolygon) {
            raster::rasterize_triangle_onto_image<raster::TriangleRasterizationMode::ScanlinePolygon>(triangle, colour, image);
        } else {
            assert (0);
        }
    });
}
