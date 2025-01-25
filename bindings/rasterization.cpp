#include <nanobind/nanobind.h>

#include "colours.h"
#include "geometry.h"
#include "image.h"
#include "rasterize.h"
#include "nb_image.h"
#include "shaders.h"

namespace nb = nanobind;

enum RasterStrategy {
    Bounded,
    Integer,
    ScanlinePolygon,
    Default = Bounded
};

void init_rasterization(nb::module_& m)
{
    nb::enum_<RasterStrategy>(m, "RasterStrategy")
        .value("Bounded", RasterStrategy::Bounded)
        .value("Integer", RasterStrategy::Integer)
        .value("ScanlinePolygon", RasterStrategy::ScanlinePolygon);
    
    m.def("rasterize", [](const Triangle &triangle, const PyRGBA &py_colour, PyImageRGBA py_image, RasterStrategy mode = RasterStrategy::Default) {
        auto colour = depythonize_rgba255(py_colour);
        auto image = depythonize_imageview_rgba255(py_image);
        CompositOverShader shader(image, colour);
        if (mode == RasterStrategy::Bounded) {
            rasterize_triangle_bounded(triangle, image.width(), image.height(), shader);
        } else if (mode == RasterStrategy::Integer) {
            rasterize_triangle_integer(triangle, image.width(), image.height(), shader);
        } else if (mode == RasterStrategy::ScanlinePolygon) {
            rasterize_polygon_scanline(triangle, image.width(), image.height(), shader);
        } else {
            assert (0);
        }
    });
}
