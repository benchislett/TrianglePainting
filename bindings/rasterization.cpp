#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "colours.h"
#include "geometry.h"
#include "image.h"
#include "rasterize.h"
#include "nb_image.h"
#include "shaders.h"

namespace nb = nanobind;

void init_rasterization(nb::module_& m)
{
    nb::enum_<RasterStrategy>(m, "RasterStrategy")
        .value("Bounded", RasterStrategy::Bounded)
        .value("Integer", RasterStrategy::Integer)
        .value("ScanlinePolygon", RasterStrategy::ScanlinePolygon);

    nb::class_<RasterConfig>(m, "RasterConfig")
        .def(nb::init<RasterStrategy, int, int>())
        .def_rw("strategy", &RasterConfig::strategy)
        .def_rw("image_width", &RasterConfig::image_width)
        .def_rw("image_height", &RasterConfig::image_height);

    m.def("rasterize", [](std::shared_ptr<Shape> shape, CompositOverShader &shader, const RasterConfig &config) {
        rasterize<CompositOverShader>(shape, shader, config);
    });

    m.def("rasterize", [](std::shared_ptr<Shape> shape, OptimalColourShader &shader, const RasterConfig &config) {
        rasterize<OptimalColourShader>(shape, shader, config);
    });

    m.def("rasterize", [](std::shared_ptr<Shape> shape, DrawLossFGShader &shader, const RasterConfig &config) {
        rasterize<DrawLossFGShader>(shape, shader, config);
    });
}
