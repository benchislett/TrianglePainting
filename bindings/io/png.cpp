#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "io/image.h"
#include "io/png.h"

#include "nb_image.h"

#include <tuple>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

void init_io_png_rw(nb::module_& m)
{
    m.def("load_png_rgb", [](const std::string& filename) {
        return pythonize_imageview_rgb255(io::load_png_rgb(filename).release_view());
    });

    m.def("load_png_rgba", [](const std::string& filename) {
        return pythonize_imageview_rgba255(io::load_png_rgba(filename).release_view());
    });

    m.def("save_png_rgb", [](const std::string& filename, PyImageRGB image) {
        io::save_png_rgb(filename, depythonize_imageview_rgb255(image));
    });

    m.def("save_png_rgba", [](const std::string& filename, PyImageRGBA image) {
        io::save_png_rgba(filename, depythonize_imageview_rgba255(image));
    });
}
