#include <nanobind/nanobind.h>

#include "shaders.h"
#include "nb_image.h"

namespace nb = nanobind;

void init_shaders(nb::module_& m) {
    nb::class_<CompositOverShader>(m, "CompositOverShader")
        .def("__init__", [](CompositOverShader* self, const PyImageRGBA& bg, const PyRGBA& col) {
            new (self) CompositOverShader(depythonize_imageview_rgba255(bg), depythonize_rgba255(col));
        })
        .def_prop_rw("background", [](const CompositOverShader& self) -> PyImageRGBA {
            return pythonize_imageview_rgba255(self.background);
        }, [](CompositOverShader& self, PyImageRGBA& bg) {
            self.background = depythonize_imageview_rgba255(bg);
        })
        .def_prop_rw("colour", [](const CompositOverShader& self) -> PyRGBA {
            return pythonize_rgba255(self.colour);
        }, [](CompositOverShader& self, const PyRGBA& col) {
            self.colour = depythonize_rgba255(col);
        });

    nb::class_<OptimalColourShader>(m, "OptimalColourShader")
        .def("__init__",
             [](OptimalColourShader* self,
                unsigned char alpha,
                const PyImageRGBA& t,
                const PyImageRGBA& fg,
                const PyImageRGBA& bg,
                const nb::ndarray<int, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>& err) {
                 new (self) OptimalColourShader(
                     alpha,
                     depythonize_imageview_rgba255(t),
                     depythonize_imageview_rgba255(fg),
                     depythonize_imageview_rgba255(bg),
                     ImageView<int>((int*) err.data(), err.shape(0), err.shape(1)));
             })
        .def_prop_rw("target",
                     [](const OptimalColourShader& self) {
                         return pythonize_imageview_rgba255(self.target);
                     },
                     [](OptimalColourShader& self, PyImageRGBA& t) {
                         self.target = depythonize_imageview_rgba255(t);
                     })
        .def_prop_rw("foreground",
                     [](const OptimalColourShader& self) {
                         return pythonize_imageview_rgba255(self.foreground);
                     },
                     [](OptimalColourShader& self, PyImageRGBA& fg) {
                         self.foreground = depythonize_imageview_rgba255(fg);
                     })
        .def_prop_rw("background",
                     [](const OptimalColourShader& self) {
                         return pythonize_imageview_rgba255(self.background);
                     },
                     [](OptimalColourShader& self, PyImageRGBA& bg) {
                         self.background = depythonize_imageview_rgba255(bg);
                     })
        .def_prop_rw("error_mask",
                     [](const OptimalColourShader& self) {
                         // No direct pythonize for int-type image, so omitted
                         return 0; // Placeholder
                     },
                     [](OptimalColourShader& self,
                        const nb::ndarray<int, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>& err) {
                         self.error_mask = ImageView<int>((int*) err.data(), err.shape(0), err.shape(1));
                     })
        .def_prop_rw("current_alpha",
                     [](const OptimalColourShader& self) { return self.current_alpha; },
                     [](OptimalColourShader& self, unsigned char alpha) { self.current_alpha = alpha; })
        .def("final_colour_and_error",
             [](OptimalColourShader& self) {
                 auto [col, err] = self.final_colour_and_error();
                 return std::make_tuple(pythonize_rgba255(col), err);
             });

    nb::class_<DrawLossFGShader>(m, "DrawLossFGShader")
        .def("__init__",
             [](DrawLossFGShader* self,
                const PyImageRGBA& t,
                const PyImageRGBA& fg,
                const PyImageRGBA& bg,
                const PyRGBA& col,
                const nb::ndarray<int, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>& err) {
                 new (self) DrawLossFGShader(
                     depythonize_imageview_rgba255(t),
                     depythonize_imageview_rgba255(fg),
                     depythonize_imageview_rgba255(bg),
                     depythonize_rgba255(col),
                     ImageView<int>((int*) err.data(), err.shape(0), err.shape(1)));
             })
        .def_prop_rw("target",
                     [](const DrawLossFGShader& self) {
                         return pythonize_imageview_rgba255(self.target);
                     },
                     [](DrawLossFGShader& self, const PyImageRGBA& t) {
                         self.target = depythonize_imageview_rgba255(t);
                     })
        .def_prop_rw("foreground",
                     [](const DrawLossFGShader& self) {
                         return pythonize_imageview_rgba255(self.foreground);
                     },
                     [](DrawLossFGShader& self, const PyImageRGBA& fg) {
                         self.foreground = depythonize_imageview_rgba255(fg);
                     })
        .def_prop_rw("background",
                     [](const DrawLossFGShader& self) {
                         return pythonize_imageview_rgba255(self.background);
                     },
                     [](DrawLossFGShader& self, const PyImageRGBA& bg) {
                         self.background = depythonize_imageview_rgba255(bg);
                     })
        .def_prop_rw("colour",
                     [](const DrawLossFGShader& self) { return pythonize_rgba255(self.colour); },
                     [](DrawLossFGShader& self, const PyRGBA& col) {
                         self.colour = depythonize_rgba255(col);
                     })
        .def_prop_rw("error_mask",
                     [](const DrawLossFGShader& self) {
                         // No direct pythonize for int-type image, so omitted
                         return 0; // Placeholder
                     },
                     [](DrawLossFGShader& self,
                        const nb::ndarray<int, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>& err) {
                         self.error_mask = ImageView<int>((int*) err.data(), err.shape(0), err.shape(1));
                     })
        .def_prop_ro("total_error",
                     [](const DrawLossFGShader& self) { return self.total_error; });
}
