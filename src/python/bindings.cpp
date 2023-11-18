#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>

#include "dna.h"
#include "evaluate_naive.h"

namespace nb = nanobind;

NB_MODULE(evoapp, m) {
    nb::class_<DNATri50::Primitive>(m, "Tri")
        .def(nb::init<>())
        .def_rw("vertices", &DNATri50::Primitive::vertices)
        .def_rw("r", &DNATri50::Primitive::r)
        .def_rw("g", &DNATri50::Primitive::g)
        .def_rw("b", &DNATri50::Primitive::b);

    nb::class_<DNATri50>(m, "DNA")
        .def(nb::init<>())
        .def_static("random", &DNATri50::gen_rand)
        .def_static("fromarray", [](nb::ndarray<float, nb::shape<9*50>, nb::c_contig, nb::device::cpu> arg) {
            return *(DNATri50*)(arg.data());
        })
        .def_rw("polys", &DNATri50::polys);

    m.def("render", [](const DNATri50& dna) {
        size_t shape[3] = { resolution, resolution, 3 };
        return nb::ndarray<nb::numpy, const float, nb::shape<2, nb::any>>(
            tri_render_cpu(dna), 3, shape);
    });

    m.def("loss", [](const DNATri50& dna, nb::ndarray<float, nb::shape<resolution, resolution, 3>, nb::c_contig, nb::device::cpu> arg) {
        return tri_loss_cpu(dna, (const float*) arg.data());
    });
}
