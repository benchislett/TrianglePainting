#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>

#include "dna.h"
#include "evaluate_naive.h"
#include "evaluate_cuda.h"

namespace nb = nanobind;

NB_MODULE(evoapp, m) {
    nb::class_<PolyT>(m, "Polygon")
        .def("__init__", ([](PolyT* t, nb::ndarray<float, nb::shape<2, NVert>, nb::c_contig, nb::device::cpu> arg) {
            struct _tmp {
                array<float, NVert> vxs;
                array<float, NVert> vys;
            } tmp;
            tmp = *(const struct _tmp*)(arg.data());
            new (t) PolyT(tmp.vxs, tmp.vys);
        }))
        .def("__init__", ([](PolyT* t, nb::ndarray<float, nb::shape<NVert, 2>, nb::c_contig, nb::device::cpu> arg) {
            auto tmp = *(const array<pair<float, float>, NVert>*)(arg.data());
            new (t) PolyT(tmp);
        }))
        .def("getVertex", &PolyT::getVertex)
        .def("setVertex", [](PolyT& p, int i, float x, float y) {
            p.setVertex(i, x, y);
        })
        .def("setVertex", [](PolyT& p, int i, pair<float, float> vert) {
            p.setVertex(i, vert);
        })
        .def("transpose", &PolyT::transpose)
        .def("test", &PolyT::test)
        .def_rw("verts_x", &PolyT::verts_x)
        .def_rw("verts_y", &PolyT::verts_y)
        .def_static("params", &PolyT::params);

    nb::class_<PrimT>(m, "Primitive")
        .def(nb::init<>())
        .def_rw("poly", &PrimT::poly)
        .def_rw("r", &PrimT::r)
        .def_rw("g", &PrimT::g)
        .def_rw("b", &PrimT::b)
        .def_static("params", &PrimT::params);
    
    nb::class_<DNAT>(m, "DNA")
        .def(nb::init<>())
        .def(nb::init<DNAT>())
        .def_rw("primitives", &DNAT::primitives)
        .def_static("params", &DNAT::params)
        .def_static("fromarray", [](nb::ndarray<float, nb::shape<DNAT::params()>, nb::c_contig, nb::device::cpu> arg) {
            DNAT res;
            int idx = 0;
            const float* data = arg.data();
            int stride = arg.stride(0);
            for (int i = 0; i < NPoly; i++) {
                Primitive<NVert> &p = res.primitives[i];
                for (int j = 0; j < NVert; j++) { p.poly.verts_x[j] = data[idx]; idx += stride; p.poly.verts_y[j] = data[idx]; idx += stride; }
                p.r = data[idx]; idx += stride;
                p.g = data[idx]; idx += stride;
                p.b = data[idx]; idx += stride;
            }
            return res;
        })
        .def("toarray", [](const DNAT& dna) {
            float* data = (float*) malloc(DNAT::params() * sizeof(float));
            int idx = 0;
            for (int i = 0; i < NPoly; i++) {
                Primitive p = dna.primitives[i];
                for (int j = 0; j < NVert; j++) { data[idx++] = p.poly.verts_x[j]; data[idx++] = p.poly.verts_y[j]; }
                data[idx++] = p.r; data[idx++] = p.g; data[idx++] = p.b;
            }
            size_t shape[1] = { DNAT::params() };
            return nb::ndarray<nb::numpy, float>(
                data, 1, shape);
        });

    m.def("render", [](const DNAT& dna) {
        size_t shape[3] = { resolution, resolution, 3 };
        return nb::ndarray<nb::numpy, const float, nb::shape<2, nb::any>>(
            tri_render_cpu(dna), 3, shape);
    });

#ifdef HAS_CUDA
    m.def("render_gpu", [](const DNAT& dna) {
        size_t shape[3] = { resolution, resolution, 3 };
        return nb::ndarray<nb::numpy, const float, nb::shape<2, nb::any>>(
            tri_render_gpu(dna), 3, shape);
    });
#endif

    nb::class_<LossStateCPU>(m, "LossStateCPU")
        .def("__init__", ([](LossStateCPU* t, nb::ndarray<float, nb::shape<resolution, resolution, 3>, nb::c_contig, nb::device::cpu> arg) {
            new (t) LossStateCPU((const float*) arg.data());
        }))
        .def("loss", &LossStateCPU::loss);

#ifdef HAS_CUDA
    nb::class_<LossStateGPU>(m, "LossStateGPU")
        .def("__init__", ([](LossStateGPU* t, nb::ndarray<float, nb::shape<resolution, resolution, 3>, nb::c_contig, nb::device::cpu> arg) {
            new (t) LossStateGPU((const float*) arg.data());
        }))
        .def("loss", &LossStateGPU::loss);
#endif

}
