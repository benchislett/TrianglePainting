#pragma once

#include "common.h"
#include "numerical/tensor_containers.h"

#include <cassert>

namespace numerical {

    template<typename T = float, bool RowMajor = true>
    struct ReferenceMatrixVectorProduct {
        static void compute(const PODMatrix<T, RowMajor> &A, const PODVector<T> &x, PODVector<T> &y) {
            assert(A.cols() == x.size());
            assert(A.rows() == y.size());

            for (int i = 0; i < A.rows(); i++) {
                y[i] = T(0);
                for (int j = 0; j < A.cols(); j++) {
                    y[i] += A(i, j) * x[j];
                }
            }
        }
    };

    template<typename T = float, bool RowMajor = true, bool ReduceRows = true>
    struct ReferenceMatrixReduction {
        static void compute(const PODMatrix<T, RowMajor> &A, PODVector<T> &y) {
            int reduceDim = ReduceRows ? A.rows() : A.cols();
            int reduceSize = ReduceRows ? A.cols() : A.rows();
            
            for (int i = 0; i < reduceDim; i++) {
                y[i] = T(0);
                for (int j = 0; j < reduceSize; j++) {
                    if constexpr (RowMajor == ReduceRows)
                        y[i] += A[i * reduceSize + j];
                    else
                        y[i] += A[j * reduceDim + i];
                }
            }
        }
    };
};
