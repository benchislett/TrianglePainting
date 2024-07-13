#include "numerical/tensor_containers.h"
#include "numerical/reference_tensor_programs.h"

#include <gtest/gtest.h>

#include <random>

#define TENSOR_TEST_EPSILON 1e-3

void fillRandom(float* data, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        data[i] = dis(gen);
    }
}

TEST(TensorPrograms, ReferenceValidation) {
    int n = 10; // number of rows (== column width)
    int m = 5; // number of columns (== row width)

    numerical::ManagedMatrix<float, true> A(n, m);
    numerical::ManagedVector<float> x(m);
    numerical::ManagedVector<float> y(n);

    fillRandom(A.data, n * m);
    fillRandom(x.data, m);

    numerical::ReferenceMatrixVectorProduct<float, true>::compute(A, x, y);

    for (int i = 0; i < n; i++) {
        // Using row-major matrix-vector product, we expect `y[i] = sum_j A[i, j] * x[j]`
        float ref = 0.f;
        for (int j = 0; j < m; j++) { // reduce along row `i`, over each of the `m` columns
            ref += A[i * m + j] * x[j];
        }

        ASSERT_NEAR(y[i], ref, TENSOR_TEST_EPSILON);
    }

    numerical::ManagedMatrix<float, false> A_colmajor(n, m); // column-major representation of A
    numerical::ManagedMatrix<float, true> A_T(m, n); // transposed representation of A

    // Transpose the matrix to convert the representation into column-major
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A_colmajor[j * n + i] = A[i * m + j];
            A_T[j * n + i] = A[i * m + j];
        }
    }

    numerical::ManagedVector<float> y_new(n);
    numerical::ReferenceMatrixVectorProduct<float, false>::compute(A_colmajor, x, y_new);

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(y[i], y_new[i], TENSOR_TEST_EPSILON); // The results should be independent of layout
    }

    numerical::ManagedVector<float> A_reduced_rows(n);
    numerical::ManagedVector<float> A_reduced_cols(m);
    numerical::ManagedVector<float> A_cm_reduced_rows(n);
    numerical::ManagedVector<float> A_cm_reduced_cols(m);
    numerical::ManagedVector<float> A_times_unit(n);
    numerical::ManagedVector<float> A_T_times_unit(m);

    numerical::ManagedVector<float> unit_n(n);
    numerical::ManagedVector<float> unit_m(m);

    for (int i = 0; i < n; i++) {
        unit_n[i] = 1.f;
    }
    for (int i = 0; i < m; i++) {
        unit_m[i] = 1.f;
    }

    numerical::ReferenceMatrixReduction<float, true, true>::compute(A, A_reduced_rows);
    numerical::ReferenceMatrixReduction<float, true, false>::compute(A, A_reduced_cols);
    numerical::ReferenceMatrixReduction<float, false, true>::compute(A_colmajor, A_cm_reduced_rows);
    numerical::ReferenceMatrixReduction<float, false, false>::compute(A_colmajor, A_cm_reduced_cols);

    numerical::ReferenceMatrixVectorProduct<float, true>::compute(A, unit_m, A_times_unit);
    numerical::ReferenceMatrixVectorProduct<float, true>::compute(A_T, unit_n, A_T_times_unit);

    for (int i = 0; i < n; i++) {
        float ref = 0.f;
        for (int j = 0; j < m; j++) {
            ref += A[i * m + j];
        }
        ASSERT_NEAR(A_reduced_rows[i], ref, TENSOR_TEST_EPSILON);
        ASSERT_NEAR(A_cm_reduced_rows[i], ref, TENSOR_TEST_EPSILON);
        ASSERT_NEAR(A_times_unit[i], ref, TENSOR_TEST_EPSILON);
    }

    for (int i = 0; i < m; i++) {
        float ref = 0.f;
        for (int j = 0; j < n; j++) {
            ref += A[j * m + i];
        }
        ASSERT_NEAR(A_reduced_cols[i], ref, TENSOR_TEST_EPSILON);
        ASSERT_NEAR(A_cm_reduced_cols[i], ref, TENSOR_TEST_EPSILON);
        ASSERT_NEAR(A_T_times_unit[i], ref, TENSOR_TEST_EPSILON);
    }
}
