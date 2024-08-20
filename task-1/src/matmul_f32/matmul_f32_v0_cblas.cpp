#include "playground/matmul.hpp"
#include <cblas.h>

namespace playground
{
PG_MATMUL_SIG(float32_t, CBLAS_VERSION, M, N, K, A, B, C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
}  // namespace playground