#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/static.hpp"
#include "playground/utils.hpp"

namespace playground
{
PG_MATMUL_SIG(float32_t, CUBLAS_VER, M, N, K, A, B, C)
{
    const float32_t Alpha = 1.0f;
    const float32_t Beta = 0.0f;
    cublasSgemm(s_getCublasHandle<float32_t>(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &Alpha, B, N, A, K,
                &Beta, C, N);
}
}  // namespace playground