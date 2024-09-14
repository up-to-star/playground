#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/static.hpp"
#include "playground/system.hpp"
#include "playground/utils.hpp"

namespace playground
{
PG_MATMUL_SIG(float16_t, CUBLAS_VER, M, N, K, A, B, C)
{
    const float16_t Alpha = 1.0f;
    const float16_t Beta = 0.0f;
    cublasGemmEx(s_getCublasHandle<float16_t>(), CUBLAS_OP_N, CUBLAS_OP_N, N, M,
                 K, &Alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &Beta, C,
                 CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
}  // namespace playground
