#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>


using namespace nvcuda;

#define div_ceil(n, m) (((n) + (m) - 1) / (m))


namespace playground
{

template <typename DType>
__global__ void hgemm_wmma_mma4x2_warp2x4(const DType* __restrict__ A,
                                          const DType* __restrict__ B,
                                          DType* __restrict__ C, const size_t M,
                                          const size_t N, const size_t K)
{
    
}

PLAYGROUND_MATMUL_SIG(float16_t, 7, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128;

    dim3 blockDim(32, 8);
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    hgemm_wmma_mma4x2_warp2x4<float16_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}