#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace playground
{

template <typename DType>
__global__ void hgemm_naive_v1(const DType* __restrict__ A,
                            const DType* __restrict__ B, DType* __restrict__ C,
                            const size_t M, const size_t N, const size_t K)
{
    size_t n = blockDim.x * blockIdx.x + threadIdx.x;
    size_t m = blockDim.y * blockIdx.y + threadIdx.y;

    if (m < M && n < N) {
        float16_t temp = 0.0;
#pragma unroll
        for (size_t k = 0; k < K; k++) {
            temp += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = temp;
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 2, M, N, K, A, B, C)
{
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    hgemm_naive_v1<float16_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}