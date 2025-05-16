#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace playground
{
    template <typename DType>
    __global__ void sgemmV0(const DType* A, const DType* B, DType* C, const size_t M,
                            const size_t N, const size_t K)
    {
        size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
        if (ty < M && tx < N) {
            DType c = 0.0f;
            for (size_t i = 0; i < K; i++) {
                c += A[ty * K + i] * B[i * N + tx];
            }
            C[ty * N + tx] = c;
        }
    }
    
    PLAYGROUND_MATMUL_SIG(float32_t, 2, M, N, K, A, B, C)
    {
        dim3 blockDim(32, 32);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
        sgemmV0<float32_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
}