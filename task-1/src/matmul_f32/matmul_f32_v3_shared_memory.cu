#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

namespace playground
{
template <typename DType, unsigned int BLOCK_SIZE>
__global__ void sgemmV0(const DType* A, const DType* B, DType* C, size_t M,
                        size_t N, size_t K) {
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ DType sa[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ DType sb[BLOCK_SIZE][BLOCK_SIZE];

    DType temp = 0.0;

    for (int s = 0; s < K; s += BLOCK_SIZE) {
        if (ty < M && (s + threadIdx.x) < N) {
            sa[threadIdx.y][threadIdx.x] = A[ty * K + s + threadIdx.x];
        } else {
            sa[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (tx < N && (s + threadIdx.y) < K) {
            sb[threadIdx.y][threadIdx.x] = B[(s + threadIdx.y) * N + tx];
        } else {
            sb[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            temp += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (ty < M && tx < N) {
        C[ty * N + tx] = temp;
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 3, M, N, K, A, B, C)
{
    constexpr unsigned int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    sgemmV0<float32_t, BLOCK_SIZE><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
}