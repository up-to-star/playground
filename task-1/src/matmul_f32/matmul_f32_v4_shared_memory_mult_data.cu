#include "playground/matmul.hpp"
#include "playground/system.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace playground
{
template <typename DType, unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void sgemmV0(const DType* A, const DType* B, DType* C, size_t M,
                        size_t N, size_t K) {
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;

    const int STEP = BLOCK_SIZE * STRIDE;

    __shared__ DType sa[STEP][STEP];
    __shared__ DType sb[STEP][STEP];

    DType temp[STRIDE][STRIDE] = {0};

    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    // 计算C矩阵中当前线程块负责的起始位置
    size_t row = by * STEP;
    size_t col = bx * STEP;

    for (int s = 0; s < K; s += STEP) {
        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                size_t globalRow = row + ty * STRIDE + i;
                size_t globalCol = s + tx * STRIDE + j;
                if (globalRow < M && globalCol < K) {
                    sa[ty * STRIDE + i][tx * STRIDE + j] = A[globalRow * K + globalCol];
                } else {
                    sa[ty * STRIDE + i][tx * STRIDE + j] = 0;
                }
            }
        }

        for (int i = 0; i < STRIDE; i++) {
            for (int j = 0; j < STRIDE; j++) {
                size_t globalRow = s + ty * STRIDE + i;
                size_t globalCol = col + tx * STRIDE + j;
                if (globalRow < K && globalCol < N) {
                    sb[ty * STRIDE + i][tx * STRIDE + j] =
                        B[globalRow * N + globalCol];
                } else {
                    sb[ty * STRIDE + i][tx * STRIDE + j] = 0;
                }
            }
        }
        __syncthreads();

        // 计算部分积
        for (int k = 0; k < STEP; k++) {
            for (int i = 0; i < STRIDE; i++) {
                for (int j = 0; j < STRIDE; j++) {
                    temp[i][j] +=
                        sa[ty * STRIDE + i][k] * sb[k][tx * STRIDE + j];
                }
            }
        }

        // 同步确保所有线程完成计算，再加载下一个块
        __syncthreads();
    }

    // 将结果写入全局内存
    for (int i = 0; i < STRIDE; i++) {
        for (int j = 0; j < STRIDE; j++) {
            size_t globalRow = row + ty * STRIDE + i;
            size_t globalCol = col + tx * STRIDE + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = temp[i][j];
            }
        }
    }
}
    
    PLAYGROUND_MATMUL_SIG(float32_t, 4, M, N, K, A, B, C)
    {
        constexpr unsigned int BLOCK_SIZE = 32;
        constexpr unsigned int STRIDE = 2;
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x / STRIDE, (M + blockDim.y - 1) / blockDim.y / STRIDE);
        sgemmV0<float32_t, BLOCK_SIZE, STRIDE><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
}