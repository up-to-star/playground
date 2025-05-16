#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (*reinterpret_cast<float4*>(std::addressof(pointer)))

// 定义两个宏，一个用于读（保留 const），一个用于写（不保留 const）
#define READ_FLOAT4(pointer)                                                   \
    (*reinterpret_cast<const float4*>(std::addressof(pointer)))
#define WRITE_FLOAT4(pointer)                                                  \
    (*reinterpret_cast<float4*>(std::addressof(pointer)))

namespace playground
{

template <typename DType>
__global__ void sgemm(const DType * __restrict__ A, const DType * __restrict__ B, DType * __restrict__ C, const int M, const int N, const int K) 
{
    const int BM = 128, BN = 128, BK = 8;
    const int TM = 8, TN = 8;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ DType sa[BM][BK];
    __shared__ DType sb[BK][BN];

    DType r_c[TM][TN] = {0};

    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m; // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;  // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        auto temp = READ_FLOAT4(A[load_a_gmem_addr]);
        WRITE_FLOAT4(sa[load_a_smem_m][load_a_smem_k]) = temp;

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gemem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        auto tb = READ_FLOAT4(B[load_b_gemem_addr]);
        WRITE_FLOAT4(sb[load_b_smem_k][load_b_smem_n]) = tb;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += sa[comp_a_smem_m][k] * sb[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            auto trc = READ_FLOAT4(r_c[i][j]);
            WRITE_FLOAT4(C[store_c_gmem_addr]) = trc;
        }
    }
}

PLAYGROUND_MATMUL_SIG(float32_t, 5, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128;
    const int TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm<float32_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}