#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define div_ceil(n, m) (((n) + (m) - 1) / (m))

#define READ_HALF2(pointer)                                                    \
    (*reinterpret_cast<const half2*>(std::addressof(pointer)))
#define WRITE_HALF2(pointer)                                                    \
    (*reinterpret_cast<half2*>(std::addressof(pointer)))

namespace playground
{

template <typename DType, const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_naive_v2(const DType* __restrict__ A,
                               const DType* __restrict__ B,
                               DType* __restrict__ C, const size_t M,
                               const size_t N, const size_t K)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ float16_t sa[BM][BK], sb[BK][BN];

    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    if (load_a_gmem_m >= M || load_b_gmem_n >= N)
        return;

    float16_t r_c[TM][TN] = {{__float2half(0.0)}};

    for (size_t bk = 0; bk < div_ceil(K, BK); bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        WRITE_HALF2(sa[load_a_smem_m][load_a_smem_k + 0]) =
            READ_HALF2(A[load_a_gmem_addr + 0]);
        WRITE_HALF2(sa[load_a_smem_m][load_a_smem_k + 2]) =
            READ_HALF2(A[load_a_gmem_addr + 2]);

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        WRITE_HALF2(sb[load_b_smem_k][load_b_smem_n + 0]) =
            READ_HALF2(B[load_b_gmem_addr + 0]);
        WRITE_HALF2(sb[load_b_smem_k][load_b_smem_n + 2]) =
            READ_HALF2(B[load_b_gmem_addr + 2]);

        __syncthreads();
#pragma unroll
        for (size_t k = 0; k < BK; k++) {
#pragma unroll
            for (size_t m = 0; m < TM; m++) {
#pragma unroll
                for (size_t n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += sa[comp_a_smem_m][k] * sb[k][comp_b_smem_n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (size_t m = 0; m < TM; m++) {
        int store_c_gmem_m = by * BM + ty * TM + m;
#pragma unroll
        for (size_t n = 0; n < TN; n += 2) {
            int store_c_gmem_n = bx * BN + tx * TN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            WRITE_HALF2(C[store_c_gmem_addr]) = READ_HALF2(r_c[m][n]);
        }
    }
}

PLAYGROUND_MATMUL_DEC(float16_t, 3, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128;
    const int BK = 8;
    const int TM = 8, TN = 8;

    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    hgemm_naive_v2<float16_t, BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}