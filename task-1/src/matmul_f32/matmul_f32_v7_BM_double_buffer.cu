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
__global__ void sgemmV7(const DType * __restrict__ A, const DType * __restrict__ B, DType * __restrict__ C, const int M, const int N, const int K) 
{
    const int BM = 128, BN = 128, BK = 8;
    const int TM = 8, TN = 8;
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ DType sa[2][BK][BM];
    __shared__ DType sb[2][BK][BN];

    DType r_c[TM][TN] = {0.0};
    DType r_load_a[4];
    DType r_load_b[4];
    DType r_comp_a[TM];
    DType r_comp_b[TN];



    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m; // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    {
        int load_a_gmem_k = load_a_smem_k;  // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        WRITE_FLOAT4(r_load_a[0]) = READ_FLOAT4(A[load_a_gmem_addr]);
        sa[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        sa[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        sa[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        sa[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        int load_b_gmem_k = load_b_smem_k;
        int load_b_gemem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        auto t_b = READ_FLOAT4(B[load_b_gemem_addr]);
        WRITE_FLOAT4(r_load_b[0]) = t_b;
        WRITE_FLOAT4(sb[0][load_b_smem_k][load_b_smem_n]) = t_b;
    }

    #pragma unroll
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;
        int load_a_gmem_k = bk * BK + load_a_smem_k;  // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        WRITE_FLOAT4(r_load_a[0]) = READ_FLOAT4(A[load_a_gmem_addr]);
        // sa[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        // sa[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        // sa[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        // sa[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gemem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        auto t_b = READ_FLOAT4(B[load_b_gemem_addr]);
        WRITE_FLOAT4(r_load_b[0]) = t_b;
        // WRITE_FLOAT4(sb[load_b_smem_k][load_b_smem_n]) = t_b;

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            WRITE_FLOAT4(r_comp_a[0]) = READ_FLOAT4(sa[smem_sel][k][ty * TM / 2]);
            WRITE_FLOAT4(r_comp_a[4]) = READ_FLOAT4(sa[smem_sel][k][ty * TM / 2 + BM / 2]);
            WRITE_FLOAT4(r_comp_b[0]) = READ_FLOAT4(sb[smem_sel][k][tx * TN / 2]);
            WRITE_FLOAT4(r_comp_b[4]) = READ_FLOAT4(sb[smem_sel][k][tx * TN / 2 + BN / 2]);
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    r_c[m][n] += r_comp_a[m] * r_comp_b[n];
                }
            }
        }

        sa[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        sa[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        sa[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        sa[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        WRITE_FLOAT4(sb[smem_sel_next][load_b_smem_k][load_b_smem_n]) = READ_FLOAT4(r_load_b[0]);
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(C[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(C[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

PLAYGROUND_MATMUL_DEC(float32_t, 7, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128;
    const int TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemmV7<float32_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}