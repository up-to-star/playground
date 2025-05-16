#include "playground/system.hpp"
#include "playground/matmul.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>


namespace playground
{
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#include <cuda_fp16.h>

// 定义一个结构体来保存4个half值
struct float4h
{
    __half2 val1;
    __half2 val2;

    // 构造函数
    __device__ float4h()
    {
    }

    // 加载函数
    __device__ static auto load(const __half* ptr) -> float4h
    {
        float4h result;
        result.val1 = *reinterpret_cast<const __half2*>(ptr);
        result.val2 = *reinterpret_cast<const __half2*>(ptr + 2);
        return result;
    }

    // 存储函数
    __device__ void store(__half* ptr) const
    {
        *reinterpret_cast<__half2*>(ptr) = val1;
        *reinterpret_cast<__half2*>(ptr + 2) = val2;
    }
};

template <typename DType>
__global__ void sgemm_V1(const DType* __restrict__ a, const DType* __restrict__ b,
                         DType* __restrict__ c, const int M, const int N,
                         const int K)
{

//     const int BM = 128;
//     const int BN = 128;
//     const int BK = 8;
//     const int TM = 8;
//     const int TN = 8;

//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int tid = ty * blockDim.x + tx;

//     __shared__ DType s_a[BM][BK];
//     __shared__ DType s_b[BK][BN];

//     float r_c[TM][TN] = {0.0};

//     int load_a_smem_m = tid >> 1;         // tid/2, row of s_a
//     int load_a_smem_k = (tid & 1) << 2;   // (tid % 2 == 0) ? 0 : 4, col of s_a
//     int load_b_smem_k = tid >> 5;         // tid/32, row of s_b
//     int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

//     int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
//     int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

//     for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
//         int load_a_gmem_k = bk * BK + load_a_smem_k;  // global col of a
//         int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
//         FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
//         int load_b_gmem_k = bk * BK + load_b_smem_k;  // global row of b
//         int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
//         FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

//         __syncthreads();

// #pragma unroll
//         for (int k = 0; k < BK; k++) {
// #pragma unroll
//             for (int m = 0; m < TM; m++) {
// #pragma unroll
//                 for (int n = 0; n < TN; n++) {
//                     int comp_a_smem_m = ty * TM + m;
//                     int comp_b_smem_n = tx * TN + n;
//                     r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
//                 }
//             }
//         }

//         __syncthreads();
//     }

// #pragma unroll
//     for (int i = 0; i < TM; i++) {
//         int store_c_gmem_m = by * BM + ty * TM + i;
// #pragma unroll
//         for (int j = 0; j < TN; j += 4) {
//             int store_c_gmem_n = bx * BN + tx * TN + j;
//             int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
//             FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
//         }
//     }
}

template <>
__global__ void sgemm_V1<float16_t>(const __half* __restrict__ a,
                                 const __half* __restrict__ b,
                                 __half* __restrict__ c, const int M,
                                 const int N, const int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ __half s_a[BM][BK];
    __shared__ __half s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;         // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;   // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;         // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        if ((load_a_gmem_m < M) && (load_a_gmem_k < K)) {
            float4h temp_a = float4h::load(&a[load_a_gmem_addr]);
            s_a[load_a_smem_m][load_a_smem_k] = temp_a.val1.x;
            s_a[load_a_smem_m][load_a_smem_k + 1] = temp_a.val1.y;
            s_a[load_a_smem_m][load_a_smem_k + 2] = temp_a.val2.x;
            s_a[load_a_smem_m][load_a_smem_k + 3] = temp_a.val2.y;
        }

        int load_b_gmem_k = bk * BK + load_b_smem_k;  // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);

        if ((load_b_gmem_n < N) && (load_b_gmem_k < K)) {
            float4h temp_b = float4h::load(&b[load_b_gmem_addr]);
            s_b[load_b_smem_k][load_b_smem_n] = temp_b.val1.x;
            s_b[load_b_smem_k + 1][load_b_smem_n] = temp_b.val1.y;
            s_b[load_b_smem_k + 2][load_b_smem_n] = temp_b.val2.x;
            s_b[load_b_smem_k + 3][load_b_smem_n] = temp_b.val2.y;
        }

        __syncthreads();

// 计算部分保持不变
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += __half2float(s_a[comp_a_smem_m][k]) *
                                 __half2float(s_b[k][comp_b_smem_n]);
                }
            }
        }

        __syncthreads();
    }

// 结果写回全局内存
#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN;
             j += 4) {  // 注意这里是+=4，因为我们一次处理四个half
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            float4h result;
            result.val1 = make_half2(__float2half(r_c[i][j]),
                                     __float2half(r_c[i][j + 1]));
            result.val2 = make_half2(__float2half(r_c[i][j + 2]),
                                     __float2half(r_c[i][j + 3]));
            result.store(&c[store_c_gmem_addr]);
        }
    }
}

    PLAYGROUND_MATMUL_SIG(float16_t, 5, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM -1) / BM);
    sgemm_V1<float16_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

}