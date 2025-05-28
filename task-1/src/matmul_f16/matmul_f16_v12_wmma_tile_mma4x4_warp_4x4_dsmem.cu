#include "playground/system.hpp"
#include "playground/matmul.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cuda_runtime_api.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes)                                           \
    asm volatile(                                                              \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),     \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                              \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),     \
        "l"(src), "n"(bytes))

HOST_DEVICE_INLINE
int div_ceil(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

namespace playground
{

template <typename DType, const int WMMA_M = 16, const int WMMA_N = 16,
          const int WMMA_K = 16, const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 4, const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4, const int A_PAD = 0, const int B_PAD = 0,
          const int K_STAGE = 2, const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(512)
    hgemm_wmma_mma4x4_warp4x4_dsmem_stage(const DType* __restrict__ A,
                                          const DType* __restrict__ B,
                                          DType* __restrict__ C, const int M,
                                          const int N, const int K)
{
    const int bx = blockIdx.x + ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x,
              by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 256
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 256
    constexpr int BK = WMMA_K;
    if (bx >= N / BN || by >= M / BM) {
        return;
    }

    // 动态共享内存
    extern __shared__ DType smem[];

    DType* sa = smem;
    DType* sb = smem + K_STAGE * BM * (BK + A_PAD);
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id >> 2;
    const int warp_n = warp_id & 3;

    // shared memeor 中的索引
    const int load_a_smem_m = tid >> 1;
    const int load_a_smem_k = (tid & 1) << 3;
    const int load_b_smem_k = tid >> 5;
    const int load_b_smem_n = (tid & 31) << 3;
    const int load_a_gmem_m = by * BM + load_a_smem_m;
    const int load_b_gmem_n = bx * BN + load_b_smem_n;

    if (load_a_gmem_m >= M || load_b_gmem_n >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DType>
        c_frag[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0);
        }
    }

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(sa);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(sb);

    // 预加载前几个数据
#pragma unroll
    for (int k = 0; k < K_STAGE - 1; k++) {
        const int load_a_gmem_k = k * WMMA_K + load_a_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_k = k * WMMA_K + load_b_smem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_a_smem_ptr =
            smem_a_base_ptr + (k * s_a_stage_offset +
                               load_a_smem_m * (BK + A_PAD) + load_a_smem_k) *
                                  sizeof(DType);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16);

        uint32_t load_b_smem_ptr =
            smem_b_base_ptr + (k * s_b_stage_offset +
                               load_b_smem_k * (BN + B_PAD) + load_b_smem_n) *
                                  sizeof(DType);
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

#pragma unroll 32
    for (int k = K_STAGE - 1; k < NUM_K_TILES; k++) {
        const int smem_sel = (k + 1) % K_STAGE;
        const int smem_sel_next = k % K_STAGE;

        const int load_a_gmem_k = k * WMMA_K + load_a_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_k = k * WMMA_K + load_b_smem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_a_smem_ptr =
            smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                               load_a_smem_m * (BK + A_PAD) + load_a_smem_k) *
                                  sizeof(DType);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16);
        uint32_t load_b_smem_ptr =
            smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                               load_b_smem_k * (BN + B_PAD) + load_b_smem_n) *
                                  sizeof(DType);
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DType,
                       wmma::row_major>
            a_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DType,
                       wmma::row_major>
            b_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
            const int warp_a_smem_m =
                warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
            DType* load_a_smem_frag_ptr = sa + smem_sel * s_a_stage_offset +
                                          warp_a_smem_m * (BK + A_PAD) + 0;
            wmma::load_matrix_sync(a_frag[i], load_a_smem_frag_ptr, BK + A_PAD);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            DType* load_b_smem_frag_ptr = sb + smem_sel * s_b_stage_offset +
                                          0 * (BN + B_PAD) + warp_b_smem_n;
            wmma::load_matrix_sync(b_frag[j], load_b_smem_frag_ptr, BN + B_PAD);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    if (K_STAGE > 1) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 处理最后几个
    {
#pragma unroll
        for (int k = 0; k < K_STAGE - 1; k++) {
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DType,
                           wmma::row_major>
                a_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DType,
                           wmma::row_major>
                b_frag[WARP_TILE_N];

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
                const int warp_a_smem_m =
                    warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
                DType* load_a_smem_frag_ptr = sa +
                                              stage_sel * s_a_stage_offset +
                                              warp_a_smem_m * (BK + A_PAD) + 0;
                wmma::load_matrix_sync(a_frag[i], load_a_smem_frag_ptr, BK + A_PAD);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                const int warp_b_smem_n =
                    warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
                DType* load_b_smem_frag_ptr = sb +
                                              stage_sel * s_b_stage_offset +
                                              0 * (BN + B_PAD) + warp_b_smem_n;
                wmma::load_matrix_sync(b_frag[j], load_b_smem_frag_ptr, BN + B_PAD);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int store_c_gmem_m =
                by * BM + warp_m * WMMA_M * WARP_TILE_M + i * WMMA_M;
            const int store_c_gmem_n =
                bx * BN + warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            wmma::store_matrix_sync(C + store_c_gmem_m * N + store_c_gmem_n, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}


PLAYGROUND_MATMUL_SIG(float16_t, 12, M, N, K, A, B, C)
{
    constexpr int BM = 256, BN = 256;
    constexpr int BK = 16;
    constexpr int A_PAD = 8, B_PAD = 8;
    dim3 blockDim(32, 16);
    // dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    constexpr int K_STAGE = 3;
    size_t sharedMemSize =
        K_STAGE * (BM * (BK + A_PAD) + BK * (BN + B_PAD)) * sizeof(float16_t);
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4, WMMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4, WARP_TILE_N = 4;
    const int BX = div_ceil(N, BN), BY = div_ceil(M, BM);
    const int NSPLIT = 4096;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    // const int split_num = 1;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);
    cudaFuncSetAttribute(
        hgemm_wmma_mma4x4_warp4x4_dsmem_stage<
            float16_t, WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
            WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, K_STAGE, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    hgemm_wmma_mma4x4_warp4x4_dsmem_stage<float16_t, WMMA_M, WMMA_N, WMMA_K,
                                          WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
                                          WARP_TILE_N, A_PAD, B_PAD, K_STAGE, true>
        <<<gridDim, blockDim, sharedMemSize>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}


}