#include "playground/system.hpp"
#include "playground/matmul.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define HOST_DEVICE_INLINE __device__ __host__ inline
#define WARP_SIZE  32
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

__device__ __forceinline__ int2 swizzle_block_zorder(int bx, int by,
                                                     int gridDimX)
{
    unsigned int morton = 0;
    for (int i = 0; i < (sizeof(unsigned int) * 8 / 2); i++) {
        morton |= (bx & (1 << i)) << i | (by & (1 << i)) << (i + 1);
    }
    int swizzled_bx = morton % gridDimX;
    int swizzled_by = morton / gridDimX;
    return make_int2(swizzled_bx, swizzled_by);
}

__device__ __forceinline__ int2 swizzle_block_tiled(int bx, int by,
                                                    int gridDimX, int gridDimY,
                                                    int tile_size = 4)
{
    int tile_x = bx / tile_size;
    int tile_y = by / tile_size;
    int inner_x = bx % tile_size;
    int inner_y = by % tile_size;
    int new_bx =
        tile_y * tile_size + inner_x;  // 交换 tile_x 和 tile_y 以改变访问顺序
    int new_by = tile_x * tile_size + inner_y;
    return make_int2(new_bx % gridDimX, new_by % gridDimY);
}

template <typename DType, const int WMMA_M = 16, const int WMMA_N = 16,
          const int WMMA_K = 16, const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2, const int WARP_TILE_M = 2,
          const int WARP_TILE_N = 4, const int A_PAD = 0, const int B_PAD = 0,
          const int K_STAGES = 2, const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256) hgemm_wmma16x16x16_mma4x2_warp2x4(const DType* __restrict__ A,
                                                  const DType* __restrict__ B,
                                                  DType* __restrict__ C,
                                                  const int M, const int N,
                                                  const int K)
{

    // int2 swizzled_block =
    //     swizzle_block_tiled(blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
    // const int bx = swizzled_block.x;
    // const int by = swizzled_block.y;

    // int2 swizzled_block =
    //     swizzle_block_zorder(blockIdx.x, blockIdx.y, gridDim.x);
    // const int bx = swizzled_block.x;
    // const int by = swizzled_block.y;
    const int bx = blockIdx.x + ((int) BLOCK_SWIZZLE) * blockIdx.z *
    gridDim.x, by = blockIdx.y;
    // const int bx = blockIdx.x, by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
    constexpr int BK = WMMA_K;

    __shared__ DType sa[K_STAGES][BM][BK + A_PAD], sb[K_STAGES][BK][BN + B_PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    const int load_a_smem_m = tid >> 1;
    const int load_a_smem_k = (tid & 1) << 3;
    const int load_b_smem_k = tid >> 4;
    const int load_b_smem_n = (tid & 15) << 3;

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

    constexpr int s_a_stage_offset = BM * (A_PAD + BK);
    constexpr int s_b_stage_offset = BK * (B_PAD + BN);
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(sa);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(sb);

    // 预加载前 k - 1 个数据
    {
#pragma unroll
        for (int k = 0; k < (K_STAGES - 1); k++) {
            const int load_a_gmem_k = k * WMMA_K + load_a_smem_k;
            const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
            const int load_b_gmem_k = k * WMMA_K + load_b_smem_k;
            const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

            uint32_t load_a_smem_ptr =
                smem_a_base_ptr +
                 (k * s_a_stage_offset + load_a_smem_m * (BK + A_PAD) +
                  load_a_smem_k) *
                     sizeof(DType);
            CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16);

            uint32_t load_b_smem_ptr =
                smem_b_base_ptr +
                (k * s_b_stage_offset + load_b_smem_k * (BN + B_PAD) +
                 load_b_smem_n) *
                    sizeof(DType);
            CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
            CP_ASYNC_COMMIT_GROUP();
        }
        CP_ASYNC_WAIT_GROUP(K_STAGES - 2);
        __syncthreads();
    }

#pragma unroll
    for (int k = K_STAGES - 1; k < NUM_K_TILES; k++) {
        const int smem_sel = (k + 1) % K_STAGES;
        const int smem_sel_next = k % K_STAGES;

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
                warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M;
            wmma::load_matrix_sync(a_frag[i], &sa[smem_sel][warp_a_smem_m][0], BK + A_PAD);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N;
            wmma::load_matrix_sync(b_frag[j], &sb[smem_sel][0][warp_b_smem_n],
                                   BN + B_PAD);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
        CP_ASYNC_WAIT_GROUP(K_STAGES - 2);
        __syncthreads();
    }

    if (K_STAGES - 2 > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 处理最后 (K_STAGE-1) k iters.
    {
#pragma unroll
        for (int k = 0; k < K_STAGES - 1; k++) {
            const int stage_sel = ((NUM_K_TILES - (K_STAGES - 1) + k) % K_STAGES);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DType,
                           wmma::row_major>
                a_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DType,
                           wmma::row_major>
                b_frag[WARP_TILE_N];

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
                const int warp_a_smem_m =
                    warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M;
                wmma::load_matrix_sync(
                    a_frag[i], &sa[stage_sel][warp_a_smem_m][0], BK + A_PAD);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                const int warp_b_smem_n =
                    warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N;
                wmma::load_matrix_sync(
                    b_frag[j], &sb[stage_sel][0][warp_b_smem_n], BN + B_PAD);
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
                by * BM + warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M;
            const int store_c_gmem_n =
                bx * BN + warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N;
            wmma::store_matrix_sync(C + store_c_gmem_m * N + store_c_gmem_n,
                                    c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}



PLAYGROUND_MATMUL_DEC(float16_t, 11, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128;
    const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    const int WMMA_TILE_M = 4, WMMA_TILE_N = 2;
    const int WARP_TILE_M = 2, WARP_TILE_N = 4;
    const int A_PAD = 8, B_PAD = 8;
    dim3 blockDim(256);
    const int BX = div_ceil(N, BN), BY = div_ceil(M, BM);
    const int NSPLIT = 2048;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);
    // dim3 gridDim(BX, BY);
    hgemm_wmma16x16x16_mma4x2_warp2x4<float16_t, WMMA_M, WMMA_N, WMMA_K,
                                      WMMA_TILE_M, WMMA_TILE_N, WARP_TILE_M,
                                      WARP_TILE_N, A_PAD, B_PAD, 3, true>
        <<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

}