#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cfloat>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/types.h>

using namespace nvcuda;

#define WARP_SIZE 32

#define HOST_DEVICE_INLINE __device__ __host__ inline
HOST_DEVICE_INLINE
int div_ceil(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
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


namespace playground
{

template <typename DType, const int WMMA_M = 16, const int WMMA_N = 16,
          const int WMMA_K = 16, const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2, const int WARP_TILE_M = 2,
          const int WARP_TILE_N = 4, const int OFFSET = 0>
__global__ void hgemm_wmma_mma4x2_warp2x4_double_buffer_ptx(
    const DType* __restrict__ A, const DType* __restrict__ B,
    DType* __restrict__ C, const size_t M, const size_t N, const size_t K)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
    constexpr int BK = WMMA_K;

    __shared__ DType sa[2][BM][BK + OFFSET], sb[2][BK][BN + OFFSET];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    // const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id >> 1;
    const int warp_n = warp_id & 1;

    // sa, sb 索引
    const int load_a_smem_m = tid >> 1;
    const int load_a_smem_k = (tid & 1) << 3;
    const int load_b_smem_k = tid >> 4;
    const int load_b_smem_n = (tid & 15) << 3;

    const int load_a_gmem_m = by * BM + load_a_smem_m;
    const int load_b_gmem_n = bx * BN + load_b_smem_n;
    if (load_a_gmem_m >= static_cast<int>(M) || load_b_gmem_n >= static_cast<int>(N)) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DType>
        c_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (size_t i = 0; i < static_cast<size_t>(WARP_TILE_M); i++) {
#pragma unroll
        for (size_t j = 0; j < static_cast<size_t>(WARP_TILE_N); j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0);
        }
    }

    // k = 0 is loading here, buffer 0
    {
        const int load_a_gmem_k = 0 * WMMA_K + load_a_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_k = 0 * WMMA_K + load_b_smem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        // __cvta_generic_to_shared()
        // 是CUDA 9.0+引入的专门用于异步拷贝的地址转换函数：
        // 功能：将常规共享内存指针转换为异步拷贝指令能识别的特殊格式
        // 因为异步拷贝指令（如cp.async）需要特定格式的地址
        uint32_t load_a_smem_ptr =
            __cvta_generic_to_shared(&sa[0][load_a_smem_m][load_a_smem_k]);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16);

        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&sb[0][load_b_smem_k][load_b_smem_n]);
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();
    

#pragma unroll
    for (int k = 1; k < NUM_K_TILES; k++) {
        const int smem_sel = (k - 1) & 1;
        const int smem_sel_next = k & 1;

        const int load_a_gmem_k = k * WMMA_M + load_a_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_k = k * WMMA_K + load_b_smem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_a_smem_ptr = __cvta_generic_to_shared(
            &sa[smem_sel_next][load_a_smem_m][load_a_smem_k]);
        CP_ASYNC_CG(load_a_smem_ptr, &A[load_a_gmem_addr], 16);

        uint32_t load_b_smem_ptr = __cvta_generic_to_shared(
            &sb[smem_sel_next][load_b_smem_k][load_b_smem_n]);
        CP_ASYNC_CG(load_b_smem_ptr, &B[load_b_gmem_addr], 16);

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
            wmma::load_matrix_sync(a_frag[i], &sa[smem_sel][warp_a_smem_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            wmma::load_matrix_sync(b_frag[j], &sb[smem_sel][0][warp_b_smem_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    // processing last k tile
    {
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
            wmma::load_matrix_sync(a_frag[i], &sa[1][warp_a_smem_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * WMMA_N * WARP_TILE_N + j * WMMA_N;
            wmma::load_matrix_sync(b_frag[j], &sb[1][0][warp_b_smem_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
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
            wmma::store_matrix_sync(C + store_c_gmem_m * N + store_c_gmem_n, c_frag[i][j], N, wmma::mem_row_major);
        }
    }
}

PLAYGROUND_MATMUL_SIG(float16_t, 10, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4, WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2, WARP_TILE_N = 4;
    constexpr int OFFSET = 8;
    // constexpr int OFFSET = (64 - (BK % 64)) % 64;
    dim3 blockDim(32, 8);
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    hgemm_wmma_mma4x2_warp2x4_double_buffer_ptx<
        float16_t, WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
        WARP_TILE_M, WARP_TILE_N, OFFSET>
        <<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}