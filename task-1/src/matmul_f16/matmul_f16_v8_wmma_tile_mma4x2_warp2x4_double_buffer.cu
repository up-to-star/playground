#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cfloat>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define div_ceil(n, m) (((n) + (m) - 1) / (m))
#define WARP_SIZE 32
#define READ128BITS(pointer)                                                   \
    (*reinterpret_cast<const float4*>(std::addressof(pointer)))
#define WRITE128BITS(pointer)                                                  \
    (*reinterpret_cast<float4*>(std::addressof(pointer)))


namespace playground
{

template <typename DType, const int WMMA_M = 16, const int WMMA_N = 16,
          const int WMMA_K = 16, const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2, const int WARP_TILE_M = 2, const int WARP_TILE_N = 4>
__global__ void hgemm_wmma_mma4x2_warp2x4_double_buffer(
    const DType* __restrict__ A, const DType* __restrict__ B,
    DType* __restrict__ C, const size_t M, const size_t N, const size_t K)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M;
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N;
    constexpr int BK = WMMA_K;

    __shared__ DType sa[2][BM][BK], sb[2][BK][BN];

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

    // 全局 m, n
    const int load_a_gmem_m = by * BM + load_a_smem_m;
    const int load_b_geme_n = bx * BN + load_b_smem_n;
    if (load_a_gmem_m >= static_cast<int>(M) || load_b_geme_n >= static_cast<int>(N)) {
        return;
    }
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DType>
        c_frag[WARP_TILE_M][WARP_TILE_N];
#pragma unroll
    for (size_t i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (size_t j = 0; j < WARP_TILE_N; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0);
        }
    }
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DType,
                   wmma::row_major>
        a_frag[WARP_TILE_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DType,
                   wmma::row_major>
        b_frag[WARP_TILE_N];

    // 预加载第一个tile
    {
        const int load_a_gmem_k = 0 * WMMA_K + load_a_smem_k;
        const int load_b_gmem_k = 0 * WMMA_K + load_b_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_geme_n;
        WRITE128BITS(sa[0][load_a_smem_m][load_a_smem_k]) =
            READ128BITS(A[load_a_gmem_addr]);
        WRITE128BITS(sb[0][load_b_smem_k][load_b_smem_n]) = READ128BITS(B[load_b_gmem_addr]);
    }
    __syncthreads();

#pragma unroll
    for (size_t k = 1; k < static_cast<size_t>(NUM_K_TILES); k++) {
        int smem_sel = (k - 1) & 1;
        int smem_sel_next = k & 1;

        const int load_a_gmem_k = k * WMMA_K + load_a_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_k = k * WMMA_K + load_b_smem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_geme_n;

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_M; i++) {
            const int warp_a_smem_m =
                warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M;
            wmma::load_matrix_sync(a_frag[i], &sa[smem_sel][warp_a_smem_m][0], BK);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N;
            wmma::load_matrix_sync(b_frag[j], &sb[smem_sel][0][warp_b_smem_n], BN);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (size_t j = 0; j < WARP_TILE_N; j++) {
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }

        // 加载下一个buffer
        WRITE128BITS(sa[smem_sel_next][load_a_smem_m][load_a_smem_k]) =
            READ128BITS(A[load_a_gmem_addr]);
        WRITE128BITS(sb[smem_sel_next][load_b_smem_k][load_b_smem_n]) =
            READ128BITS(B[load_b_gmem_addr]);

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (size_t j = 0; j < WARP_TILE_N; j++) {
            const int store_c_gmem_m =
                by * BM + warp_m * WARP_TILE_M * WMMA_M + i * WMMA_M;
            const int store_c_gmem_n =
                bx * BN + warp_n * WARP_TILE_N * WMMA_N + j * WMMA_N;
            wmma::store_matrix_sync(C + store_c_gmem_m * N + store_c_gmem_n, c_frag[i][j], N, wmma::mem_row_major);
        }
    }

}

PLAYGROUND_MATMUL_SIG(float16_t, 8, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128;
    dim3 blockDim(32, 8);
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    hgemm_wmma_mma4x2_warp2x4_double_buffer<float16_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}