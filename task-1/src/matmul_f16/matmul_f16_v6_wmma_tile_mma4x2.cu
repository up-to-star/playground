#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define div_ceil(n, m) (((n) + (m) - 1) / (m))
#define WARP_SIZE 32

#define READ32BITS(pointer)                                                    \
    (*reinterpret_cast<const half2*>(std::addressof(pointer)))
#define WRITE32BITS(pointer) (*reinterpret_cast<half2*>(std::addressof(pointer)))

#define READ64BITS(pointer)                                                    \
    (*reinterpret_cast<const float2*>(std::addressof(pointer)))
#define WRITE64BITS(pointer) (*reinterpret_cast<float2*>(std::addressof(pointer)))

namespace playground
{

template <typename DType, const int WMMA_M = 16, const int WMMA_N = 16,
          const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_mma4x2(const DType* __restrict__ A,
                         const DType* __restrict__ B, DType* __restrict__ C,
                         const size_t M, const size_t N, const size_t K)
{
    // 每个block 256 个 thread，8 个 warp
    const int bx = blockIdx.x, by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M;  // 16 * 4 = 64
    constexpr int BN = WMMA_N * WMMA_TILE_N;  // 16 * 2 = 32
    constexpr int BK = WMMA_K; // 16

    // 共享内存
    __shared__ DType sa[BM][BK], sb[BK][BN];

    // 8 个 warp 分布到 4x2的网格中，计算warp的索引
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    // const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    // 数据加载到共享内存
    const int load_a_smem_m = tid >> 2;
    const int load_a_smem_k = (tid & 3) << 2;
    const int load_b_smem_k = tid >> 4;
    const int load_b_smem_n = (tid & 15) << 1;
    const int load_a_gemem_m = by * BM + load_a_smem_m;
    const int load_b_gemem_n = bx * BN + load_b_smem_n;

    if (load_a_gemem_m >= M || load_b_gemem_n >= N) {
        return;
    }
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DType> c_frag;
    wmma::fill_fragment(c_frag, 0.0);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DType,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DType,
                   wmma::row_major>
        b_frag;

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; k++) {
        int load_a_geme_k = k * WMMA_K + load_a_smem_k;
        int load_a_gmem_addr = load_a_gemem_m * K + load_a_geme_k;
        int load_b_gmem_k = k * WMMA_K + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gemem_n;
        WRITE64BITS(sa[load_a_smem_m][load_a_smem_k]) =
            READ64BITS(A[load_a_gmem_addr]);
        WRITE32BITS(sb[load_b_smem_k][load_b_smem_n]) =
            READ32BITS(B[load_b_gmem_addr]);
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &sa[warp_m * WMMA_M][0], BK);
        wmma::load_matrix_sync(b_frag, &sb[0][warp_n * WMMA_N], BN);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // 算出来计算的块的起始位置
    const int store_a_gmem_m = by * BM + warp_m * WMMA_M;
    const int store_a_gmem_n = bx * BN + warp_n * WMMA_N;

    // 感觉这些接口是以wmma块为单位进行加载的
    wmma::store_matrix_sync(C + store_a_gmem_m * N + store_a_gmem_n, c_frag, N, wmma::mem_row_major);
        
}

PLAYGROUND_MATMUL_SIG(float16_t, 6, M, N, K, A, B, C)
{
    const int BM = 64, BN = 32;

    dim3 blockDim(32, 8);
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    hgemm_wmma_mma4x2<float16_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}