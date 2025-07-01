#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define div_ceil(n, m) (((n) + (m) - 1) / (m))

namespace playground
{

template <typename DType, const int WMMA_M = 16, const int WMMA_N = 16,
          const int WMMA_K = 16>
__global__ void hgemm_wmma_naive(const DType* __restrict__ A,
                                 const DType* __restrict__ B,
                                 DType* __restrict__ C, const size_t M,
                                 const size_t N, const size_t K)
{
    const int load_gmem_a_m = blockIdx.y * WMMA_M;
    const int load_gmem_b_n = blockIdx.x * WMMA_N;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    if (load_gmem_a_m >= static_cast<int>(M) ||
        load_gmem_b_n >= static_cast<int>(N)) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, DType,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, DType,
                   wmma::row_major>
        b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, DType> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; k++) {
        wmma::load_matrix_sync(a_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
        wmma::load_matrix_sync(b_frag, B + (k * WMMA_K) * N + load_gmem_b_n,
                               N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // __syncthreads();
    }
    wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, c_frag, N,
                            wmma::mem_row_major);
}

PLAYGROUND_MATMUL_DEC(float16_t, 5, M, N, K, A, B, C)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    dim3 blockDim(8, 4);
    dim3 gridDim(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));
    hgemm_wmma_naive<float16_t, WMMA_M, WMMA_N, WMMA_K>
        <<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

}  // namespace playground
