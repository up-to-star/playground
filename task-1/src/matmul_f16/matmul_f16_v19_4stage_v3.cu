#include "playground/common.hpp"
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/types.h>

namespace playground
{

#define A_PAD 0
#define B_PAD 0

#define WARP_ROWS 64
#define WARP_COLS 64

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 4, const int MMA_TILE_N = 2,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 8>
__global__ void hgemm_mma_4stage_v3(const float16_t* __restrict__ A,
                                    const float16_t* __restrict__ B,
                                    float16_t* __restrict__ C, const int M,
                                    const int N, const int K)
{
    const int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // 256
    const int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // 128
    const int BK = MMA_K * 2;

    extern __shared__ float16_t smem[];
    float16_t* sa = smem;
    float16_t* sb = smem + BM * BK * 4;
    float16_t* sc = smem;

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
    uint32_t RA[2][WARP_TILE_M][4];
    uint32_t RB[2][WARP_TILE_N][2];

#pragma unroll
    for (size_t i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (size_t j = 0; j < WARP_TILE_N; j++) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    size_t bx =
        (blockIdx.y % 2 == 0) ? (blockDim.x - 1 - blockIdx.x) : blockIdx.x;
    size_t by = blockIdx.y;
    size_t tid = threadIdx.x + threadIdx.y * blockDim.x +
                 threadIdx.z * blockDim.x * blockDim.y;
    size_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    size_t s_a_off = 0;
    size_t s_b_off = 0;

    // buff 0
    // a: global -> smem, 256 * 32, 一个线程拷贝四行数据
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;

        int load_a_gmem_m = by * BM + row;
        int load_a_gmem_k = 0 * 32 + col;
        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
    }

    // buff 0
    // b: global -> smem, 32 * 128, 一个线程拷贝两行数据
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;

        int load_b_gmem_k = 0 * 32 + row;
        int load_b_gmem_n = bx * BN + col;

        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    s_a_off += BM * BK;
    s_b_off += BK * BN;

    // buff 1
    // a: global -> smem, 256 * 32, 一个线程拷贝四行数据
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;

        int load_a_gmem_m = by * BM + row;
        int load_a_gmem_k = 1 * 32 + col;

        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
    }

    // buff 1
    // b: global -> smem, 32 * 128, 一个线程拷贝两数据
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;

        int load_b_gmem_k = 1 * 32 + row;
        int load_b_gmem_n = bx * BN + col;

        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    s_a_off += BM * BK;
    s_b_off += BK * BN;

    // buff 2
    // a: global -> smem, 256 * 32, 一个线程拷贝四行数据
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ((tid >> 2) << 2) + i;
        int col = (tid & 3) << 3;

        int load_a_gmem_m = by * BM + row;
        int load_a_gmem_k = 2 * 32 + col;

        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    // buff 2
    // b: global -> smem, 32 * 128, 一个线程拷贝两列数据
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int row = ((tid >> 4) << 1) + i;
        int col = (tid & 15) << 3;

        int load_b_gmem_k = 2 * 32 + row;
        int load_b_gmem_n = bx * BN + col;

        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_ptr =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();

    CP_ASYNC_WAIT_GROUP(2);
    __syncthreads();

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

    s_a_off = 0;
    s_b_off = 0;

#pragma unroll
    for (size_t i = 0; i < WARP_TILE_M; i++) {
        int row = tz * 64 + i * 16 + tx % 16;
        int col = 0 * 16 + (tx / 16) * 8;
        col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
        uint32_t smem_base =
            __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    smem_base);
    }

#pragma unroll
    for (size_t i = 0; i < WARP_TILE_N; i++) {
        int row = 0 * 16 + tx % 16;
        int col = ty * 64 + i * 8;
        col = col ^ ((row & ((1 << 4) - 1)) << 3);
        uint32_t smem_base =
            __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
        LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                      smem_base);
    }

    // 计算
    for (int k = 3; k < K / BK; k++) {
        int smem_sel = (k + 1) % 4;
        int smem_sel_next = k % 4;
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_M; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = 1 * 16 + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_N; i++) {
            int row = 1 * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < WARP_ROWS / MMA_M; m++) {
#pragma unroll
            for (int n = 0; n < WARP_COLS / MMA_N; n++) {
                int n_ = (m & 1) ? (7 - n) : n;
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
                
            }
        }

        s_a_off = smem_sel_next * BM * BK;
        s_b_off = smem_sel_next * BK * BN;

        // a: global -> smem
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int row = ((tid >> 2) << 2) + i;
            int col = (tid & 3) << 3;

            int load_a_gmem_m = by * BM + row;
            int load_a_gmem_k = k * 32 + col;

            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_ptr =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            CP_ASYNC_CG(smem_ptr, &A[load_a_gmem_addr], 16);
        }

#pragma unroll
        for (int i = 0; i < 2; i++) {
            int row = ((tid >> 4) << 1) + i;
            int col = (tid & 15) << 3;

            int load_b_gmem_k = k * 32 + row;
            int load_b_gmem_n = bx * BN + col;

            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_ptr =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
            CP_ASYNC_CG(smem_ptr, &B[load_b_gmem_addr], 16);
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(2);

        __syncthreads();

        smem_sel = (k - 2) % 4;
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_M; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = 0 * 16 + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_N; i++) {
            int row = 0 * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < WARP_ROWS / MMA_M; m++) {
#pragma unroll
            for (int n = 0; n < WARP_COLS / MMA_N; n++) {
                int n_ = (m & 1) ? (7 - n) : n;
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
    }

    int smem_sel = (K / BK - 3) % 4;
#pragma unroll
    for (int k = 1; k >= 0; k--) {
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;
        for (size_t i = 0; i < WARP_TILE_M; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = k * 16 + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_N; i++) {
            int row = k * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < WARP_ROWS / MMA_M; m++) {
#pragma unroll
            for (int n = 0; n < WARP_COLS / MMA_N; n++) {
                int n_ = (m & 1) ? (7 - n) : n;
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
        if (k == 1) {
            smem_sel = (smem_sel + 1) % 4;
            CP_ASYNC_WAIT_GROUP(1);
            __syncthreads();
        }
    }

    smem_sel = (K / BK - 2) % 4;
#pragma unroll
    for (int k = 1; k >= 0; k--) {
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;
        for (size_t i = 0; i < WARP_TILE_M; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = k * 16 + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_N; i++) {
            int row = k * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < WARP_ROWS / MMA_M; m++) {
#pragma unroll
            for (int n = 0; n < WARP_COLS / MMA_N; n++) {
                int n_ = (m & 1) ? (7 - n) : n;
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
        if (k == 1) {
            smem_sel = (smem_sel + 1) % 4;
            CP_ASYNC_WAIT_GROUP(0);
            __syncthreads();
        }
    }

    smem_sel = (K / BK - 1) % 4;
#pragma unroll
    for (int k = 1; k >= 0; k--) {
        s_a_off = smem_sel * BM * BK;
        s_b_off = smem_sel * BK * BN;
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;
        for (size_t i = 0; i < WARP_TILE_M; i++) {
            int row = tz * 64 + i * 16 + tx % 16;
            int col = k * MMA_K + (tx / 16) * 8;
            col = col ^ (((row >> 1) & ((1 << 2) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sa + s_a_off + OFFSET(row, col, 32));
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        smem_base);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_TILE_N; i++) {
            int row = k * 16 + tx % 16;
            int col = ty * 64 + i * 8;
            col = col ^ ((row & ((1 << 4) - 1)) << 3);
            uint32_t smem_base =
                __cvta_generic_to_shared(sb + s_b_off + OFFSET(row, col, 128));
            LDMATRIX_X2_T(RB[reg_store_idx][i][0], RB[reg_store_idx][i][1],
                          smem_base);
        }

#pragma unroll
        for (int m = 0; m < WARP_ROWS / MMA_M; m++) {
#pragma unroll
            for (int n = 0; n < WARP_COLS / MMA_N; n++) {
                int n_ = (m & 1) ? (7 - n) : n;
                HMMA16816(RC[m][n_][0], RC[m][n_][1], RA[reg_load_idx][m][0],
                          RA[reg_load_idx][m][1], RA[reg_load_idx][m][2],
                          RA[reg_load_idx][m][3], RB[reg_load_idx][n_][0],
                          RB[reg_load_idx][n_][1], RC[m][n_][0], RC[m][n_][1]);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (size_t j = 0; j < WARP_TILE_N; j++) {
            int row = tz * 64 + i * 16 + tx / 4;
            int col = ty * 64 + j * 8 + (tx % 4) * 2;
            int col1 = col ^ ((row & ((1 << 4) - 1)) << 3);
            int col2 = col ^ (((row + 8) & ((1 << 4) - 1)) << 3);

            (reinterpret_cast<uint32_t*>(sc + OFFSET(row, col1, BN)))[0] =
                RC[i][j][0];
            (reinterpret_cast<uint32_t*>(sc + OFFSET(row + 8, col2, BN)))[0] =
                RC[i][j][1];
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 16; i++) {
        int row = i * 16 + tid / 16;
        int col1 = (tid % 16) * 8;
        int col2 = col1 ^ ((row & ((1 << 4) - 1)) << 3);
        INT4(C + OFFSET(by * BM + row, bx * BN + col1, N)) =
            INT4(sc + OFFSET(row, col2, BN));
        
    }
    
}


PLAYGROUND_MATMUL_DEC(float16_t, 20, M, N, K, A, B, C)
{
    const int BM = 256, BN = 128, BK = 32;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    const int MMA_TILE_M = 4, MMA_TILE_N = 2;
    const int WARP_TILE_M = 4, WARP_TILE_N = 8;
    const int K_STAGE = 4;
    cudaFuncSetAttribute(
        hgemm_mma_4stage_v3<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                            WARP_TILE_M, WARP_TILE_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
    size_t sharedMemSize = std::max(
        K_STAGE * (BM * (BK + A_PAD) + BK * (BN + B_PAD)) * sizeof(float16_t),
        128 * 256 * sizeof(float16_t));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    // const int NSPLIT = 2048;
    // int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 blockDim(32, 2, 4);
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    hgemm_mma_4stage_v3<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                        WARP_TILE_M, WARP_TILE_N>
        <<<gridDim, blockDim, sharedMemSize>>>(A, B, C, M, N, K);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

}  // namespace playground