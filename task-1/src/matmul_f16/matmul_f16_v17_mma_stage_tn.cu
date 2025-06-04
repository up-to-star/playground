#include "playground/common.hpp"
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <sys/types.h>

namespace playground
{

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2>
__global__ void hgemm_mma_stage_tn(const float16_t* __restrict__ A,
                                   const float16_t* __restrict__ B,
                                   float16_t* __restrict__ C, const int M,
                                   const int N, const int K)
{
    const int bx = blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
    constexpr int BK = MMA_K;
    extern __shared__ float16_t smem[];
    float16_t* sa = smem;
    float16_t* sb = smem + K_STAGE * BM * (BK + A_PAD);

    constexpr int sa_stage_offset = BM * (BK + A_PAD);
    constexpr int sb_stage_offset = BN * (BK + B_PAD);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id >> 1;

    const int load_a_smem_m = tid >> 1;
    const int load_a_smem_k = (tid & 1) << 3;
    const int load_b_smem_n = tid / 2;
    const int load_b_smem_k = (tid & 1) << 3;
    const int load_a_gmem_m = by * BM + load_a_smem_m;
    const int load_b_gmem_n = bx * BN + load_b_smem_n;
    if (load_a_gmem_m >= M || load_b_gmem_n >= N) {
        return;
    }

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(sa);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(sb);

#pragma unroll
    for (int k = 0; k < K_STAGE - 1; k++) {
        const int load_a_gmem_k = k * BK + load_a_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_k = k * BK + load_b_smem_k;
        const int load_b_gmem_addr = load_b_gmem_n * K + load_b_gmem_k;

        uint32_t load_a_smem_ptr =
            smem_a_base_ptr + (k * sa_stage_offset +
                               load_a_smem_m * (BK + A_PAD) + load_a_smem_k) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_a_smem_ptr, &A[load_a_gmem_addr], 16);

        uint32_t load_b_smem_ptr =
            smem_b_base_ptr + (k * sb_stage_offset +
                               load_b_smem_n * (BK + B_PAD) + load_b_smem_k) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

#pragma unroll 16
    for (int k = K_STAGE - 1; k < NUM_K_TILES; k++) {
        const int smem_sel = (k + 1) % K_STAGE;
        const int smem_sel_next = k % K_STAGE;
        const int load_a_gmem_k = k * BK + load_a_smem_k;
        const int load_b_gmem_k = k * BK + load_b_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_addr = load_b_gmem_n * K + load_b_gmem_k;

        uint32_t load_a_smem_ptr =
            smem_a_base_ptr + (smem_sel_next * sa_stage_offset +
                               load_a_smem_m * (BK + A_PAD) + load_a_smem_k) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_a_smem_ptr, &A[load_a_gmem_addr], 16);

        uint32_t load_b_smem_ptr =
            smem_b_base_ptr + (smem_sel_next * sb_stage_offset +
                               load_b_smem_n * (BK + B_PAD) + load_b_smem_k) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
            const int warp_a_smem_m = warp_m * MMA_M * WARP_TILE_M + i * MMA_M;
            const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
            const int lane_a_smem_k = (lane_id >> 4) << 3;
            uint32_t lane_a_smem_ptr =
                smem_a_base_ptr +
                (smem_sel * sa_stage_offset + lane_a_smem_m * (BK + A_PAD) +
                 lane_a_smem_k) *
                    sizeof(float16_t);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                        lane_a_smem_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n = warp_n * MMA_N * WARP_TILE_N + j * MMA_N;
            const int lane_b_smem_n = warp_b_smem_n + lane_id % 8;
            const int lane_b_smem_k = ((lane_id / 8) % 2) * 8;
            uint32_t lane_b_smem_ptr =
                smem_b_base_ptr +
                (smem_sel * sb_stage_offset + lane_b_smem_n * (BK + B_PAD) +
                 lane_b_smem_k) *
                    sizeof(float16_t);
            LDMATRIX_X2(RB[j][0], RB[j][1], lane_b_smem_ptr);
        }
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1],
                          RA[i][2], RA[i][3], RB[j][0], RB[j][1], RC[i][j][0],
                          RC[i][j][1]);
                
            }
        }
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    if (K_STAGE - 2 > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    {
#pragma unroll
        for (int k = 0; k < K_STAGE - 1; k++) {
            uint32_t RA[WARP_TILE_M][4];
            uint32_t RB[WARP_TILE_N][2];
            const int smem_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
                const int warp_a_smem_m =
                    warp_m * MMA_M * WARP_TILE_M + i * MMA_M;
                const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
                const int lane_a_smem_k = (lane_id >> 4) << 3;
                uint32_t lane_a_smem_ptr =
                    smem_a_base_ptr +
                    (smem_sel * sa_stage_offset +
                     lane_a_smem_m * (BK + A_PAD) + lane_a_smem_k) *
                        sizeof(float16_t);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                            lane_a_smem_ptr);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                const int warp_b_smem_n =
                    warp_n * MMA_N * WARP_TILE_N + j * MMA_N;
                const int lane_b_smem_n = warp_b_smem_n + lane_id % 8;
                const int lane_b_smem_k = ((lane_id / 8) % 2) << 3;
                uint32_t lane_b_smem_ptr =
                    smem_b_base_ptr +
                    (smem_sel * sb_stage_offset +
                     lane_b_smem_n * (BK + B_PAD) + lane_b_smem_k) *
                        sizeof(float16_t);
                LDMATRIX_X2(RB[j][0], RB[j][1], lane_b_smem_ptr);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; j++) {
                    HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1],
                              RA[i][2], RA[i][3], RB[j][0], RB[j][1],
                              RC[i][j][0], RC[i][j][1]);
                }
            }
        }
    }

    {
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8
            // half. thus, we only need 8 memory issues with 128 bits after
            // shfl_sync. may reuse RA[4][4] as RC0 ? only new RC1[4][4].
            uint32_t RC0[WARP_TILE_N][4];
            uint32_t RC1[WARP_TILE_N][4];
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store
                // 8 half. thus, we only need 8 memory issues with 128 bits
                // after shfl_sync.
                RC0[j][0] = RC[i][j][0];
                RC1[j][0] = RC[i][j][1];
                RC0[j][1] =
                    __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
                RC0[j][2] =
                    __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
                RC0[j][3] =
                    __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
                RC1[j][1] =
                    __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
                RC1[j][2] =
                    __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
                RC1[j][3] =
                    __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
            }

            if (lane_id % 4 == 0) {
                int store_warp_smem_c_m =
                    warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int store_lane_gmem_c_m =
                    by * BM + store_warp_smem_c_m + lane_id / 4;
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    int store_warp_smem_c_n =
                        warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                    int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
                    int store_gmem_c_addr_0 =
                        store_lane_gmem_c_m * N + store_lane_gmem_c_n;
                    int store_gmem_c_addr_1 =
                        (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
                    LDST128BITS(C[store_gmem_c_addr_0]) =
                        LDST128BITS(RC0[j][0]);
                    LDST128BITS(C[store_gmem_c_addr_1]) =
                        LDST128BITS(RC1[j][0]);
                }
            }
        }
    }
    
}


PLAYGROUND_MATMUL_DEC(float16_t, 18, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128, BK = 16;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    const int MMA_TILE_M = 2, MMA_TILE_N = 4;
    const int WARP_TILE_M = 4, WARP_TILE_N = 4;
    const int A_PAD = 8, B_PAD = 8;
    const int K_STAGE = 4;

    const int BX = div_ceil(N, BN), BY = div_ceil(M, BM);
    size_t sharedMemSize = K_STAGE *
                           (BM * (BK + A_PAD) + BN * (BK + B_PAD)) *
                           sizeof(float16_t);
    const int NSPLIT = 4096;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 blockDim(32, 8);
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    cudaFuncSetAttribute(
        hgemm_mma_stage_tn<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                           WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, K_STAGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);

    hgemm_mma_stage_tn<
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N,
        A_PAD, B_PAD, K_STAGE>
        <<<gridDim, blockDim, sharedMemSize>>>(A, B, C, M, N, K);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
}