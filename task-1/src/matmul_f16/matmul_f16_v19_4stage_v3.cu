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

#define A_PAD 8
#define B_PAD 8

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 8>
__global__ void hgemm_mma_4stage_v3(const float16_t* __restrict__ A,
                                    const float16_t* __restrict__ B,
                                    float16_t* __restrict__ C, const int M,
                                    const int N, const int K)
{
    const int bx = blockIdx.z * gridDim.x + blockIdx.x, by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    const int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // 128
    const int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // 256
    const int BK = 32;

    extern __shared__ float16_t smem[];
    float16_t* sa = smem;
    float16_t* sb = smem + 4 * BM * (BK + A_PAD);

    constexpr int sa_stage_offset = BM * (BK + A_PAD);
    constexpr int sb_stage_offset = BK * (BN + B_PAD);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id / 2;

    // 一个线程需要加载A的两行
    const int load_a_smem_m = (tid >> 2) << 1;
    const int load_a_smem_k = (tid & 3) << 3;
    // 一个线程需要加载B的四行
    const int load_b_smem_k = (tid >> 5) << 2;
    const int load_b_smem_n = (tid & 31) << 3;
    const int load_a_gmem_m = bx * BM + load_a_smem_m;
    const int load_b_gmem_n = by * BN + load_b_smem_n;

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

    int load_a_smem_addr_0 =
        smem_a_base_ptr +
        OFFSET(load_a_smem_m, load_a_smem_k, BK + A_PAD) * sizeof(float16_t);
    int load_a_smem_addr_1 =
        load_a_smem_addr_0 + (BK + A_PAD) * sizeof(float16_t);
    int load_b_smem_addr_0 =
        smem_b_base_ptr +
        OFFSET(load_b_smem_k, load_b_smem_n, BN + B_PAD) * sizeof(float16_t);
    int load_b_smem_addr_1 =
        load_b_smem_addr_0 + (BN + B_PAD) * sizeof(float16_t);
    int load_b_smem_addr_2 =
        load_b_smem_addr_0 + 2 * (BN + B_PAD) * sizeof(float16_t);
    int load_b_smem_addr_3 =
        load_b_smem_addr_0 + 3 * (BN + B_PAD) * sizeof(float16_t);
    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

#pragma unroll
    for (int k = 0; k < 3; k++) {
        CP_ASYNC_CA(load_a_smem_addr_0 +
                        k * sa_stage_offset * (int) sizeof(float16_t),
                    &A[load_a_gmem_addr], 16);
        CP_ASYNC_CA(load_a_smem_addr_1 +
                        k * sa_stage_offset * (int) sizeof(float16_t),
                    &A[load_a_gmem_addr + K], 16);
        CP_ASYNC_CA(load_b_smem_addr_0 +
                        k * sb_stage_offset * (int) sizeof(float16_t),
                    &B[load_b_gmem_addr], 16);
        CP_ASYNC_CA(load_b_smem_addr_1 +
                        k * sb_stage_offset * (int) sizeof(float16_t),
                    &B[load_b_gmem_addr + N], 16);
        CP_ASYNC_CA(load_b_smem_addr_2 +
                        k * sb_stage_offset * (int) sizeof(float16_t),
                    &B[load_b_gmem_addr + 2 * N], 16);
        CP_ASYNC_CA(load_b_smem_addr_3 +
                        k * sb_stage_offset * (int) sizeof(float16_t),
                    &B[load_b_gmem_addr + 3 * N], 16);
        CP_ASYNC_COMMIT_GROUP();
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
    }
    CP_ASYNC_WAIT_GROUP(2);
    __syncthreads();

#pragma unroll 32
    for (int k = 3; k < NUM_K_TILES; k++) {
        const int smem_sel = (k + 1) % 4;
        const int smem_sel_next = k % 4;
        CP_ASYNC_CA(load_a_smem_addr_0 + smem_sel_next * sa_stage_offset *
                                             (int) sizeof(float16_t),
                    &A[load_a_gmem_addr], 16);
        CP_ASYNC_CA(load_a_smem_addr_1 + smem_sel_next * sa_stage_offset *
                                             (int) sizeof(float16_t),
                    &A[load_a_gmem_addr + K], 16);
        CP_ASYNC_CA(load_b_smem_addr_0 + smem_sel_next * sb_stage_offset *
                                             (int) sizeof(float16_t),
                    &B[load_b_gmem_addr], 16);
        CP_ASYNC_CA(load_b_smem_addr_1 + smem_sel_next * sb_stage_offset *
                                             (int) sizeof(float16_t),
                    &B[load_b_gmem_addr + N], 16);
        CP_ASYNC_CA(load_b_smem_addr_2 + smem_sel_next * sb_stage_offset *
                                             (int) sizeof(float16_t),
                    &B[load_b_gmem_addr + 2 * N], 16);
        CP_ASYNC_CA(load_b_smem_addr_3 + smem_sel_next * sb_stage_offset *
                                             (int) sizeof(float16_t),
                    &B[load_b_gmem_addr + 3 * N], 16);
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
        CP_ASYNC_COMMIT_GROUP();

        uint32_t RA[2][WARP_TILE_M][4] = {0};
        uint32_t RB[2][WARP_TILE_N][2] = {0};

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        // smem -> reg
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
            const int warp_a_smem_m =
                warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
            const int lane_a_smem_k = (lane_id / 16) << 3;
            uint32_t lane_a_smem_ptr_1 =
                smem_a_base_ptr +
                (smem_sel * sa_stage_offset + lane_a_smem_m * (BK + A_PAD) +
                 lane_a_smem_k) *
                    sizeof(float16_t);
            LDMATRIX_X4(RA[0][i][0], RA[0][i][1], RA[0][i][2], RA[0][i][3],
                        lane_a_smem_ptr_1);
            uint32_t lane_a_smem_ptr_2 =
                smem_a_base_ptr +
                (smem_sel * sa_stage_offset + lane_a_smem_m * (BK + A_PAD) +
                 lane_a_smem_k + 16) *
                    sizeof(float16_t);
            LDMATRIX_X4(RA[1][i][0], RA[1][i][1], RA[1][i][2], RA[1][i][3],
                        lane_a_smem_ptr_2);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            const int lane_b_smem_k = lane_id;
            const int lane_b_smem_n = warp_b_smem_n;
            uint32_t lane_b_smem_ptr =
                smem_b_base_ptr +
                (smem_sel * sb_stage_offset + lane_b_smem_k * (BN + B_PAD) +
                 lane_b_smem_n) *
                    sizeof(float16_t);
            if (lane_id < MMA_K) {
                LDMATRIX_X2_T(RB[0][j][0], RB[0][j][1], lane_b_smem_ptr);
            } else {
                LDMATRIX_X2_T(RB[1][j][0], RB[1][j][1], lane_b_smem_ptr);
            }
            
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                size_t j_s = (i % 2) ? (WARP_TILE_N - j - 1) : j;
                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[0][i][0],
                          RA[0][i][1], RA[0][i][2], RA[0][i][3], RB[0][j_s][0],
                          RB[0][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[1][i][0],
                          RA[1][i][1], RA[1][i][2], RA[1][i][3], RB[1][j_s][0],
                          RB[1][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
        CP_ASYNC_WAIT_GROUP(2);
        __syncthreads();
    }
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    // 处理最后几个
    {
#pragma unroll
        for (int k = 0; k < 3; k++) {
            int smem_sel = ((NUM_K_TILES - (4 - 1) + k) % 4);
            uint32_t RA[2][WARP_TILE_M][4] = {0};
            uint32_t RB[2][WARP_TILE_N][2] = {0};
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
                const int warp_a_smem_m =
                    warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
                const int lane_a_smem_k = (lane_id / 16) << 3;
                uint32_t lane_a_smem_ptr_1 =
                    smem_a_base_ptr +
                    (smem_sel * sa_stage_offset +
                     lane_a_smem_m * (BK + A_PAD) + lane_a_smem_k) *
                        sizeof(float16_t);
                LDMATRIX_X4(RA[0][i][0], RA[0][i][1], RA[0][i][2], RA[0][i][3],
                            lane_a_smem_ptr_1);
                uint32_t lane_a_smem_ptr_2 =
                    smem_a_base_ptr +
                    (smem_sel * sa_stage_offset +
                     lane_a_smem_m * (BK + A_PAD) + lane_a_smem_k + 16) *
                        sizeof(float16_t);
                LDMATRIX_X4(RA[1][i][0], RA[1][i][1], RA[1][i][2], RA[1][i][3],
                            lane_a_smem_ptr_2);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                const int warp_b_smem_n =
                    warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                const int lane_b_smem_k = lane_id % 16;
                const int lane_b_smem_n = warp_b_smem_n;
                uint32_t lane_b_smem_ptr_1 =
                    smem_b_base_ptr +
                    (smem_sel * sb_stage_offset +
                     lane_b_smem_k * (BN + B_PAD) + lane_b_smem_n) *
                        sizeof(float16_t);
                LDMATRIX_X2_T(RB[0][j][0], RB[0][j][1], lane_b_smem_ptr_1);

                uint32_t lane_b_smem_ptr_2 =
                    smem_b_base_ptr +
                    (smem_sel * sb_stage_offset +
                     (lane_b_smem_k + 16) * (BN + B_PAD) + lane_b_smem_n) *
                        sizeof(float16_t);
                LDMATRIX_X2_T(RB[1][j][0], RB[1][j][1], lane_b_smem_ptr_2);
            }
            // MMA 计算
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; j++) {
                    size_t j_s = (i % 2) ? (WARP_TILE_N - j - 1) : j;
                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[0][i][0],
                              RA[0][i][1], RA[0][i][2], RA[0][i][3],
                              RB[0][j_s][0], RB[0][j_s][1], RC[i][j_s][0],
                              RC[i][j_s][1]);
                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[1][i][0],
                              RA[1][i][1], RA[1][i][2], RA[1][i][3],
                              RB[1][j_s][0], RB[1][j_s][1], RC[i][j_s][0],
                              RC[i][j_s][1]);
                }
            }
        }
    }
    // reg -> gmem, MMA_MxMMA_N=16x8
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int store_warp_smem_c_m =
                warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int store_warp_smem_c_n =
                warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            // mapping lane smem index -> global index.
            // [16][8],
            // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
            // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
            int store_lane_gmem_c_m =
                by * BM + store_warp_smem_c_m + lane_id / 4;
            int store_lane_gmem_c_n =
                bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
            int store_gmem_c_addr_0 =
                store_lane_gmem_c_m * N + store_lane_gmem_c_n;
            int store_gmem_c_addr_1 =
                (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
            LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
            LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
        }
    }
}


PLAYGROUND_MATMUL_DEC(float16_t, 20, M, N, K, A, B, C)
{
    const int BM = 128, BN = 256, BK = 32;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    const int MMA_TILE_M = 2, MMA_TILE_N = 4;
    const int WARP_TILE_M = 4, WARP_TILE_N = 8;
    const int K_STAGE = 4;
    cudaFuncSetAttribute(
        hgemm_mma_4stage_v3<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                            WARP_TILE_M, WARP_TILE_N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);
    const int BX = div_ceil(N, BN), BY = div_ceil(M, BM);
    size_t sharedMemSize =
        K_STAGE * (BM * (BK + A_PAD) + BK * (BN + B_PAD)) *
        sizeof(float16_t);
    const int NSPLIT = 2048;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 blockDim(32, 8);
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);
    hgemm_mma_4stage_v3<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                        WARP_TILE_M, WARP_TILE_N>
        <<<gridDim, blockDim, sharedMemSize>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

}  // namespace playground