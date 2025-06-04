#include "playground/common.hpp"
#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace playground
{

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int WARP_TILE_K = 2, const int A_PAD = 0, const int B_PAD = 0,
          const int K_STAGE = 2>
__global__ void hgemm_mma_stage_v1(const float16_t* __restrict__ A,
                                   const float16_t* __restrict__ B,
                                   float16_t* __restrict__ C, const int M,
                                   const int N, const int K)
{
    const int bx = blockIdx.z * gridDim.x + blockIdx.x, by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
    const int BK = MMA_K;

    extern __shared__ float16_t smem[];
    float16_t* sa = smem;
    float16_t* sb = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;

    constexpr int sa_stage_offset = BM * (BK + A_PAD);
    constexpr int sb_stage_offset = BK * (BN + B_PAD);
    // 这里有疑问
    constexpr int sa_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
    constexpr int sb_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;  // 定位线程在 Warp 中的局部位置。
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id / 2;

    const int load_a_smem_m = tid / 2;
    const int load_a_smem_k = (tid & 1) << 3;
    const int load_b_smem_k = tid / 16;
    const int load_b_smem_n = (tid & 15) << 3;
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

    // 预加载数据
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
        const int load_a_gmem_k = k * BK * WARP_TILE_K + load_a_smem_k;
        const int load_b_gmem_k = k * BK * WARP_TILE_K + load_b_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_a_smem_ptr =
            smem_a_base_ptr + (k * sa_stage_offset +
                               load_a_smem_m * (BK + A_PAD) + load_a_smem_k) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_a_smem_ptr, &A[load_a_gmem_addr], 16);
        uint32_t load_a_smem_mma_k_ptr =
            smem_a_base_ptr + sa_mma_k_store_offset * sizeof(float16_t) +
            (k * sa_stage_offset + load_a_smem_m * (BK + A_PAD) +
             load_a_smem_k) *
                sizeof(float16_t);
        CP_ASYNC_CA(load_a_smem_mma_k_ptr, &A[load_a_gmem_addr + 16], 16);

        uint32_t load_b_smem_ptr =
            smem_b_base_ptr + (k * sb_stage_offset +
                               load_b_smem_k * (BN + B_PAD) + load_b_smem_n) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        const int load_b_gmem_k_mma_k =
            k * BK * WARP_TILE_K + MMA_K + load_b_smem_k;
        const int load_b_gmem_addr_mma_k = load_b_gmem_k_mma_k * N + load_b_gmem_n;
        uint32_t load_b_smem_mma_k_ptr =
            smem_b_base_ptr + sb_mma_k_store_offset * sizeof(float16_t) +
            (k * sb_stage_offset + load_b_smem_k * (BN + B_PAD) +
             load_b_smem_n) *
                sizeof(float16_t);
        CP_ASYNC_CA(load_b_smem_mma_k_ptr, &B[load_b_gmem_addr_mma_k], 16);
        CP_ASYNC_COMMIT_GROUP();
    }
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    uint32_t RA[2][WARP_TILE_M][4];
    uint32_t RB[2][WARP_TILE_N][2];

    int reg_store_idx = 0;
    int reg_load_idx = 1;

    {
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
            const int warp_a_smem_m = warp_m * MMA_M * WARP_TILE_M + i * MMA_M;
            const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
            const int lane_a_smem_k = (lane_id / 16) << 3;
            uint32_t lane_a_smem_ptr =
                smem_a_base_ptr +
                (0 * sa_stage_offset + lane_a_smem_m * (BK + A_PAD) +
                 lane_a_smem_k) *
                    sizeof(float16_t);
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], lane_a_smem_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n = warp_n * MMA_N * WARP_TILE_N + j * MMA_N;
            const int lane_b_smem_k = lane_id % 16;
            const int lane_b_smem_n = warp_b_smem_n;
            uint32_t lane_b_smem_ptr =
                smem_b_base_ptr +
                (0 * sb_stage_offset + lane_b_smem_k * (BN + B_PAD) +
                 lane_b_smem_n) *
                    sizeof(float16_t);
            LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                          lane_b_smem_ptr);
        }
    }
#pragma unroll 16
    for (int k = K_STAGE - 1; k < NUM_K_TILES; k++) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;
        const int smem_sel = (k + 1) % K_STAGE;
        const int smem_sel_next = k % K_STAGE;
        const int load_a_gmem_k = k * BK * WARP_TILE_K + load_a_smem_k;
        const int load_b_gmem_k = k * BK * WARP_TILE_K + load_b_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        uint32_t load_a_smem_ptr =
            smem_a_base_ptr + (smem_sel_next * sa_stage_offset +
                               load_a_smem_m * (BK + A_PAD) + load_a_smem_k) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_a_smem_ptr, &A[load_a_gmem_addr], 16);
        uint32_t load_a_smem_mma_k_ptr =
            smem_a_base_ptr + sa_mma_k_store_offset * sizeof(float16_t) +
            (smem_sel_next * sa_stage_offset + load_a_smem_m * (BK + A_PAD) +
             load_a_smem_k) *
                sizeof(float16_t);
        CP_ASYNC_CA(load_a_smem_mma_k_ptr, &A[load_a_gmem_addr + 16], 16);
        uint32_t load_b_smem_ptr =
            smem_b_base_ptr + (smem_sel_next * sb_stage_offset +
                               load_b_smem_k * (BN + B_PAD) + load_b_smem_n) *
                                  sizeof(float16_t);
        CP_ASYNC_CA(load_b_smem_ptr, &B[load_b_gmem_addr], 16);
        const int load_b_gmem_k_mma_k =
            k * BK * WARP_TILE_K + MMA_K + load_b_smem_k;
        const int load_b_gmem_addr_mma_k =
            load_b_gmem_k_mma_k * N + load_b_gmem_n;

        uint32_t load_b_smem_mma_k_ptr =
            smem_b_base_ptr + sb_mma_k_store_offset * sizeof(float16_t) +
            (smem_sel_next * sb_stage_offset + load_b_smem_k * (BN + B_PAD) +
             load_b_smem_n) *
                sizeof(float16_t);
        CP_ASYNC_CA(load_b_smem_mma_k_ptr, &B[load_b_gmem_addr_mma_k], 16);
        CP_ASYNC_COMMIT_GROUP();

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        // smem -> reg buffer1, second MMA_K, 16-31
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
            const int warp_a_smem_m =
                warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
            const int lane_a_smem_k = (lane_id / 16) << 3;
            uint32_t lane_a_smem_ptr =
                smem_a_base_ptr + sa_mma_k_store_offset * sizeof(float16_t) + 
                (smem_sel * sa_stage_offset + lane_a_smem_m * (BK + A_PAD) +
                 lane_a_smem_k) *
                    sizeof(float16_t);
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        lane_a_smem_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            const int lane_b_smem_k = lane_id % 16;
            const int lane_b_smem_n = warp_b_smem_n;
            uint32_t lane_b_smem_ptr =
                smem_b_base_ptr + sb_mma_k_store_offset * sizeof(float16_t) + 
                (smem_sel * sb_stage_offset + lane_b_smem_k * (BN + B_PAD) +
                 lane_b_smem_n) *
                    sizeof(float16_t);
            LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                          lane_b_smem_ptr);
        }
        // MMA 计算 first mma_k
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                HMMA16816(RC[i][j][0], RC[i][j][1], RA[reg_load_idx][i][0],
                          RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                          RA[reg_load_idx][i][3], RB[reg_load_idx][j][0],
                          RB[reg_load_idx][j][1], RC[i][j][0], RC[i][j][1]);
            }
        }
        reg_load_idx ^= 1;
        reg_store_idx ^= 1;

        // MMA 计算 second mma_k
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                HMMA16816(RC[i][j][0], RC[i][j][1], RA[reg_load_idx][i][0],
                          RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                          RA[reg_load_idx][i][3], RB[reg_load_idx][j][0],
                          RB[reg_load_idx][j][1], RC[i][j][0], RC[i][j][1]);
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();

        // 加载 下一个 k 到 reg buffer
        const int smem_sel_reg = (smem_sel + 1) % K_STAGE;
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; i++) {
            const int warp_a_smem_m =
                warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
            const int lane_a_smem_k = (lane_id / 16) << 3;

            uint32_t lane_a_smem_ptr =
                smem_a_base_ptr +
                (smem_sel_reg * sa_stage_offset +
                 lane_a_smem_m * (BK + A_PAD) + lane_a_smem_k) *
                    sizeof(float16_t);
            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                        RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                        lane_a_smem_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
            const int warp_b_smem_n =
                warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            const int lane_b_smem_k = lane_id % 16;
            const int lane_b_smem_n = warp_b_smem_n;
            uint32_t lane_b_smem_ptr =
                smem_b_base_ptr +
                (smem_sel_reg * sb_stage_offset +
                 lane_b_smem_k * (BN + B_PAD) + lane_b_smem_n) *
                    sizeof(float16_t);
            LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                          lane_b_smem_ptr);
        }
    }
    if (K_STAGE - 2 > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 处理最后几个
    {
#pragma unroll
        for (int k = 0; k < K_STAGE - 1; k++) {
            reg_store_idx ^= 1;
            reg_load_idx ^= 1;
            int smem_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
                const int warp_a_smem_m =
                    warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
                const int lane_a_smem_k = (lane_id / 16) << 3;
                uint32_t lane_a_smem_ptr =
                    smem_a_base_ptr +
                    sa_mma_k_store_offset * sizeof(float16_t) +
                    (smem_sel * sa_stage_offset +
                     lane_a_smem_m * (BK + A_PAD) + lane_a_smem_k) *
                        sizeof(float16_t);
                LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                            RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                            lane_a_smem_ptr);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                const int warp_b_smem_n =
                    warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                const int lane_b_smem_k = lane_id % 16;
                const int lane_b_smem_n = warp_b_smem_n;
                uint32_t lane_b_smem_ptr =
                    smem_b_base_ptr +
                    sb_mma_k_store_offset * sizeof(float16_t) +
                    (smem_sel * sb_stage_offset +
                     lane_b_smem_k * (BN + B_PAD) + lane_b_smem_n) *
                        sizeof(float16_t);
                LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                              lane_b_smem_ptr);
            }
            // MMA 计算 first
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; j++) {
                    HMMA16816(RC[i][j][0], RC[i][j][1], RA[reg_load_idx][i][0],
                              RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                              RA[reg_load_idx][i][3], RB[reg_load_idx][j][0],
                              RB[reg_load_idx][j][1], RC[i][j][0],
                              RC[i][j][1]);
                }
            }
            reg_load_idx ^= 1;
            reg_store_idx ^= 1;
            // MMA 计算 second
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; j++) {
                    HMMA16816(RC[i][j][0], RC[i][j][1], RA[reg_load_idx][i][0],
                              RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                              RA[reg_load_idx][i][3], RB[reg_load_idx][j][0],
                              RB[reg_load_idx][j][1], RC[i][j][0],
                              RC[i][j][1]);
                }
            }

            // 加载 下一个 k 到 reg buffer
            const int smem_sel_reg = (smem_sel + 1) % K_STAGE;
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; i++) {
                const int warp_a_smem_m =
                    warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                const int lane_a_smem_m = warp_a_smem_m + lane_id % 16;
                const int lane_a_smem_k = (lane_id / 16) << 3;

                uint32_t lane_a_smem_ptr =
                    smem_a_base_ptr +
                    (smem_sel_reg * sa_stage_offset +
                     lane_a_smem_m * (BK + A_PAD) + lane_a_smem_k) *
                        sizeof(float16_t);
                LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                            RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                            lane_a_smem_ptr);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; j++) {
                const int warp_b_smem_n =
                    warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                const int lane_b_smem_k = lane_id % 16;
                const int lane_b_smem_n = warp_b_smem_n;
                uint32_t lane_b_smem_ptr =
                    smem_b_base_ptr +
                    (smem_sel_reg * sb_stage_offset +
                     lane_b_smem_k * (BN + B_PAD) + lane_b_smem_n) *
                        sizeof(float16_t);
                LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                              lane_b_smem_ptr);
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


PLAYGROUND_MATMUL_DEC(float16_t, 17, M, N, K, A, B, C)
{
    const int BM = 128, BN = 128, BK = 32;
    const int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    const int MMA_TILE_M = 2, MMA_TILE_N = 4;
    const int WARP_TILE_M = 4, WARP_TILE_N = 4, WARP_TILE_K = 2;
    const int A_PAD = 8, B_PAD = 8;
    const int K_STAGE = 3;
    cudaFuncSetAttribute(
        hgemm_mma_stage_v1<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                           WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD,
                           K_STAGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 163840);
    const int BX = div_ceil(N, BN), BY = div_ceil(M, BM);
    size_t sharedMemSize = K_STAGE * WARP_TILE_K *
                           (BM * (BK + A_PAD) + BK * (BN + B_PAD)) *
                           sizeof(float16_t);
    const int NSPLIT = 4096;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 blockDim(32, 8);
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);
    hgemm_mma_stage_v1<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
                       WARP_TILE_M, WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD,
                       K_STAGE>
        <<<gridDim, blockDim, sharedMemSize>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
}  // namespace playground