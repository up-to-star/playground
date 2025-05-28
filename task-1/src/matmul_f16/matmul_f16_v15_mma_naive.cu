#include "playground/system.hpp"
#include "playground/matmul.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define HOST_DEVICE_INLINE __device__ __host__ inline
HOST_DEVICE_INLINE
int div_ceil(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
#define WARP_SIZE 32

#define READ128BITS(pointer)                                                   \
    (*reinterpret_cast<const float4*>(std::addressof(pointer)))
#define WRITE128BITS(pointer)                                                  \
    (*reinterpret_cast<float4*>(std::addressof(pointer)))
#define LDST32BITS(pointer) (*reinterpret_cast<half2*>(std::addressof(pointer)))


#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, "    \
                 "%2, %3}, [%4];\n"                                            \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                      \
                 : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"     \
        : "=r"(R0), "=r"(R1)                                                   \
        : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, "     \
                 "%1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"                \
                 : "=r"(RD0), "=r"(RD1)                                        \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), \
                   "r"(RC0), "r"(RC1))


namespace playground
{

template <typename DType, const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16>
__global__ void hgemm_mma_m16n8k16_naive(const DType* __restrict__ A,
                                         const DType* __restrict__ B,
                                         DType* __restrict__ C, const int M,
                                         const int N, const int K)
{
    const int bx = blockIdx.x, by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M;
    constexpr int BN = MMA_N;
    constexpr int BK = MMA_K;

    __shared__ DType sa[MMA_N][MMA_K];
    __shared__ DType sb[MMA_K][MMA_N];
    __shared__ DType sc[MMA_M][MMA_N];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    // sa 16x16, 每行16个half, 一个线程加载8，需要2个线程，一共需要32个线程
    const int load_a_smem_m = tid >> 1;
    const int load_a_smem_k = (tid & 1) << 3;

    // sb 16x8, 每行8个half，需要1个线程，一共需要16个线程，只需要一半线程加载
    const int load_b_smem_k = tid;
    const int load_b_smem_n = 0;
    const int load_a_gmem_m = by * BM + load_a_smem_m;
    const int load_b_gmem_n = bx * BN + load_b_smem_n;
    if (load_a_gmem_m >= M || load_b_gmem_n >= N) {
        return;
    }

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; k++) {
        const int load_a_gmem_k = k * BK + load_a_smem_k;
        const int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;

        // A -> sa
        WRITE128BITS(sa[load_a_smem_m][load_a_smem_k]) =
            READ128BITS(A[load_a_gmem_addr]);
        if (lane_id < MMA_K) {
            const int load_b_gmem_k = k * BK + load_b_smem_k;
            const int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
            WRITE128BITS(sb[load_b_smem_k][load_b_smem_n]) =
                READ128BITS(B[load_b_gmem_addr]);
        }
        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        // 加载到寄存器
        uint32_t load_a_smem_ptr =
            __cvta_generic_to_shared(&sa[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4_T(RA[0], RA[1], RA[2], RA[3], load_a_smem_ptr);
        uint32_t load_b_smem_ptr =
            __cvta_generic_to_shared(&sb[lane_id % 16][0]);
        LDMATRIX_X2_T(RB[0], RB[1], load_b_smem_ptr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0],
                  RC[1]);
        __syncthreads();
    }
    LDST32BITS(sc[lane_id / 4][(lane_id % 4) * 2]) = LDST32BITS(RC[0]);
    LDST32BITS(sc[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);
    __syncthreads();

    if (lane_id < MMA_M) {
        const int store_c_gmem_m = by * BM + lane_id;
        const int store_c_gmem_n = bx * BN;
        const int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        WRITE128BITS(C[store_c_gmem_addr]) = READ128BITS(sc[lane_id][0]);
    }

}

PLAYGROUND_MATMUL_SIG(float16_t, 15, M, N, K, A, B, C)
{
    const int BM = 16, BN = 8;
    dim3 blockDim(32);
    dim3 gridDim(div_ceil(N, BN), div_ceil(M, BM));
    hgemm_mma_m16n8k16_naive<float16_t><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}
}