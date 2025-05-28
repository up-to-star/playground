#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cstdio>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define ASYNC_COPY_TO_SHARED(dst, src, size)                                   \
    asm volatile("cp.async.ca.shared.global [%0], [%1], " #size ";\n"          \
                 :                                                             \
                 : "r"(dst), "l"(src))

#define COMMIT_ASYNC_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define WAIT_ASYNC_GROUP() asm volatile("cp.async.wait_group 0;\n" ::)

namespace playground
{
__global__ void hgemm_v13_triple_buffered(const float16_t* __restrict__ a,
                                          const float16_t* __restrict__ b,
                                          float16_t* __restrict__ c,
                                          const int M, const int N, const int K)
{
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int by = blockIdx.y;
    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    if (bx >= N / BN || by >= M / BM) {
        return;
    }
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // const int tid = threadIdx.x + threadIdx.y * blockDim.x +
    //                 threadIdx.z * blockDim.x * blockDim.y;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ float16_t smem[];
    float16_t* s_a = smem;
    float16_t* s_b = smem + 3 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, float16_t, wmma::row_major>
        frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, float16_t, wmma::row_major>
        frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float16_t> frag_c[4][4];

#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 =
        s_a_base_addr +
        OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(float16_t);
    int load_a_smem_addr_1 =
        load_a_smem_addr_0 + (BK + APAD) * sizeof(float16_t);
    int load_b_smem_addr_0 =
        s_b_base_addr +
        OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(float16_t);
    int load_b_smem_addr_1 =
        load_b_smem_addr_0 + (BN + BPAD) * sizeof(float16_t);
    int load_b_smem_addr_2 =
        load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(float16_t);
    int load_b_smem_addr_3 =
        load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(float16_t);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    {
        // buffer 0
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_0, &a[load_a_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_1, &a[load_a_gmem_addr + K], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_0, &b[load_b_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_1, &b[load_b_gmem_addr + N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_2, &b[load_b_gmem_addr + 2 * N],
                             16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_3, &b[load_b_gmem_addr + 3 * N],
                             16);


        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        // buffer1
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_0 +
                                 s_a_db_offset * (int) sizeof(float16_t),
                             &a[load_a_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_1 +
                                 s_a_db_offset * (int) sizeof(float16_t),
                             &a[load_a_gmem_addr + K], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_0 +
                                 s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_1 +
                                 s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_2 +
                                 s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 2 * N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_3 +
                                 s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 3 * N], 16);

        COMMIT_ASYNC_GROUP();
        WAIT_ASYNC_GROUP();

        __syncthreads();
    }

#pragma unroll 32
    for (int bk = 2; bk < K / BK; bk++) {

        // int smem_sel = (bk & 1) ^ 1;
        // int smem_sel_next = ((bk - 1) & 1) ^ 1;
        int smem_sel = (bk % 3);
        int smem_sel_next = ((bk + 1) % 3);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        ASYNC_COPY_TO_SHARED(load_a_smem_addr_0 + smem_sel_next *
                                                      s_a_db_offset *
                                                      (int) sizeof(float16_t),
                             &a[load_a_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_1 + smem_sel_next *
                                                      s_a_db_offset *
                                                      (int) sizeof(float16_t),
                             &a[load_a_gmem_addr + K], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_0 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_1 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_2 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 2 * N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_3 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 3 * N], 16);

        wmma::load_matrix_sync(frag_a[0][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 0],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 16],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16],
            BK + APAD);

        wmma::load_matrix_sync(
            frag_b[0][0], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][1],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][2],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][3],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48],
            BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 16],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 32],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 48],
                               BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        }

        COMMIT_ASYNC_GROUP();
        if (bk > 2) {
            WAIT_ASYNC_GROUP();
        }
        

        __syncthreads();
    }

    // int smem_sel = ((K / BK) & 1) ^ 1;
#pragma unroll
    for (int smem_sel = (K / BK) % 3; smem_sel < 3; smem_sel++) {
        wmma::load_matrix_sync(frag_a[0][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 0],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 16],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16],
            BK + APAD);

        wmma::load_matrix_sync(
            frag_b[0][0], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][1],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][2],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][3],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48],
            BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 16],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 32],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 48],
                               BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        }
        __syncthreads();
    }

    

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16],
                                    frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

__global__ void hgemm_v13_quad_buffered(const float16_t* __restrict__ a,
                                          const float16_t* __restrict__ b,
                                          float16_t* __restrict__ c,
                                          const int M, const int N, const int K)
{
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int by = blockIdx.y;
    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    if (bx >= N / BN || by >= M / BM) {
        return;
    }
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // const int tid = threadIdx.x + threadIdx.y * blockDim.x +
    //                 threadIdx.z * blockDim.x * blockDim.y;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ float16_t smem[];
    float16_t* s_a = smem;
    float16_t* s_b = smem + 4 * BM * (BK + APAD);
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, float16_t, wmma::row_major>
        frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, float16_t, wmma::row_major>
        frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float16_t> frag_c[4][4];

#pragma unroll
    for (auto & i : frag_c) {
#pragma unroll
        for (auto & j : i) {
            wmma::fill_fragment(j, 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    int load_a_smem_addr_0 =
        s_a_base_addr +
        OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(float16_t);
    int load_a_smem_addr_1 =
        load_a_smem_addr_0 + (BK + APAD) * sizeof(float16_t);
    int load_b_smem_addr_0 =
        s_b_base_addr +
        OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(float16_t);
    int load_b_smem_addr_1 =
        load_b_smem_addr_0 + (BN + BPAD) * sizeof(float16_t);
    int load_b_smem_addr_2 =
        load_b_smem_addr_0 + 2 * (BN + BPAD) * sizeof(float16_t);
    int load_b_smem_addr_3 =
        load_b_smem_addr_0 + 3 * (BN + BPAD) * sizeof(float16_t);

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    // Preload 4 buffers (changed from 2 in original)
#pragma unroll
    for (int buf = 0; buf < 3; buf++) {
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_0 +
                                 buf * s_a_db_offset * (int) sizeof(float16_t),
                             &a[load_a_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_1 +
                                 buf * s_a_db_offset * (int) sizeof(float16_t),
                             &a[load_a_gmem_addr + K], 16);

        ASYNC_COPY_TO_SHARED(load_b_smem_addr_0 +
                                 buf * s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_1 +
                                 buf * s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_2 +
                                 buf * s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 2 * N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_3 +
                                 buf * s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 3 * N], 16);

        COMMIT_ASYNC_GROUP();
        if (buf < 3) {
            WAIT_ASYNC_GROUP();
        }

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
    }

#pragma unroll 32
    for (int bk = 3; bk < K / BK; bk++) {

        // int smem_sel = (bk & 1) ^ 1;
        // int smem_sel_next = ((bk - 1) & 1) ^ 1;
        int smem_sel = (bk & 3);
        int smem_sel_next = ((bk + 1) & 3);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        ASYNC_COPY_TO_SHARED(load_a_smem_addr_0 + smem_sel_next *
                                                      s_a_db_offset *
                                                      (int) sizeof(float16_t),
                             &a[load_a_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_1 + smem_sel_next *
                                                      s_a_db_offset *
                                                      (int) sizeof(float16_t),
                             &a[load_a_gmem_addr + K], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_0 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_1 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_2 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 2 * N], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_3 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + 3 * N], 16);

        wmma::load_matrix_sync(frag_a[0][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 0],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 16],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16],
            BK + APAD);

        wmma::load_matrix_sync(
            frag_b[0][0], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][1],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][2],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][3],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48],
            BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 16],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 32],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 48],
                               BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        }

        COMMIT_ASYNC_GROUP();
        if (bk > 4) {
            WAIT_ASYNC_GROUP();
        }

        __syncthreads();
    }

    // int smem_sel = ((K / BK) & 1) ^ 1;
#pragma unroll
    for (int smem_sel = (K / BK) & 3; smem_sel < 4; smem_sel++) {
        wmma::load_matrix_sync(frag_a[0][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 0],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[0][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 0],
            BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0],
                               &s_a[smem_sel * s_a_db_offset +
                                    (comp_c_frag_m * 64) * (BK + APAD) + 16],
                               BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][1],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 16) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][2],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 32) * (BK + APAD) + 16],
            BK + APAD);
        wmma::load_matrix_sync(
            frag_a[1][3],
            &s_a[smem_sel * s_a_db_offset +
                 (comp_c_frag_m * 64 + 48) * (BK + APAD) + 16],
            BK + APAD);

        wmma::load_matrix_sync(
            frag_b[0][0], &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][1],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 16],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][2],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 32],
            BN + BPAD);
        wmma::load_matrix_sync(
            frag_b[0][3],
            &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + 48],
            BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 16],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 32],
                               BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3],
                               &s_b[smem_sel * s_b_db_offset +
                                    16 * (BN + BPAD) + comp_c_frag_n * 64 + 48],
                               BN + BPAD);

#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        }
        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16],
                                    frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

__global__ void hgemm_v14_quad_buffered_bn128(const float16_t* __restrict__ a,
                                              const float16_t* __restrict__ b,
                                              float16_t* __restrict__ c,
                                              const int M, const int N,
                                              const int K)
{
    const int BM = 128;
    const int BN = 128;  // 修改为128
    const int BK = 32;

    int by = blockIdx.y;
    int bx = blockIdx.z * gridDim.x + blockIdx.x;
    if (bx >= N / BN || by >= M / BM) {
        return;
    }

    // 调整线程索引计算
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    extern __shared__ float16_t smem[];
    float16_t* s_a = smem;
    float16_t* s_b = smem + 4 * BM * (BK + APAD);  // 4 buffers
    int s_a_db_offset = BM * (BK + APAD);
    int s_b_db_offset = BK * (BN + BPAD);

    // 调整WMMA fragment数量 (BN减半，所以减少frag_b数量)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, float16_t, wmma::row_major>
        frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, float16_t, wmma::row_major>
        frag_b[2][2];  // 从4改为2
    wmma::fragment<wmma::accumulator, 16, 16, 16, float16_t>
        frag_c[4][2];  // 从4改为2

// 初始化fragments
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {  // 从4改为2
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    // 加载索引设置
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_k = (tid >> 4) << 2;  // 调整因为BN减小
    int load_b_smem_n = (tid & 15) << 3;  // 从31改为15

    int s_a_base_addr = __cvta_generic_to_shared(s_a);
    int s_b_base_addr = __cvta_generic_to_shared(s_b);

    // 地址计算
    int load_a_smem_addr_0 =
        s_a_base_addr +
        OFFSET(load_a_smem_m, load_a_smem_k, BK + APAD) * sizeof(float16_t);
    int load_a_smem_addr_1 =
        load_a_smem_addr_0 + (BK + APAD) * sizeof(float16_t);

    int load_b_smem_addr_0 =
        s_b_base_addr +
        OFFSET(load_b_smem_k, load_b_smem_n, BN + BPAD) * sizeof(float16_t);
    int load_b_smem_addr_1 =
        load_b_smem_addr_0 + (BN + BPAD) * sizeof(float16_t);

    // 全局内存索引
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    // 计算片段索引 (调整因为BN减小)
    int comp_c_frag_m = wid >> 1;  // 从 &1 改为 >>1
    int comp_c_frag_n = wid & 1;   // 从 >>1 改为 &1

    // 预加载4个buffer
    for (int buf = 0; buf < 3; buf++) {
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_0 +
                                 buf * s_a_db_offset * (int) sizeof(float16_t),
                             &a[load_a_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_1 +
                                 buf * s_a_db_offset * (int) sizeof(float16_t),
                             &a[load_a_gmem_addr + K], 16);

        ASYNC_COPY_TO_SHARED(load_b_smem_addr_0 +
                                 buf * s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_1 +
                                 buf * s_b_db_offset * (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + N], 16);

        COMMIT_ASYNC_GROUP();
        if (buf < 3) {
            WAIT_ASYNC_GROUP();
        }

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
    }

// 主计算循环
#pragma unroll 32
    for (int bk = 3; bk < K / BK; bk++) {
        int smem_sel = (bk & 3);
        int smem_sel_next = ((bk + 1) & 3);

        // 加载下一个tile
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        ASYNC_COPY_TO_SHARED(load_a_smem_addr_0 + smem_sel_next *
                                                      s_a_db_offset *
                                                      (int) sizeof(float16_t),
                             &a[load_a_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_a_smem_addr_1 + smem_sel_next *
                                                      s_a_db_offset *
                                                      (int) sizeof(float16_t),
                             &a[load_a_gmem_addr + K], 16);

        ASYNC_COPY_TO_SHARED(load_b_smem_addr_0 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr], 16);
        ASYNC_COPY_TO_SHARED(load_b_smem_addr_1 + smem_sel_next *
                                                      s_b_db_offset *
                                                      (int) sizeof(float16_t),
                             &b[load_b_gmem_addr + N], 16);

// 从当前buffer加载fragments
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(
                frag_a[0][i],
                &s_a[smem_sel * s_a_db_offset +
                     (comp_c_frag_m * 64 + i * 16) * (BK + APAD) + 0],
                BK + APAD);
            wmma::load_matrix_sync(
                frag_a[1][i],
                &s_a[smem_sel * s_a_db_offset +
                     (comp_c_frag_m * 64 + i * 16) * (BK + APAD) + 16],
                BK + APAD);
        }

#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(
                frag_b[0][j],
                &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + j * 16],
                BN + BPAD);
            wmma::load_matrix_sync(
                frag_b[1][j],
                &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) +
                     comp_c_frag_n * 64 + j * 16],
                BN + BPAD);
        }

// 矩阵乘法
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        }

        COMMIT_ASYNC_GROUP();
        if (bk > 4) {
            WAIT_ASYNC_GROUP();
        }

        __syncthreads();
    }

// 处理剩余的buffers
#pragma unroll
    for (int smem_sel = (K / BK) % 4; smem_sel < 4; smem_sel++) {
        // 加载fragments并计算 (与主循环相同)
        // ...
#pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(
                frag_a[0][i],
                &s_a[smem_sel * s_a_db_offset +
                     (comp_c_frag_m * 64 + i * 16) * (BK + APAD) + 0],
                BK + APAD);
            wmma::load_matrix_sync(
                frag_a[1][i],
                &s_a[smem_sel * s_a_db_offset +
                     (comp_c_frag_m * 64 + i * 16) * (BK + APAD) + 16],
                BK + APAD);
        }

#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::load_matrix_sync(
                frag_b[0][j],
                &s_b[smem_sel * s_b_db_offset + comp_c_frag_n * 64 + j * 16],
                BN + BPAD);
            wmma::load_matrix_sync(
                frag_b[1][j],
                &s_b[smem_sel * s_b_db_offset + 16 * (BN + BPAD) +
                     comp_c_frag_n * 64 + j * 16],
                BN + BPAD);
        }

// 矩阵乘法
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j],
                               frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j],
                               frag_c[i][j]);
            }
        }
    }

    // 存储结果
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16],
                                    frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}

PLAYGROUND_MATMUL_SIG(float16_t, 14, M, N, K, A, B, C)
{
    const int BM = 128, BN = 256, BK = 32;
    dim3 blockDim(8, 32);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    const int NSPLIT = 4096;
    int split_num = (N + NSPLIT - 1) / NSPLIT;
    dim3 gridDim((BX + split_num - 1) / split_num, BY, split_num);

    cudaFuncSetAttribute(hgemm_v13_quad_buffered,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);

    unsigned int dsmem =
        4 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(float16_t);
    hgemm_v13_quad_buffered<<<gridDim, blockDim, dsmem>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
}  // namespace playground