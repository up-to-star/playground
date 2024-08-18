#pragma once

#include "playground/system.hpp"

namespace playground
{

template <typename DType, uint8_t Version>
void matmul(const size_t M, const size_t N, const size_t K,
            const DType* const A, const DType* const B,
            DType* const C) = delete;

#define cBLAS_VERSION 0
#define cuBLAS_VERSION 1

#define MATMUL(DType, Version)                                                 \
    template <>                                                                \
    void matmul<DType, Version>(const size_t M, const size_t N,                \
                                const size_t K, const DType* const A,          \
                                const DType* const B, DType* const C);         \

// =============================================================================
// Declaration of library matmul functions.
// -----------------------------------------------------------------------------
/**
 * @brief Matrix multiplication, fp16, cBLAS.
 */
MATMUL(float16_t, cBLAS_VERSION)

/**
 * @brief Matrix multiplication, fp32, cBLAS.
 */
MATMUL(float32_t, cBLAS_VERSION)

/**
 * @brief Matrix multiplication, fp16, cuBLAS.
 */
MATMUL(float16_t, cuBLAS_VERSION)

/**
 * @brief Matrix multiplication, fp32, cuBLAS.
 */
MATMUL(float32_t, cuBLAS_VERSION)

// =============================================================================
// Declaration of self-implemented matmul functions.
// e.g. MATMUL(float16_t, 2)
//      MATMUL(float32_t, 2)
// -----------------------------------------------------------------------------

// ...

}  // namespace playground