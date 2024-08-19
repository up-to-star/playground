#pragma once

#include "playground/system.hpp"

namespace playground
{

template <typename DType, uint8_t Version>
void matmul(const size_t M, const size_t N, const size_t K,
            const DType* const A, const DType* const B,
            DType* const C) = delete;

constexpr uint8_t MatmulcBlasVersion = 0;
constexpr uint8_t MatmulcuBlasVersion = 1;

// Playground matmul signature.
#define PG_MATMUL(DType, Version, M, N, K, A, B, C)                    \
    template <>                                                                \
    void matmul<DType, Version>(const size_t M, const size_t N,                \
                                const size_t K, const DType* const A,          \
                                const DType* const B, DType* const C)

// =============================================================================
// Declaration of library matmul functions.
// -----------------------------------------------------------------------------
/**
 * @brief Matrix multiplication, fp16, cBLAS.
 */
PG_MATMUL(float16_t, MatmulcBlasVersion, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32, cBLAS.
 */
PG_MATMUL(float32_t, MatmulcBlasVersion, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp16, cuBLAS.
 */
PG_MATMUL(float16_t, MatmulcuBlasVersion, M, N, K, A, B, C);

/**
 * @brief Matrix multiplication, fp32, cuBLAS.
 */
PG_MATMUL(float32_t, MatmulcuBlasVersion, M, N, K, A, B, C);

// =============================================================================
// Declaration of self-implemented matmul functions.
// e.g. PG_MATMUL(float16_t, 2)
//      PG_MATMUL(float32_t, 2)
// -----------------------------------------------------------------------------

// ...

}  // namespace playground