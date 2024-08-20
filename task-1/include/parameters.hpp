#include <cstdint>
#include <cuda_fp16.h>
#include <string_view>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace params
{

// =============================================================================
// @Note `DataType` and `MatmulVersion` are managed by CMake automatically.
// -----------------------------------------------------------------------------
#ifdef TEST_FLOAT16
using DataType = playground::float16_t;
constexpr std::string_view DataTypeName = "f16";
#else
using DataType = playground::float32_t;
constexpr std::string_view DataTypeName = "f32";
#endif
#ifndef TEST_KERNEL_VERSION
    #define TEST_KERNEL_VERSION playground::MatmulcBlasVersion
#endif
constexpr auto MatmulVersion = playground::uint8_t(TEST_KERNEL_VERSION);
// -----------------------------------------------------------------------------

// Size of Matrices
// mat(M,K)@mat(K,N)=mat(M,N)
constexpr playground::size_t M = 4096, N = 4096, K = 4096;

// Repeated Times
constexpr playground::size_t NumRep = 100;

// Warmup Times
constexpr playground::size_t NumWarmup = 10;

}  // namespace params
