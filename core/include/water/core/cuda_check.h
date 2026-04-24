#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace water::detail {

[[noreturn]] inline void cuda_throw(cudaError_t err, const char* expr, const char* file, int line) {
    std::string msg = "CUDA error at ";
    msg += file; msg += ":"; msg += std::to_string(line);
    msg += "\n  call: "; msg += expr;
    msg += "\n  err:  "; msg += cudaGetErrorString(err);
    throw std::runtime_error(msg);
}

} // namespace water::detail

// WATER_CUDA_CHECK(expr) — invoke `expr` and throw on cudaError != cudaSuccess.
#define WATER_CUDA_CHECK(expr)                                                \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            ::water::detail::cuda_throw(_e, #expr, __FILE__, __LINE__);       \
        }                                                                     \
    } while (0)

// WATER_CUDA_CHECK_LAST() — checks cudaGetLastError(); use after kernel launches.
#define WATER_CUDA_CHECK_LAST()                                               \
    do {                                                                      \
        cudaError_t _e = cudaGetLastError();                                  \
        if (_e != cudaSuccess) {                                              \
            ::water::detail::cuda_throw(_e, "cudaGetLastError", __FILE__, __LINE__); \
        }                                                                     \
    } while (0)
