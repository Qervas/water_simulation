#pragma once

#include "water/core/cuda_check.h"
#include "water/core/types.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace water {

// A move-only RAII wrapper around stream-ordered CUDA memory.
//
// The buffer is allocated on a CUDA stream (default: the per-thread default
// stream) and freed on that same stream. This avoids global synchronization
// that traditional cudaMalloc/cudaFree imposes.
//
// Usage:
//   DeviceBuffer<float> buf(1024);              // allocates on default stream
//   buf.fill_zero();
//   kernel<<<grid, block>>>(buf.data(), buf.size());
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count, cudaStream_t stream = 0)
        : ptr_(nullptr), size_(count), stream_(stream) {
        if (count > 0) {
            WATER_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_),
                                              count * sizeof(T), stream_));
        }
    }

    ~DeviceBuffer() {
        if (ptr_) {
            // cudaFreeAsync silently ignores errors during destruction; we
            // use the no-throw form deliberately.
            cudaFreeAsync(ptr_, stream_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), stream_(other.stream_) {
        other.ptr_ = nullptr; other.size_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeAsync(ptr_, stream_);
            ptr_ = other.ptr_; size_ = other.size_; stream_ = other.stream_;
            other.ptr_ = nullptr; other.size_ = 0;
        }
        return *this;
    }

    // Raw device pointer. Valid until destruction.
    T*       data()       noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // Set every byte to zero on the stream.
    void fill_zero() {
        if (ptr_ && size_) {
            WATER_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream_));
        }
    }

    // Async copy from host pointer.
    void copy_from_host(const T* src, std::size_t count) {
        WATER_CUDA_CHECK(cudaMemcpyAsync(ptr_, src, count * sizeof(T),
                                          cudaMemcpyHostToDevice, stream_));
    }

    // Async copy to host pointer.
    void copy_to_host(T* dst, std::size_t count) const {
        WATER_CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, count * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream_));
    }

    cudaStream_t stream() const noexcept { return stream_; }

private:
    T*           ptr_    = nullptr;
    std::size_t  size_   = 0;
    cudaStream_t stream_ = 0;
};

} // namespace water
