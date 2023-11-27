#pragma once
#include "global.h"
#include <cuda_runtime.h>
#include "CudaMemory.h"  // Include the CudaMemory header

template <typename T>
class DArray {
    static_assert(
        std::is_same<T, float3>::value || std::is_same<T, float>::value || 
        std::is_same<T, int>::value, "DArray must be of int, float, or float3.");

public:
    explicit DArray(unsigned int length) :
        _length(length),
        d_array(length) {
        this->clear();
    }

    DArray(const DArray&) = delete;
    DArray& operator=(const DArray&) = delete;

    T* addr(int offset = 0) const {
        return d_array.get() + offset;
    }

    unsigned int length() const { return _length; }
    void clear() { 
        if (_length > 0) {
            checkCudaErrors(cudaMemset(this->addr(), 0, sizeof(T) * _length)); 
        }
    }

    ~DArray() noexcept { }

private:
    const unsigned int _length;
    CudaMemory<T> d_array;  // Replace std::shared_ptr with CudaMemory
};
