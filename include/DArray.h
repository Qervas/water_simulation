

#pragma once
#include "global.h"
#include <cuda_runtime.h>
template <typename T>
class DArray {
	static_assert(
		std::is_same<T, float3>::value || std::is_same<T, float>::value || 
		std::is_same<T, int>::value, "DataArray must be of int, float or float3.");
public:
	explicit DArray(const unsigned int length) :
		_length(length),
		d_array([length]() {
			T* ptr; 
			checkCudaErrors(cudaMalloc((void**)& ptr, sizeof(T) * length));
			std::shared_ptr<T> t(new(ptr)T[length], [](T* ptr) {checkCudaErrors(cudaFree(ptr)); });
			return t;
		}()) {
			this->clear();
	}

	DArray(const DArray&) = delete;
	DArray& operator=(const DArray&) = delete;

	T* addr(const int offset= 0) const {
		return d_array.get() + offset;
	}

	unsigned int length() const { return _length; }
	void clear(){ checkCudaErrors(cudaMemset(this->addr(), 0, sizeof(T) * this->length())); }
	
	~DArray() noexcept { }

private:
	const unsigned int _length;
	const std::shared_ptr<T> d_array;
};