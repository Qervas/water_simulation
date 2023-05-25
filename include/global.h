
#pragma once

const int block_size = 256;
#define EPSILON (1e-6f)
#define PI (3.14159265358979323846f)
#include <iostream>
#include <cuda_runtime.h>
#define checkCudaErrors(call) do { gpuAssert((call), __FILE__, __LINE__); } while (0)
#define CHECK_KERNEL(); 	{cudaError_t err = cudaGetLastError();if(err)printf("CUDA Error at %s:%d:\t%s\n",__FILE__,__LINE__,cudaGetErrorString(err));}
#define MAX_A (1000.0f)
inline void gpuAssert(cudaError_t err, const char* file, int line){

    if (err != cudaSuccess){
        std::cerr << "CUDA error at " << file << ":" << line << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

namespace ThrustHelper {
	template<typename T>
	struct plus {
		T _a;
		plus(const T a) :_a(a) {}
		__host__ __device__
			T operator()(const T& lhs) const {
			return lhs + _a;
		}
	};

	template <typename T>
	struct abs_plus{
		__host__ __device__
			T operator()(const T& lhs, const T& rhs) const {
			return abs(lhs) + abs(rhs);
		}
	};
}