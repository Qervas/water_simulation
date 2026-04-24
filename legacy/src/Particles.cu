

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include "DArray.h"
#include "Particles.h"

void Particles::advect(float dt)
{	
	thrust::transform(thrust::device,
		pos.addr(), pos.addr() + size(),
		vel.addr(),
		pos.addr(),
		[dt]__host__ __device__(const float3& lhs, const float3& rhs) { return lhs + dt*rhs; }	
	);
}
