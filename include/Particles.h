

#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "DArray.h"
#include "global.h"
class Particles {
public:
	explicit Particles(const std::vector<float3>& p)
		:pos(p.size()), vel(p.size()) {
		checkCudaErrors(cudaMemcpy(pos.addr(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));
	}

	Particles(const Particles&) = delete;
	Particles& operator=(const Particles&) = delete;

	unsigned int size() const {
		return pos.length();
	}
	float3* getPosPtr() const {
		return pos.addr();
	}
	float3* getVelPtr() const {
		return vel.addr();
	}
	const DArray<float3>& getPos() const {
		return pos;
	}
	void advect(float dt);

	virtual ~Particles() noexcept { }

protected:
	DArray<float3> pos;
	DArray<float3> vel;
};
