

#pragma once
#include "Particles.h"
#include "global.h"
class SPHParticles final : public Particles {
public:
	explicit SPHParticles(const std::vector<float3>& p)
		:Particles(p),
		pressure(p.size()),
		density(p.size()),
		mass(p.size()),
		particle2Cell(p.size()) {
		CUDA_CALL(cudaMemcpy(pos.addr(), &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));
	}

	SPHParticles(const SPHParticles&) = delete;
	SPHParticles& operator=(const SPHParticles&) = delete;

	float* getPressurePtr() const {
		return pressure.addr();
	}
	const DArray<float>& getPressure() const {
		return pressure;
	}
	float* getDensityPtr() const {
		return density.addr();
	}
	const DArray<float>& getDensity() const {
		return density;
	}
	int* getParticle2Cell() const {
		return particle2Cell.addr();
	}
	float* getMassPtr() const {
		return mass.addr();
	}

	virtual ~SPHParticles() noexcept { }

protected:
	DArray<float> pressure;
	DArray<float> density;
	DArray<float> mass;
	DArray<int> particle2Cell; // lookup key
};
