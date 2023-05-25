

#pragma once
#include<memory>
#include "BasicSPHSolver.h"

class SPHSystem {
public:
	SPHSystem(
		std::shared_ptr<SPHParticles>& fluidParticles,
		std::shared_ptr<SPHParticles>& boundaryParticles,
		std::shared_ptr<BasicSPHSolver>& solver,
		float3 spaceSize,
		float sphCellLength,
		float sphSmoothingRadius,
		float dt,
		float sphM0,
		float sphRho0,
		float sphRhoBoundary,
		float sphStiff,
		float sphVisc,
		float sphSurfaceTensionIntensity,
		float sphAirPressure,
		float3 sphG,
		int3 cellSize);
	SPHSystem(const SPHSystem&) = delete;
	SPHSystem& operator=(const SPHSystem&) = delete;

	float step();

	int size() const {
		return fluidSize();
	}
	int fluidSize() const {
		return (*_fluids).size();
	}
	int boundarySize() const {
		return (*_boundaries).size();
	}
	int totalSize() const {
		return (*_fluids).size() + (*_boundaries).size();
	}
	auto getFluids() const {
		return static_cast<const std::shared_ptr<SPHParticles>>(_fluids);
	}
	auto getBoundaries() const {
		return static_cast<const std::shared_ptr<SPHParticles>>(_boundaries);
	}
	~SPHSystem() noexcept { }
private:
	std::shared_ptr<SPHParticles> _fluids;
	const std::shared_ptr<SPHParticles> _boundaries;
	std::shared_ptr<BasicSPHSolver> _solver;
	DArray<int> cellStartFluid;
	DArray<int> cellStartBoundary;
	const float3 _spaceSize;
	const float _sphSmoothingRadius;
	const float _sphCellLength;
	const float _dt;
	const float _sphRho0;
	const float _sphRhoBoundary;
	const float _sphStiff;
	const float3 _sphG;
	const float _sphVisc;
	const float _sphSurfaceTensionIntensity;
	const float _sphAirPressure;
	const int3 _cellSize;
	DArray<int> bufferInt;
	void computeBoundaryMass();
	void neighborSearch(const std::shared_ptr<SPHParticles>& particles, DArray<int>& cellStart);
};