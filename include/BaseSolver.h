

#pragma once

class BaseSolver {
public:
	virtual void step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, float3 spaceSize,
		int3 cellSize, float cellLength, float radius, float dt,
		float rho0, float rhoB, float stiff, float visc, float3 G,
		float surfaceTensionIntensity, float airPressure) = 0;
	virtual ~BaseSolver(){}
protected:
	virtual void advect(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize) = 0;
	virtual void force(std::shared_ptr<SPHParticles>& fluids, float dt, float3 G) = 0;
};
