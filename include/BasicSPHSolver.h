

#pragma once
class BasicSPHSolver{
public:
	void step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, float3 spaceSize,
		int3 cellSize, float cellLength, float radius, float dt,
		float rho0, float rhoB, float stiff, float visc, float3 G,
		float surfaceTensionIntensity, float airPressure) ;
	explicit BasicSPHSolver(int num) :bufferFloat3(num) {}
	 ~BasicSPHSolver() noexcept { }
protected:
	 void force(std::shared_ptr<SPHParticles>& fluids, float dt, float3 G);
	 void advect(std::shared_ptr<SPHParticles>& fluids, float dt, float3 spaceSize);
	 void pressure(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, float rho0, float stiff,
		int3 cellSize, float cellLength, float radius, float dt);
	 void diffuse(std::shared_ptr<SPHParticles>& fluids, const DArray<int>& cellStartFluid,
		int3 cellSize, float cellLength, float rho0,
		float radius, float visc, float dt);
	 void handleSurface(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		float rho0, float rhoB, int3 cellSize, float cellLength, float radius,
		float dt, float surfaceTensionIntensity, float airPressure);
private:
	DArray<float3> bufferFloat3;
	void computeDensity(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, int3 cellSize, float cellLength, float radius) const;
	void surfaceDetection(DArray<float3>& colorGrad, const std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		float rho0, float rhoB, int3 cellSize, float cellLength, float radius);
	void applySurfaceEffects(std::shared_ptr<SPHParticles>& fluids, const DArray<float3>& colorGrad,
		const DArray<int>& cellStartFluid, float rho0, int3 cellSize, float cellLength,
		float radius, float dt, float surfaceTensionIntensity, float airPressure);
};