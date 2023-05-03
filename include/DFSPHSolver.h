

#pragma once

class DFSPHSolver final : public BasicSPHSolver {
public:
	virtual void step(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary, float3 spaceSize,
		int3 cellSize, float cellLength, float radius, float dt,
		float rho0, float rhoB, float stiff, float visc, float3 G,
		float surfaceTensionIntensity, float airPressure) override;
	explicit DFSPHSolver(int num,
		float defaultDensityErrorThreshold = 1e-3f,
		float defaultDivergenceErrorThreshold = 1e-3f,
		int defaultMaxIter = 20)
		:BasicSPHSolver(num),
		alpha(num),
		bufferFloat(num),
		bufferInt(num),
		error(num),
		denWarmStiff(num),
		densityErrorThreshold(defaultDensityErrorThreshold),
		divergenceErrorThreshold(defaultDivergenceErrorThreshold), 
		maxIter(defaultMaxIter){}
	virtual ~DFSPHSolver() noexcept { }
protected:
	// overwrite and hide the project function in BasicSPHSolver
	// in project, correct density error from alpha
	virtual int project(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		float rho0, int3 cellSize, float cellLength, float radius, float dt,
		float errorThreshold, int maxIter);

private:
	void computeDensityAlpha(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		int3 cellSize, float cellLength, float radius);
	int correctDivergenceError(std::shared_ptr<SPHParticles>& fluids, const std::shared_ptr<SPHParticles>& boundaries,
		const DArray<int>& cellStartFluid, const DArray<int>& cellStartBoundary,
		float rho0, int3 cellSize, float cellLength, float radius, float dt,
		float errorThreshold, int maxIter);
	DArray<float> alpha;
	DArray<float> bufferFloat;
	DArray<int> bufferInt;
	DArray<float> error;
	DArray<float> denWarmStiff;
	const float densityErrorThreshold;
	const float divergenceErrorThreshold;
	const int maxIter;
};