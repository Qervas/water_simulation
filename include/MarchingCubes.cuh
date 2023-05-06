#pragma once

#include <memory>
#include <vector>
#include "SPHParticles.h"


__global__ void generateVerticesKernel(std::shared_ptr<SPHParticles> particles, float3* vertices, uint3* indices, 
                                    float3* posPtr,const float* density, const int* particle2Cell, int num);
__global__ void generateNormalsKernel(float3* normals, const float3* vertices, const uint3* indices, int num);
__global__ void countVerticesKernel(const uint3* indices, int* numVertices, int numTriangles) ;
__device__ float interpolateDensity(float d_isolevel, float density1, float density2);
__device__ float3 vertexInterpolation(float d_isolevel, float3 pos1, float3 pos2, float density1, float density2);


class MarchingCubes {
public:
    MarchingCubes() = default;
    MarchingCubes(std::shared_ptr<SPHParticles> particles, float d_isolevel, float* density,
                            int* particle2Cell);
    ~MarchingCubes();

    void generateMesh_CUDA();

    std::vector<float3> getVertices();
    std::vector<uint3> getIndices();
    std::vector<float3> getNormals();

private:
    std::shared_ptr<SPHParticles> h_particles;
    float h_isolevel;
    float* h_density;
    int* h_particle2Cell;
    int h_numParticles;

    std::vector<float3> h_vertices;
    std::vector<uint3> h_indices;
    std::vector<float3> h_normals;

    int numVertices;
};

