#include "MarchingCubes.cuh"
#include "MarchingCubesLUT.h"
// MarchingCubes.cpp
#include <device_launch_parameters.h>
#include <cmath>
#include <device_atomic_functions.h>
#include <helper_math.h>

__device__ int d_numParticles;

__device__ float d_isolevel = 0.0f; //some tuned value between density max and min
__device__ float d_sphSpacing = 0.02f;  

inline __device__ float cubic_spline_kernel(const float r, const float radius){
	const auto q = 2.0f * fabs(r) / radius;
	if (q > 2.0f || q < EPSILON) return 0.0f;
	else {
		const auto a = 0.25f / (PI * radius * radius * radius);
		return a * ((q > 1.0f) ? (2.0f - q) * (2.0f - q) * (2.0f - q) : ((3.0f * q - 6.0f) * q * q + 4.0f));
	}
}

MarchingCubes::MarchingCubes(std::shared_ptr<SPHParticles> particles, float isolevel, float* density,
                            int* particle2Cell)
    : h_particles(particles), h_isolevel(isolevel),  h_density(density),h_particle2Cell(particle2Cell) {}

MarchingCubes::~MarchingCubes() {}

void MarchingCubes::generateMesh_CUDA() {
    h_numParticles = h_particles->size();
    cudaMemcpy(&d_numParticles, &h_numParticles, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_isolevel, &h_isolevel, sizeof(float), cudaMemcpyHostToDevice);
    // Allocate device memory for vertices
    float3* d_vertices;
    cudaMalloc(&d_vertices, h_numParticles * 12 * sizeof(float3));

    // Allocate device memory for indices
    uint3* d_indices;
    cudaMalloc(&d_indices, h_numParticles * 5 * sizeof(uint3));

    // Call the generateVerticesKernel
    int blockSize = 256;
    int gridSize = (h_numParticles + blockSize - 1) / blockSize;
    
    generateVerticesKernel<<<gridSize, blockSize>>>(h_particles, d_vertices, d_indices, h_particles->getPosPtr(),
                                                    h_density, h_particle2Cell, h_numParticles);
    cudaDeviceSynchronize();

    // Allocate device memory for numVerticesGenerated
    int* d_numVerticesGenerated;
    cudaMalloc(&d_numVerticesGenerated, sizeof(int));
    cudaMemset(d_numVerticesGenerated, 0, sizeof(int));

    // Call the countVerticesKernel
    int numTriangles = h_numParticles * 5;
    gridSize = (numTriangles + blockSize - 1) / blockSize;
    countVerticesKernel<<<gridSize, blockSize>>>(d_indices, d_numVerticesGenerated, numTriangles);
    cudaDeviceSynchronize();

    // Copy the number of vertices generated from device memory to host memory
    cudaMemcpy(&numVertices, d_numVerticesGenerated, sizeof(int), cudaMemcpyDeviceToHost);

    // Resize the host-side vectors
    h_vertices.resize(numVertices);

    // Copy the vertices to the host-side vectors
    cudaMemcpy(h_vertices.data(), d_vertices, numVertices * sizeof(float3), cudaMemcpyDeviceToHost);

    // =============normals===================
    // Allocate device memory for normals
    float3* d_normals;
    cudaMalloc(&d_normals, numVertices * sizeof(float3));

    // Call the generateNormalsKernel
    // int blockSize = 256;
    gridSize = (numVertices + blockSize - 1) / blockSize;
    generateNormalsKernel<<<gridSize, blockSize>>>(d_normals, d_vertices, d_indices, numVertices);
    cudaDeviceSynchronize();

    // Resize the host-side vectors
    h_normals.resize(numVertices);

    // Copy the generated normals from device memory to host memory
    cudaMemcpy(h_normals.data(), d_normals, numVertices * sizeof(float3), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vertices);
    cudaFree(d_indices);
    cudaFree(d_normals);
    cudaFree(d_numVerticesGenerated);
}




std::vector<float3> MarchingCubes::getVertices() {
    return h_vertices;
}

std::vector<uint3> MarchingCubes::getIndices() {
    return h_indices;
}

std::vector<float3> MarchingCubes::getNormals() {
    return h_normals;
}

__device__ int findSharedVertexIndex(const float3* vertices, float3 targetVertex, int numVertices) {
    const float epsilon = 1e-6f;

    for (int i = 0; i < numVertices; ++i) {
        float3 vertex = vertices[i];
        if (fabs(vertex.x - targetVertex.x) < epsilon &&
            fabs(vertex.y - targetVertex.y) < epsilon &&
            fabs(vertex.z - targetVertex.z) < epsilon) {
            return i;
        }
    }

    return -1;  // Vertex not found
}


__global__ void generateVerticesKernel(std::shared_ptr<SPHParticles> particles, float3* vertices, uint3* indices,
                                    float3* posPtr,const float* density, const int* particle2Cell, int num) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    // Calculate the cell position
    float3 cellPos = posPtr[i];
    
    // Get the density values for the 8 corners of the cell
    float cornerDensity[8];
    // ... Fill the cornerDensity array with the density values from the given density array
    for (int j = 0; j < 8; ++j) {
        // Calculate the position of the corner
        float3 cornerPos = cellPos + make_float3((j & 1) ? d_sphSpacing : 0,
                                                 (j & 2) ? d_sphSpacing : 0,
                                                 (j & 4) ? d_sphSpacing : 0);

        // Find the particles within a neighborhood of the corner position
        // and compute their contributions to the density at the corner position
        float totalDensity = 0.0f;
        float totalKernelValue = 0.0f;
        for (int k = 0; k < num; ++k) {
            float3 particlePos = posPtr[k];
            float distance = length(cornerPos - particlePos);
            if (distance < d_sphSpacing) {
                float kernelValue = cubic_spline_kernel(distance, d_sphSpacing);
                totalDensity += density[k] * kernelValue;
                totalKernelValue += kernelValue;
            }
        }

        // Normalize the density value at the corner position
        if (totalKernelValue > 1e-6f) {
            cornerDensity[j] = totalDensity / totalKernelValue;
        } else {
            cornerDensity[j] = 0.0f;
        }
    }
    // Calculate the marching cubes index based on the d_isolevel and the corner densities
    int cubeIndex = 0;
    for (int j = 0; j < 8; ++j) {
        if (cornerDensity[j] < d_isolevel) {
            cubeIndex |= (1 << j);
        }
    }

    // Calculate the vertex positions for the cell
    float3 cellVertices[12];
    int cellVertexIndices[12];
    if (edgeTable[cubeIndex] == 0) {
        return;  // No intersection with the isosurface
    }
    // Interpolate the vertex positions on the edges
    if (edgeTable[cubeIndex] & 1) {
        cellVertices[0] = vertexInterpolation(d_isolevel, cellPos, cellPos + make_float3(d_sphSpacing, 0, 0), cornerDensity[0], cornerDensity[1]);
        cellVertexIndices[0] = findSharedVertexIndex(vertices, cellVertices[0], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 2) {
        cellVertices[1] = vertexInterpolation(d_isolevel, cellPos + make_float3(d_sphSpacing, 0, 0), cellPos + make_float3(d_sphSpacing, d_sphSpacing, 0), cornerDensity[1], cornerDensity[3]);
        cellVertexIndices[1] = findSharedVertexIndex(vertices, cellVertices[1], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 4) {
        cellVertices[2] = vertexInterpolation(d_isolevel, cellPos + make_float3(d_sphSpacing, d_sphSpacing, 0), cellPos + make_float3(0, d_sphSpacing, 0), cornerDensity[3], cornerDensity[2]);
        cellVertexIndices[2] = findSharedVertexIndex(vertices, cellVertices[2], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 8) {
        cellVertices[3] = vertexInterpolation(d_isolevel, cellPos + make_float3(0, d_sphSpacing, 0), cellPos, cornerDensity[2], cornerDensity[0]);
        cellVertexIndices[3] = findSharedVertexIndex(vertices, cellVertices[3], d_numParticles);

    }
    if (edgeTable[cubeIndex] & 16) {
        cellVertices[4] = vertexInterpolation(d_isolevel, cellPos + make_float3(0, 0, d_sphSpacing), cellPos + make_float3(d_sphSpacing, 0, d_sphSpacing), cornerDensity[4], cornerDensity[5]);
        cellVertexIndices[4] = findSharedVertexIndex(vertices, cellVertices[4], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 32) {
        cellVertices[5] = vertexInterpolation(d_isolevel, cellPos + make_float3(d_sphSpacing, 0, d_sphSpacing), cellPos + make_float3(d_sphSpacing, d_sphSpacing, d_sphSpacing), cornerDensity[5], cornerDensity[7]);
        cellVertexIndices[5] = findSharedVertexIndex(vertices, cellVertices[5], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 64) {
        cellVertices[6] = vertexInterpolation(d_isolevel, cellPos + make_float3(d_sphSpacing, d_sphSpacing, d_sphSpacing), cellPos + make_float3(0, d_sphSpacing, d_sphSpacing), cornerDensity[7], cornerDensity[6]);
        cellVertexIndices[6] = findSharedVertexIndex(vertices, cellVertices[6], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 128) {
        cellVertices[7] = vertexInterpolation(d_isolevel, cellPos + make_float3(0, d_sphSpacing, d_sphSpacing), cellPos + make_float3(0, 0, d_sphSpacing), cornerDensity[6], cornerDensity[4]);
        cellVertexIndices[7] = findSharedVertexIndex(vertices, cellVertices[7], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 256) {
        cellVertices[8] = vertexInterpolation(d_isolevel, cellPos, cellPos + make_float3(0, 0, d_sphSpacing), cornerDensity[0], cornerDensity[4]);
        cellVertexIndices[8] = findSharedVertexIndex(vertices, cellVertices[8], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 512) {
        cellVertices[9] = vertexInterpolation(d_isolevel, cellPos + make_float3(d_sphSpacing, 0, 0), cellPos + make_float3(d_sphSpacing, 0, d_sphSpacing), cornerDensity[1], cornerDensity[5]);
        cellVertexIndices[9] = findSharedVertexIndex(vertices, cellVertices[9], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 1024) {
        cellVertices[10] = vertexInterpolation(d_isolevel, cellPos + make_float3(d_sphSpacing, d_sphSpacing, 0), cellPos + make_float3(d_sphSpacing, d_sphSpacing, d_sphSpacing), cornerDensity[3], cornerDensity[7]);
        cellVertexIndices[10] = findSharedVertexIndex(vertices, cellVertices[10], d_numParticles);
    }
    if (edgeTable[cubeIndex] & 2048) {
        cellVertices[11] = vertexInterpolation(d_isolevel, cellPos + make_float3(0, d_sphSpacing, 0), cellPos + make_float3(0, d_sphSpacing, d_sphSpacing), cornerDensity[2], cornerDensity[6]);
        cellVertexIndices[11] = findSharedVertexIndex(vertices, cellVertices[11], d_numParticles);
    }


    // Store the vertex positions in the output vertices array
    for (int j = 0; triTable[cubeIndex][j] != -1; j += 3) {
        vertices[i * 15 + j]     = cellVertices[triTable[cubeIndex][j]];
        vertices[i * 15 + j + 1] = cellVertices[triTable[cubeIndex][j + 1]];
        vertices[i * 15 + j + 2] = cellVertices[triTable[cubeIndex][j + 2]];
    }
    // Store the vertex indices in the output indices array
    for (int j = 0; triTable[cubeIndex][j] != -1; j += 3) {
        indices[i * 5 + j / 3] = make_uint3(cellVertexIndices[triTable[cubeIndex][j]],
                                            cellVertexIndices[triTable[cubeIndex][j + 1]],
                                            cellVertexIndices[triTable[cubeIndex][j + 2]]);
    }
}
       

__global__ void generateNormalsKernel(float3* normals, const float3* vertices, const uint3* indices, int num) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    uint3 triangle = indices[i];
    float3 v1 = vertices[triangle.x];
    float3 v2 = vertices[triangle.y];
    float3 v3 = vertices[triangle.z];

    float3 edge1 = make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.z);
    float3 edge2 = make_float3(v3.x - v1.x, v3.y - v1.y, v3.z - v1.z);

    float3 normal = cross(edge1, edge2);
    normal = normalize(normal);

    normals[triangle.x] = normal;
    normals[triangle.y] = normal;
    normals[triangle.z] = normal;
}

__global__ void countVerticesKernel(const uint3* indices, int* numVertices, int numTriangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    atomicAdd(numVertices, 3);
}


__device__ float interpolateDensity(float isolevel, float density1, float density2) {
    return (isolevel - density1) / (density2 - density1);
}

__device__ float3 vertexInterpolation(float isolevel, float3 pos1, float3 pos2, float density1, float density2) {
    float t = interpolateDensity(isolevel, density1, density2);
    return make_float3(pos1.x + t * (pos2.x - pos1.x),
                       pos1.y + t * (pos2.y - pos1.y),
                       pos1.z + t * (pos2.z - pos1.z));
}

