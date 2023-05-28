#pragma once
#include <GL/glew.h>
#include"Camera.h"
#include<memory>
#include<cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"
#include "BasicSPHSolver.h"
#include "SPHSystem.h"
#include "global.h"
#include "ShaderUtils.h"

#include "MarchingCubes.cuh"
#define WIDTH 1920
#define HEIGHT 1440


namespace particle_attributes {
	enum { POSITION, COLOR, SIZE, };
}
extern "C" void generate_dots(float3* dot, float3* color, const std::shared_ptr<SPHParticles> particles);
extern "C" void generateMarchingCubes(float3* vertices, int* indices, const std::shared_ptr<SPHParticles>& particles, 
								int3 gridSize, float3 mcCellSize, float threshold, float sphCellLength, int* cellStart);

class Render{
public:
	Render(GLFWwindow* window);
	~Render();

	void render(float deltaTime);
	void init();
	void initSPHSystem();
	void createVBO(GLuint* vbo, const unsigned int length);
	void deleteVBO(GLuint* vbo);
	void renderParticles();
	void oneStep();
	void keyboardEvent();
	void createContainerMesh();
	void renderContainer();
	void renderSurface();
	void createSkyboxMesh();
	void renderSkybox();

	int numTriangles;



private:

	Camera camera;
	GLFWwindow* window;
	// vbo and GL variables
	GLuint container_vao;
	GLuint container_vbo;
	GLuint container_ebo;
	GLuint containerShaderProgram;
	GLuint container_texture;
	GLuint container_texture_specular_map;

	GLuint particles_vbo;
	GLuint particles_color_vbo;
	GLuint particleShaderProgram;
	
	GLuint surface_vao;
	GLuint surface_vbo[2];
	GLuint surface_ebo;
	GLuint surfaceShaderProgram;

	GLuint skybox_vao;
	GLuint skybox_vbo;
	GLuint skybox_ebo;
	GLuint skyboxShaderProgram;
	GLuint skybox_texture;
	std::string skybox_filename[6];

	//SpotLight cone
	float spotCutOff =20.0f; 
	float spotOuterCutOff = 25.5f;

	const int m_fov = 30;
	const float particle_radius = 0.004f;
	// state variables
	int frameId = 0;
	float totalTime = 0.0f;
	bool running = false;
	// particle system variables
	std::shared_ptr<SPHSystem> pSystem;
	const float3 spaceSize = make_float3(1.0f, 1.0f, 2.0f);
	const float sphSpacing = 0.02f;  
	const float sphSmoothingRadius = 2.0f * sphSpacing;
	const float sphCellLength = 1.01f * sphSmoothingRadius;
	const float dt = 0.002f;
	const float sphRho0 = 1.0f;
	const float sphRhoBoundary = 1.4f * sphRho0;
	const float sphM0 = 76.596750762082e-6f;
	const float sphStiff = 10.0f; //10.0f
	const float3 sphG = make_float3(0.0f, -9.8f, 0.0f);
	const float sphVisc = 1e-3f;  //5e-4f
	const float sphSurfaceTensionIntensity = 0.0001f;
	const float sphAirPressure = 0.0001f;
	const int3 cellSize = make_int3(ceil(spaceSize.x / sphCellLength), ceil(spaceSize.y / sphCellLength), ceil(spaceSize.z / sphCellLength));
	float3 mcCellSize = make_float3(0.004f, 0.004f, 0.004f); 
	int3 gridSize = make_int3(ceil(spaceSize.x / mcCellSize.x), ceil(spaceSize.y / mcCellSize.y), ceil(spaceSize.z / mcCellSize.z));


	DArray<int> cellStart;	

	float isolevel = 0.5;

	// MarchingCubes marchingCubes;
	std::vector<float3> vertices;
	std::vector<float3> normals;
	std::vector<uint3> indices;

	// cudaGraphicsResource_t resource_vbo, resource_particles, resources_particle_color;
};