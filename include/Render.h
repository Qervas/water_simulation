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
#include "BaseSolver.h"
#include "BasicSPHSolver.h"
#include "DFSPHSolver.h"
#include "PBDSolver.h"
#include "SPHSystem.h"
#include "global.h"
#include "ShaderUtils.h"

#define WIDTH 1920
#define HEIGHT 1080

enum  fluid_solver { SPH, DFSPH, PBD};

namespace particle_attributes {
	enum { POSITION, COLOR, SIZE, };
}
extern "C" void generate_dots(float3* dot, float3* color, const std::shared_ptr<SPHParticles> particles);

class Render{
public:
	Render(GLFWwindow* window);
	~Render();

	void render(float deltaTime);
	void init();
	void initSPHSystem(fluid_solver solver);
	void createVBO(GLuint* vbo, const unsigned int length);
	void deleteVBO(GLuint* vbo);
	void renderParticles();
	void oneStep();
	void keyboardEvent();
	void createContainerMesh();
	void renderContainer();
	


private:

	Camera camera;
	GLFWwindow* window;
	// vbo and GL variables
	GLuint container_vao;
	GLuint container_vbo;
	GLuint container_ebo;
	GLuint containerShaderProgram;

	GLuint particlesVBO;
	GLuint particlesColorVBO;
	GLuint m_particles_program;
	// const int m_window_h = 700;
	const int m_fov = 30;
	const float particle_radius = 0.001f;
	// view variables
	float rot[2] = { 0.0f, 0.0f };
	int mousePos[2] = { -1,-1 };
	bool mouse_left_down = false;
	float zoom = 0.3f;
	// state variables
	int frameId = 0;
	float totalTime = 0.0f;
	bool running = false;
	// particle system variables
	std::shared_ptr<SPHSystem> pSystem;
	const float3 spaceSize = make_float3(1.0f, 1.0f, 1.0f);
	const float sphSpacing = 0.02f;  
	const float sphSmoothingRadius = 2.0f * sphSpacing;
	const float sphCellLength = 1.01f * sphSmoothingRadius;
	const float dt = 0.002f;
	const float sphRho0 = 1.0f;
	const float sphRhoBoundary = 1.4f * sphRho0;
	const float sphM0 = 76.596750762082e-6f;
	const float sphStiff = 10.0f;
	const float3 sphG = make_float3(0.0f, -9.8f, 0.0f);
	const float sphVisc = 5e-4f;
	const float sphSurfaceTensionIntensity = 0.0001f;
	const float sphAirPressure = 0.0001f;
	const int3 cellSize = make_int3(ceil(spaceSize.x / sphCellLength), ceil(spaceSize.y / sphCellLength), ceil(spaceSize.z / sphCellLength));

};