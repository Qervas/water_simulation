#include "Render.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
Render::Render(GLFWwindow* window):camera(window), window(window){

	createContainerMesh();
	// create VBOs

	
	GLuint vertexShader = ShaderUtils::loadShader("shaders/particles.vert", GL_VERTEX_SHADER);
    GLuint fragmentShader = ShaderUtils::loadShader("shaders/particles.frag", GL_FRAGMENT_SHADER);
    m_particles_program = ShaderUtils::createShaderProgram(vertexShader, fragmentShader);
	glBindAttribLocation(m_particles_program, particle_attributes::SIZE, "pointSize");
    GLuint containerVertexShader = ShaderUtils::loadShader("shaders/container.vert", GL_VERTEX_SHADER);
    GLuint containerFragmentShader = ShaderUtils::loadShader("shaders/container.frag", GL_FRAGMENT_SHADER);
    containerShaderProgram = ShaderUtils::createShaderProgram(containerVertexShader, containerFragmentShader);

    // Clean up shader resources
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(containerVertexShader);
    glDeleteShader(containerFragmentShader);
}

Render::~Render(){
	deleteVBO(&particlesVBO);
	deleteVBO(&particlesColorVBO);
	pSystem = nullptr;
	CUDA_CALL(cudaDeviceReset());
}
void Render::render(float deltaTime){
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, WIDTH, HEIGHT);
	if (running) {
		oneStep();
	}
	camera.update(deltaTime);
	renderParticles();

	renderContainer();
	keyboardEvent();
}

void Render::init(){
	initSPHSystem(SPH);
	createVBO(&particlesVBO, sizeof(float3) * pSystem->size());
	createVBO(&particlesColorVBO, sizeof(float3) * pSystem->size());
}



void Render::initSPHSystem(fluid_solver solver) {
	// initiate fluid particles
	std::vector<float3> pos;
	for (auto i = 0; i < 36; ++i) {
		for (auto j = 0; j < 24; ++j) {
			for (auto k = 0; k < 24; ++k) {
				auto x = make_float3(0.27f + sphSpacing * j,//0.27f
					0.10f + sphSpacing * i,//0.10f
					0.27f + sphSpacing * k);//0.27f
				pos.push_back(x);
			}
		}
	}
	auto fluidParticles = std::make_shared<SPHParticles>(pos);
	// initiate boundary particles
	pos.clear();
	const auto compactSize = 2 * make_int3(ceil(spaceSize.x / sphCellLength), ceil(spaceSize.y / sphCellLength), ceil(spaceSize.z / sphCellLength));
	// front and back
	for (auto i = 0; i < compactSize.x; ++i) {
		for (auto j = 0; j < compactSize.y; ++j) {
			auto x = make_float3(i, j, 0) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(i, j, compactSize.z - 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	// top and bottom
	for (auto i = 0; i < compactSize.x; ++i) {
		for (auto j = 0; j < compactSize.z-2; ++j) {
			auto x = make_float3(i, 0, j+1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(i, compactSize.y - 1, j+1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	// left and right
	for (auto i = 0; i < compactSize.y - 2; ++i) {
		for (auto j = 0; j < compactSize.z - 2; ++j) {
			auto x = make_float3(0, i + 1, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(compactSize.x - 1, i + 1, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	auto boundaryParticles = std::make_shared<SPHParticles>(pos);
	// initiate solver and particle system
	std::shared_ptr<BaseSolver> pSolver;
	switch (solver) {
	case fluid_solver::PBD:
		pSolver = std::make_shared<PBDSolver>(fluidParticles->size());
		break;
	case fluid_solver::DFSPH:
		pSolver = std::make_shared<DFSPHSolver>(fluidParticles->size());
		break;
	default:
		pSolver = std::make_shared<BasicSPHSolver>(fluidParticles->size());
		break;
	}	
	pSystem = std::make_shared<SPHSystem>(fluidParticles, boundaryParticles, pSolver,
		spaceSize, sphCellLength, sphSmoothingRadius, dt, sphM0,
		sphRho0, sphRhoBoundary, sphStiff, sphVisc, 
		sphSurfaceTensionIntensity, sphAirPressure, sphG, cellSize);
}

void Render::createVBO(GLuint* vbo, const unsigned int length) {
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, length, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	CUDA_CALL(cudaGLRegisterBufferObject(*vbo));
}

void Render::deleteVBO(GLuint* vbo) {
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	CUDA_CALL(cudaGLUnregisterBufferObject(*vbo));
	*vbo = NULL;
}

void Render::renderParticles() {

	glUseProgram(m_particles_program);
	glUniform1f(glGetUniformLocation(m_particles_program, "pointScale"), WIDTH / tanf(m_fov*0.5f*float(PI) / 180.0f));
	glUniform1f(glGetUniformLocation(m_particles_program, "pointRadius"), particle_radius);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
    // map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    float3 *cptr;
    CUDA_CALL(cudaGLMapBufferObject((void**)&dptr, particlesVBO));
    CUDA_CALL(cudaGLMapBufferObject((void**)&cptr, particlesColorVBO));

    // calculate the dots' position and color
    generate_dots(dptr, cptr, pSystem->getFluids());

    // unmap buffer object
    CUDA_CALL(cudaGLUnmapBufferObject(particlesVBO));
    CUDA_CALL(cudaGLUnmapBufferObject(particlesColorVBO));

    glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, particlesColorVBO);
    glColorPointer(3, GL_FLOAT, 0, nullptr);
    glEnableClientState(GL_COLOR_ARRAY);

    // Pass the camera's view and projection matrices to the shaders
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(camera.getZoom()), static_cast<float>(WIDTH) / static_cast<float>(HEIGHT), 0.001f, 100.0f);

    glUniformMatrix4fv(glGetUniformLocation(m_particles_program, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(m_particles_program, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projection));

    glDrawArrays(GL_POINTS, 0, pSystem->size());

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glUseProgram(0);
}


void Render::oneStep() {
	++frameId;
	const auto milliseconds = pSystem->step();
	totalTime += milliseconds;
	printf("Frame %d - %2.2f ms, avg time - %2.2f ms/frame (%3.2f FPS)\r", 
		frameId%10000, milliseconds, totalTime / float(frameId), float(frameId)*1000.0f/totalTime);
}
static auto current = SPH;
void Render::keyboardEvent(){
	if(glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS){
		initSPHSystem(SPH);
		frameId = 0;
		totalTime = 0.0f;
		current = SPH;
	}
	else if(glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS){
		initSPHSystem(DFSPH);
		frameId = 0;
		totalTime = 0.0f;
		current = DFSPH;
	}
	else if(glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS){
		initSPHSystem(PBD);
		frameId = 0;
		totalTime = 0.0f;
		current = PBD;
	}
	else if(glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS){
		running = true;
	}
	else if(glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS){
		running = !running;
	}
	else if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS){
		initSPHSystem(current);
		frameId = 0;
		totalTime = 0.0f;		
	}

	else if(glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS){
		oneStep();
	}
	
}


void Render::createContainerMesh() {
    // Container vertices and indices
	const float cuboid_height = 0.75f;
    float containerVertices[] = {
          // Base (position, normal, color)
        -0.5f, 0.0f, -0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.0f, -0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.0f,  0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        -0.5f, 0.0f,  0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f,

        // Left wall (position, normal, color)
        -0.5f, 0.0f, -0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        -0.5f, cuboid_height, -0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        -0.5f, cuboid_height,  0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        -0.5f, 0.0f,  0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,

        // Right wall (position, normal, color)
        0.5f, 0.0f, -0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        0.5f, cuboid_height, -0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        0.5f, cuboid_height,  0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.0f,  0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f,

        // Front wall (position, normal, color)
        -0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f,
        -0.5f, cuboid_height, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f,
        0.5f, cuboid_height, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f,

        // Back wall (position, normal, color)
        -0.5f, 0.0f,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f,
        -0.5f, cuboid_height,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f,
        0.5f, cuboid_height,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f,
        0.5f, 0.0f,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f,
    };


    unsigned int containerIndices[] = {
        // Base
        0, 1, 2,
        0, 2, 3,
        // Left wall
        4, 5, 6,
        4, 6, 7,
        // Right wall
        8, 9, 10,
        8, 10, 11,
        // Front wall
        12, 13, 14,
        12, 14, 15,
        // Back wall
        16, 17, 18,
        16, 18, 19,
    };

    // VAO, VBO, and EBO setup
    glGenVertexArrays(1, &container_vao);
    glGenBuffers(1, &container_vbo);
    glGenBuffers(1, &container_ebo);

    glBindVertexArray(container_vao);

    glBindBuffer(GL_ARRAY_BUFFER, container_vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, container_ebo);

    // Buffer the container data
    glBufferData(GL_ARRAY_BUFFER, sizeof(containerVertices), containerVertices, GL_STATIC_DRAW);
    // glBufferData(GL_ARRAY_BUFFER, containerVertices.size() * sizeof(containerVertices), &containerVertices[0], GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(containerIndices), containerIndices, GL_STATIC_DRAW);
    //  //  glBufferData(GL_ELEMENT_ARRAY_BUFFER, containerIndices.size() * sizeof(unsigned int), &containerIndices[0], GL_STATIC_DRAW);

   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // Unbind VAO, VBO, and EBO
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Render::renderContainer() {
    glUseProgram(containerShaderProgram);

    // Set the projection and view matrices
    float aspect = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
    glm::mat4 projection = glm::perspective(glm::radians(camera.getZoom()), aspect, 0.001f, 100.0f);
    glUniformMatrix4fv(glGetUniformLocation(containerShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    glm::mat4 view = camera.getViewMatrix();
    glUniformMatrix4fv(glGetUniformLocation(containerShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(0.5f, 0.02f, 0.5f)); // Move the container down slightly
    model = glm::scale(model, glm::vec3(0.96f, 1.3f, 0.98f)); // Scale the container
    glUniformMatrix4fv(glGetUniformLocation(containerShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // Render as wireframe
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    //set light direction
    glm::vec3 lightPos(camera.getFront());
   glUniform3fv(glGetUniformLocation(containerShaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));


    glBindVertexArray(container_vao);
    glDrawElements(GL_TRIANGLES, 24, GL_UNSIGNED_INT, 0);//set 30 as 24 to remove the front wall view
    glBindVertexArray(0);
}
