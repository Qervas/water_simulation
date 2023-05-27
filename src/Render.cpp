#include "Render.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

Render::Render(GLFWwindow* window):camera(window), window(window), cellStart(cellSize.x* cellSize.y* cellSize.z + 1){


	createContainerMesh();
	GLuint vertexShader = ShaderUtils::loadShader("shaders/particles.vert", GL_VERTEX_SHADER);
    GLuint fragmentShader = ShaderUtils::loadShader("shaders/particles.frag", GL_FRAGMENT_SHADER);
    particleShaderProgram = ShaderUtils::createShaderProgram(vertexShader, fragmentShader);
	glBindAttribLocation(particleShaderProgram, particle_attributes::SIZE, "pointSize");
    GLuint containerVertexShader = ShaderUtils::loadShader("shaders/container.vert", GL_VERTEX_SHADER);
    GLuint containerFragmentShader = ShaderUtils::loadShader("shaders/container.frag", GL_FRAGMENT_SHADER);
    containerShaderProgram = ShaderUtils::createShaderProgram(containerVertexShader, containerFragmentShader);
	GLuint surfaceVertexShader = ShaderUtils::loadShader("shaders/surface.vert", GL_VERTEX_SHADER);
    GLuint surfacaeFragmentShader = ShaderUtils::loadShader("shaders/surface.frag", GL_FRAGMENT_SHADER);
    surfaceShaderProgram = ShaderUtils::createShaderProgram(surfaceVertexShader, surfacaeFragmentShader);

	container_texture = ShaderUtils::loadTexture("texture/texture.jpg", GL_RGB);
	container_texture_specular_map = ShaderUtils::loadTexture("texture/specular_map.jpg", GL_RED);


    // Clean up shader resources
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(containerVertexShader);
    glDeleteShader(containerFragmentShader);
	glDeleteShader(surfaceVertexShader);
	glDeleteShader(surfacaeFragmentShader);

}

Render::~Render(){
	deleteVBO(&particles_vbo);
	deleteVBO(&particles_color_vbo);
	pSystem = nullptr;
	checkCudaErrors(cudaDeviceReset());
	glDeleteProgram(surfaceShaderProgram);
	glDeleteProgram(containerShaderProgram);
	glDeleteVertexArrays(1, &container_vao);
    glDeleteBuffers(1, &container_vbo);
    glDeleteBuffers(1, &container_ebo);
	glDeleteVertexArrays(1, &surface_vao);
	glDeleteBuffers(2, surface_vbo);
    glDeleteBuffers(1, &surface_ebo);
	
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
	// renderSurface();
	keyboardEvent();
}

void Render::init(){
	initSPHSystem();
	createVBO(&particles_vbo, sizeof(float3) * pSystem->size());
	createVBO(&particles_color_vbo, sizeof(float3) * pSystem->size());
	
	// marchingCubes = MarchingCubes(pSystem->getFluids(), isolevel, pSystem->getFluids()->getDensityPtr(),
	// 							pSystem->getFluids()->getParticle2Cell());

	//init surface
	glGenVertexArrays(1, &surface_vao);
	glGenBuffers(2, surface_vbo);
	glGenBuffers(1, &surface_ebo);


}

void Render::initSPHSystem() {
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
	std::shared_ptr<BasicSPHSolver> pSolver;
	pSolver = std::make_shared<BasicSPHSolver>(fluidParticles->size());
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
	checkCudaErrors(cudaGLRegisterBufferObject(*vbo));

}

void Render::deleteVBO(GLuint* vbo) {

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
}


void Render::renderParticles() {

	glUseProgram(particleShaderProgram);
	glUniform1f(glGetUniformLocation(particleShaderProgram, "pointScale"), WIDTH / tanf(m_fov*0.5f*float(PI) / 180.0f));
	glUniform1f(glGetUniformLocation(particleShaderProgram, "pointRadius"), particle_radius);
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
    // map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    float3 *cptr;
	
	checkCudaErrors(cudaGLMapBufferObject((void**)&dptr, particles_vbo));
	checkCudaErrors(cudaGLMapBufferObject((void**)&cptr, particles_color_vbo));


    // calculate the dots' position and color
    generate_dots(dptr, cptr, pSystem->getFluids());
	
    // unmap buffer object
	checkCudaErrors(cudaGLUnmapBufferObject(particles_vbo));
	checkCudaErrors(cudaGLUnmapBufferObject(particles_color_vbo));

    // Pass the camera's view and projection matrices to the shaders
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(camera.getZoom()), static_cast<float>(WIDTH) / static_cast<float>(HEIGHT), 0.001f, 100.0f);
    glUniformMatrix4fv(glGetUniformLocation(particleShaderProgram, "viewMatrix"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(particleShaderProgram, "projectionMatrix"), 1, GL_FALSE, glm::value_ptr(projection));

	//light position
	glm::vec3 lightPos = glm::vec3(view * glm::vec4(camera.getPosition(), 1.0f));
	glUniform3fv(glGetUniformLocation(particleShaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));

	//spot light
	glm::vec3 spotDirection = glm::vec3(view * glm::vec4(camera.getFront(), 0.0f));
	float cutOff = 20.0f; 
	float outerCutOff = 25.5f;
	glUniform3fv(glGetUniformLocation(particleShaderProgram, "spotDir"), 1, glm::value_ptr(spotDirection));
	glUniform1f(glGetUniformLocation(particleShaderProgram, "spotCutOff"), cutOff);
	glUniform1f(glGetUniformLocation(particleShaderProgram, "spotOuterCutOff"), outerCutOff);


	glBindBuffer(GL_ARRAY_BUFFER, particles_vbo);
	GLuint posAttribLocation = 0; 
	glVertexAttribPointer(posAttribLocation, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(posAttribLocation);
	GLuint colorAttribLocation = 1;
	glVertexAttribPointer(colorAttribLocation, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(colorAttribLocation);

    glDrawArrays(GL_POINTS, 0, pSystem->size());

    glDisableClientState(GL_VERTEX_ARRAY);
    glUseProgram(0);
}

void Render::renderSurface(){
	// glUseProgram(surfaceShaderProgram);
	


	// marchingCubes.generateMesh_CUDA();
	// vertices = marchingCubes.getVertices();
	// normals = marchingCubes.getNormals();
	// indices = marchingCubes.getIndices();
	// glBindVertexArray(surface_vao);

    // glBindBuffer(GL_ARRAY_BUFFER, surface_vbo[0]);
    // glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float3), &vertices[0], GL_DYNAMIC_DRAW);
    // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    // glEnableVertexAttribArray(0);

    // glBindBuffer(GL_ARRAY_BUFFER, surface_vbo[1]);
    // glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float3), &normals[0], GL_DYNAMIC_DRAW);
    // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    // glEnableVertexAttribArray(1);

    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, surface_ebo);
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint3), &indices[0], GL_DYNAMIC_DRAW);

    // glDrawElements(GL_TRIANGLES, indices.size() * 3, GL_UNSIGNED_INT, 0);

    // glBindVertexArray(0);
}

void Render::oneStep() {
	pSystem->step();
}

void Render::keyboardEvent(){
	if(glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS){
		running = true;
	}
	else if(glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS){
		running = !running;
	}
	else if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS){
		initSPHSystem();
		running = false;
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
	float repeat = 2.0f;//2.0
	float length = 1.0f;//1.5
    float containerVertices[] = {
          // Base (position, normal, color, texcoord)
        -0.5f, 0.0f, -0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, //a
        0.5f, 0.0f, -0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f, //b
        0.5f, 0.0f,  0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, repeat * length, //c
        -0.5f, 0.0f,  0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat * length, //d

        // Left wall (position, normal, color)
        -0.5f, 0.0f, -0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, //a
        -0.5f, cuboid_height, -0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,
        -0.5f, cuboid_height,  0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, repeat * length,
        -0.5f, 0.0f,  0.5f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat * length,

        // Right wall (position, normal, color)
        0.5f, 0.0f, -0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,
        0.5f, cuboid_height, -0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
        0.5f, cuboid_height,  0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat * length,
        0.5f, 0.0f,  0.5f, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, repeat * length,

        // Front wall (position, normal, color)
        -0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 
        -0.5f, cuboid_height, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat,
        0.5f, cuboid_height, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, repeat, repeat,
        0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,

        // Back wall (position, normal, color)
        -0.5f, 0.0f,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat,
        -0.5f, cuboid_height,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
        0.5f, cuboid_height,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,
        0.5f, 0.0f,  0.5f, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, repeat, repeat
    };

//todo: ambient light of particles



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
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(containerIndices), containerIndices, GL_STATIC_DRAW);

   	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(float), (void*)(9 * sizeof(float)));
	glEnableVertexAttribArray(3);



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
    model = glm::translate(model, glm::vec3(0.5f, 0.011f, 1.0f)); // Move the container slightly
    model = glm::scale(model, glm::vec3(1.0f, 1.3f, 1.97f)); // Scale the container
    glUniformMatrix4fv(glGetUniformLocation(containerShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    //set light direction
    glm::vec3 lightPos(camera.getPosition());
   	glUniform3fv(glGetUniformLocation(containerShaderProgram, "lightPos"), 1, glm::value_ptr(lightPos));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, container_texture);
	glUniform1i(glGetUniformLocation(containerShaderProgram, "texture_diffuse"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, container_texture_specular_map);
	glUniform1i(glGetUniformLocation(containerShaderProgram, "texture_specular_map"), 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindVertexArray(container_vao);
    glDrawElements(GL_TRIANGLES, 30, GL_UNSIGNED_INT, 0);//set 30 as 24 to remove the front wall view
    glBindVertexArray(0);
}

