#include "Render.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

Render::Render(GLFWwindow* window):camera(window), window(window), cellStart(cellSize.x* cellSize.y* cellSize.z + 1){


	createContainerMesh();
	createSkyboxMesh();
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

	container_texture = ShaderUtils::load2DTexture("texture/container.jpg", GL_RGB);
	container_texture_specular_map = ShaderUtils::load2DTexture("texture/container_specular_map.jpg", GL_RED);

	GLuint skyboxVertexShader = ShaderUtils::loadShader("shaders/skybox.vert", GL_VERTEX_SHADER);
	GLuint skyboxFragmentShader = ShaderUtils::loadShader("shaders/skybox.frag", GL_FRAGMENT_SHADER);
	skyboxShaderProgram = ShaderUtils::createShaderProgram(skyboxVertexShader, skyboxFragmentShader);
	
	std::string skybox_filename[6];
	for(int i = 0; i < 6; i++){
		skybox_filename[i] = "texture/face"+ std::to_string(i) +".png";
	}
	skybox_texture = ShaderUtils::loadSkyboxTexture(skybox_filename);

	

    // Clean up shader resources
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(containerVertexShader);
    glDeleteShader(containerFragmentShader);
	glDeleteShader(surfaceVertexShader);
	glDeleteShader(surfacaeFragmentShader);
	glDeleteShader(skyboxVertexShader);
	glDeleteShader(skyboxFragmentShader);

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
	glDeleteVertexArrays(1, &skybox_vao);
	glDeleteBuffers(1, &skybox_vbo);
	glDeleteBuffers(1, &skybox_ebo);
	
}
void Render::render(float deltaTime){
	glClearColor(0.1, 0.1f ,0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, WIDTH, HEIGHT);
	if (running) {
		oneStep();
	}
	camera.update(deltaTime);
	renderParticles();
	renderSkybox();
	renderContainer();
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
				auto x = make_float3(0.3f + sphSpacing * j,//0.27f
					0.1f + sphSpacing * i,//0.10f
					0.3f + sphSpacing * k);//0.27f
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
	glUniform3fv(glGetUniformLocation(particleShaderProgram, "spotDir"), 1, glm::value_ptr(spotDirection));
	glUniform1f(glGetUniformLocation(particleShaderProgram, "spotCutOff"), spotCutOff);
	glUniform1f(glGetUniformLocation(particleShaderProgram, "spotOuterCutOff"), spotOuterCutOff);


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
	const float cuboid_width = 0.5f;
	const float cuboid_depth = 0.5f;
	float repeat = 2.0f;//2.0
	float factor = 1.5f;//1.5
    float containerVertices[] = {
          // Base (position, normal, color, texcoord)
        -cuboid_width, 0.0f, -cuboid_depth, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, //a
        cuboid_width, 0.0f, -cuboid_depth, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f, //b
        cuboid_width, 0.0f,  cuboid_depth, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, repeat * factor, //c
        -cuboid_width, 0.0f,  cuboid_depth, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat * factor, //d

        // Left wall (position, normal, color)
        -cuboid_width, 0.0f, -cuboid_depth, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, //a
        -cuboid_width, cuboid_height, -cuboid_depth, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,
        -cuboid_width, cuboid_height,  cuboid_depth, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, repeat * factor,
        -cuboid_width, 0.0f,  cuboid_depth, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat * factor,

        // Right wall (position, normal, color)
        cuboid_width, 0.0f, -cuboid_depth, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,
        cuboid_width, cuboid_height, -cuboid_depth, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
        cuboid_width, cuboid_height,  cuboid_depth, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat * factor,
        cuboid_width, 0.0f,  cuboid_depth, -1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat, repeat * factor,

        // Front wall (position, normal, color)
        -cuboid_width, 0.0f, -cuboid_depth, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 
        -cuboid_width, cuboid_height, -cuboid_depth, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat,
        cuboid_width, cuboid_height, -cuboid_depth, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, repeat, repeat,
        cuboid_width, 0.0f, -cuboid_depth, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,

        // Back wall (position, normal, color)
        -cuboid_width, 0.0f,  cuboid_depth, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat,
        -cuboid_width, cuboid_height,  cuboid_depth, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
        cuboid_width, cuboid_height,  cuboid_depth, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, repeat, 0.0f,
        cuboid_width, 0.0f,  cuboid_depth, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 0.5f, repeat, repeat,

		// Top (position, normal, color)
		-cuboid_width, cuboid_height, -cuboid_depth, 0.0f, -1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, repeat * factor,
		-cuboid_width, cuboid_height,  cuboid_depth, 0.0f, -1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.0f, 0.0f,
		cuboid_width, cuboid_height,  cuboid_depth, 0.0f, -1.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat * factor, 0.0f,
		cuboid_width, cuboid_height, -cuboid_depth, 0.0f, -1.0f, 0.0f, 0.5f, 0.5f, 0.5f, repeat * factor, repeat * factor
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
		// Top
		20, 21, 22,
		20, 22, 23
    };

    // VAO, VBO, and EBO setup
    glGenVertexArrays(1, &container_vao);
    glGenBuffers(1, &container_vbo);
    glGenBuffers(1, &container_ebo);

	// Bind the VAO, VBO, and EBO
    glBindVertexArray(container_vao);
    glBindBuffer(GL_ARRAY_BUFFER, container_vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, container_ebo);

    // Buffer the container data
    glBufferData(GL_ARRAY_BUFFER, sizeof(containerVertices), containerVertices, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(containerIndices), containerIndices, GL_STATIC_DRAW);

	// Set the vertex attribute pointers
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
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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
	
	glm::vec3 spotDir(camera.getFront());
	glUniform3fv(glGetUniformLocation(containerShaderProgram, "spotDir"), 1, glm::value_ptr(spotDir));


	glUniform1f(glGetUniformLocation(containerShaderProgram, "spotCutOff"), spotCutOff);
	glUniform1f(glGetUniformLocation(containerShaderProgram, "spotOuterCutOff"), spotOuterCutOff);

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

void Render::createSkyboxMesh(){
	float skyboxVertices[] = 	{
		//   Coordinates
		-1.0f, -1.0f,  1.0f,//       7--------6
		1.0f, -1.0f,  1.0f,//       /|       /|
		1.0f, -1.0f, -1.0f,//      4--------5 |
		-1.0f, -1.0f, -1.0f,//     | |      | |
		-1.0f,  1.0f,  1.0f,//     | 3------|-2
		1.0f,  1.0f,  1.0f,//      |/       |/
		1.0f,  1.0f, -1.0f,//      0--------1
		-1.0f,  1.0f, -1.0f
	};

	unsigned int skyboxIndices[] = 	{
		// Right
		1, 2, 6,
		6, 5, 1,
		// Left
		0, 4, 7,
		7, 3, 0,
		// Top
		4, 5, 6,
		6, 7, 4,
		// Bottom
		0, 3, 2,
		2, 1, 0,
		// Back
		0, 1, 5,
		5, 4, 0,
		// Front
		3, 7, 6,
		6, 2, 3
	};



	// Create VAO, VBO, and EBO
	glGenVertexArrays(1, &skybox_vao);
	glGenBuffers(1, &skybox_vbo);
	glGenBuffers(1, &skybox_ebo);

	// Bind VAO, VBO, and EBO
	glBindVertexArray(skybox_vao);
	glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, skybox_ebo);

	// Copy our vertices array in a buffer for OpenGL to use
	glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyboxIndices), skyboxIndices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (void*)0);
	glEnableVertexAttribArray(0);

	// Unbind VAO, VBO, and EBO
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Render::renderSkybox(){
    // Use the skybox shader program
	glUseProgram(skyboxShaderProgram);
	
    // Depth buffer setup for skybox
	glDepthFunc(GL_LEQUAL); 
	
    // Set the view matrix (remove translation)
	glm::mat4 view = glm::mat4(glm::mat3(camera.getViewMatrix()));
	glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));

    // Set the projection matrix
	glm::mat4 projection = glm::perspective(glm::radians(camera.getZoom()), (float)WIDTH/(float)HEIGHT, 0.01f, 100.0f);
	glUniformMatrix4fv(glGetUniformLocation(skyboxShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Bind VAO
	glBindVertexArray(skybox_vao);

    // Bind the skybox texture
	glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(skyboxShaderProgram, "skybox"), 0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, skybox_texture);

    // Draw the skybox
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    // Unbind VAO
	glBindVertexArray(0);
	
    // Set the depth function back to default
	glDepthFunc(GL_LESS); 

    // Unbind shader program
	glUseProgram(0);
}
