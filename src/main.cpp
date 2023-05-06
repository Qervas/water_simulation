

#include <iostream>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <GL/freeglut.h>
#include "Render.h"

int main() {

	if (!glfwInit()) {
		return -1;
	}

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Water Simulation", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return -1;
	}

	Render render(window);
	render.init();

	float lastFrame = 0.0f;
	while (!glfwWindowShouldClose(window)) {
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)break;
		// Process input and update camera
		float currentFrame = glfwGetTime();
		float deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		render.render(deltaTime);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	cudaDeviceSynchronize();
	glfwTerminate();

	return 0;
}