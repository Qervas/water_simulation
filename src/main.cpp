

#include <iostream>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <GL/freeglut.h>
#include "Render.h"



// void mouseFunc(const int button, const int state, const int x, const int y) {
// 	if (GLUT_DOWN == state) {
// 		if (GLUT_LEFT_BUTTON == button) {
// 			mouse_left_down = true;
// 			mousePos[0] = x;
// 			mousePos[1] = y;
// 		}
// 		else if (GLUT_RIGHT_BUTTON == button) {}
// 	}
// 	else {
// 		mouse_left_down = false;
// 	}
// 	return;
// }

// void motionFunc(const int x, const int y) {
// 	int dx, dy;
// 	if (-1 == mousePos[0] && -1 == mousePos[1])
// 	{
// 		mousePos[0] = x;
// 		mousePos[1] = y;
// 		dx = dy = 0;
// 	}
// 	else
// 	{
// 		dx = x - mousePos[0];
// 		dy = y - mousePos[1];
// 	}
// 	if (mouse_left_down)
// 	{
// 		rot[0] += (float(dy) * 180.0f) / 720.0f;
// 		rot[1] += (float(dx) * 180.0f) / 720.0f;
// 	}

// 	mousePos[0] = x;
// 	mousePos[1] = y;

// 	glutPostRedisplay();
// 	return;
// }

// void keyboardFunc(const unsigned char key, const int x, const int y) {
// 	switch (key) {
// 	case '1':
// 		initSPHSystem(fluid_solver::SPH);
// 		frameId = 0;
// 		totalTime = 0.0f;
// 		break;
// 	case '2':
// 		initSPHSystem(fluid_solver::DFSPH);
// 		frameId = 0;
// 		totalTime = 0.0f;
// 		break;
// 	case '3':
// 		initSPHSystem(fluid_solver::PBD);
// 		frameId = 0;
// 		totalTime = 0.0f;
// 		break;
// 	case 'x':
// 		running = !running;
// 		break;
// 	case ',':
// 		zoom *= 1.2f;
// 		break;
// 	case '.':
// 		zoom /= 1.2f;
// 		break;
// 	case 'q':
// 	case 'Q':
// 		onClose();
// 		break;
// 	case 'r':
// 	case 'R':
// 		rot[0] = rot[1] = 0;
// 		zoom = 0.3f;
// 		break;
// 	case 'n':
// 	case 'N':
// 		void oneStep();
// 		oneStep();
// 		break;
// 	default:
// 		;
// 	}
// }







int main() {//int argc, char* argv[]

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

		glfwTerminate();
		cudaDeviceSynchronize();

		// std::cout << "Instructions\n";
		// std::cout << "The color indicates the density of a particle.\nMagenta means higher density, navy means lesser density.\n";
		// std::cout << "Controls\n";
		// std::cout << "Space - Start/Pause\n";
		// std::cout << "Key N - One Step Forward\n";
		// std::cout << "Key Q - Quit\n";
		// std::cout << "Key 1 - Restart Simulation Using SPH Solver\n";
		// std::cout << "Key 2 - Restart Simulation Using DFSPH Solver\n";
		// std::cout << "Key 3 - Restart Simulation Using PBD Solver\n";
		// std::cout << "Key R - Reset Viewpoint\n";
		// std::cout << "Key , - Zoom In\n";
		// std::cout << "Key . - Zoom Out\n";
		// std::cout << "Mouse Drag - Change Viewpoint\n\n";
		////////////////////
		// glutMainLoop();
	// }
	// catch (...) {
	// 	std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__ << "\n";
	// }
	return 0;
}