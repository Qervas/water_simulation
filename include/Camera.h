#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    Camera(GLFWwindow* window,
           glm::vec3 position = glm::vec3(0.7f, 0.7f, 2.0f),
           glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
           float yaw = -90.0f,
           float pitch = 0.0f);

    glm::mat4 getViewMatrix() const;
    void update(float deltaTime);
    float getZoom() const {return zoom;}
    const glm::vec3& getPosition() const { return position; }
    const glm::vec3& getFront() const {return front;}

private:
    GLFWwindow* window;
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    float yaw;
    float pitch;
    float movementSpeed;
    float mouseSensitivity;
    float zoom;
    float lastX, lastY;
    bool firstMouse;

    int width, height;

    void processKeyboardInput(float deltaTime);
    void processMouseMovement();
    void processMouseScroll();
    void updateCameraVectors();

    static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);

};