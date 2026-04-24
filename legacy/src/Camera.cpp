#include "Camera.h"
#include <GLFW/glfw3.h>
#include <stdio.h>
Camera::Camera(GLFWwindow* window, glm::vec3 position, glm::vec3 up, float yaw, float pitch)
    : window(window), position(position), worldUp(up), yaw(yaw), pitch(pitch),
      movementSpeed(2.5f), mouseSensitivity(0.1f), zoom(60.0f),
      lastX(0.0f), lastY(0.0f), firstMouse(true) {
    updateCameraVectors();
    glfwSetWindowUserPointer(window, this);
    // Set the callback functions
    glfwGetWindowSize(window, &width, &height);
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + front, up);
}

void Camera::update(float deltaTime) {
    processKeyboardInput(deltaTime);
    processMouseMovement();
}

void Camera::processKeyboardInput(float deltaTime) {
    float velocity = movementSpeed * deltaTime / 10.0f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
        velocity *= 5;
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        position += front * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        position -= front * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        position -= right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        position += right * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        position += up * velocity;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS){
        position -= up * velocity;
    }

}

void Camera::processMouseMovement() {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        firstMouse = true;
        return;
    }
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (firstMouse) {
        glfwSetCursorPos(window, ( width/ 2), (height / 2));
        lastX = ( width/ 2);
        lastY = (height / 2);
        firstMouse = false;
    }
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);


    //direct 
    float xOffset = xpos - lastX;
    float yOffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    // smooth
    // double xOffset = xpos - double(width/ 2);
    // double yOffset =double(height/ 2) - ypos;
    // xOffset = xOffset * mouseSensitivity / (double)width;
    // yOffset = yOffset * mouseSensitivity / (double)height;

    yaw += xOffset * mouseSensitivity ;
    pitch += yOffset  * mouseSensitivity;
    //gimbal lock
    // yaw = fmod(yaw, 360.0f);
    // pitch = fmod(pitch, 360.0f);

    if (pitch > 89.0f) {
        pitch = 89.0f;
    }
    if (pitch < -89.0f) {
        pitch = -89.0f;
    }

    updateCameraVectors();
}

void Camera::updateCameraVectors() {
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront);
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));
}

