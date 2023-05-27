#pragma once

#include <string>
#include <GL/glew.h>

class ShaderUtils {
public:
    static GLuint loadShader(const std::string& filename, GLenum shaderType);
    static GLuint createShaderProgram(GLuint vertexShader, GLuint fragmentShader);
    static GLuint loadTexture(const char* filename);
};
