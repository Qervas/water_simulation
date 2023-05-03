#include "ShaderUtils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

GLuint ShaderUtils::loadShader(const std::string& filename, GLenum shaderType) {
    std::ifstream shaderFile(filename);

    if (!shaderFile.is_open()) {
        std::cerr << "Error: Could not open shader file " << filename << std::endl;
        return 0;
    }

    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    std::string shaderSource = shaderStream.str();

    const char* shaderSourceCStr = shaderSource.c_str();

    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSourceCStr, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        GLint maxLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<char> errorLog(maxLength);
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]);

        std::cerr << "Error: Shader compilation failed for " << filename << ": " << &errorLog[0] << std::endl;

        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint ShaderUtils::createShaderProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        GLint maxLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<char> errorLog(maxLength);
        glGetProgramInfoLog(program, maxLength, &maxLength, &errorLog[0]);

        std::cerr << "Error: Shader program linking failed: " << &errorLog[0] << std::endl;

        glDeleteProgram(program);
        return 0;
    }

    return program;
}
