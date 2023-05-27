#include "ShaderUtils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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

GLuint ShaderUtils::loadTexture(const char* filename, GLenum format){
    int width, height, numChannels;
    unsigned char* img = stbi_load(filename, &width, &height, &numChannels, 0);
    if( img == nullptr){
        std::cerr << "failed to load texture: " << filename << std::endl;
        return 0;
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, img);

    stbi_image_free(img);
    return texture;
}
