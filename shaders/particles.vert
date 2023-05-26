#version 330 core

layout(location = 1) in vec4 color;

uniform float pointRadius;  // point size in world space
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform float pointScale;

in vec4 vertex;

out vec4 fs_Color; //fragment shader color
out float fs_pointSize; //fragment shader point size

out vec3 fs_PosEye;     // center of the viewpoint space
out mat4 u_Persp;       // perspective matrix
out vec3 vs_WorldPos;
void main(void) {

    vec3 posEye = (viewMatrix * vec4(vertex.xyz, 1.0f)).xyz;
    float dist = length(posEye);

    gl_PointSize = pointRadius * pointScale / dist;

    fs_PosEye = posEye;

    fs_Color = color;

    u_Persp = projectionMatrix;

    gl_Position = projectionMatrix * viewMatrix * vec4(vertex.xyz, 1.0f);
    fs_pointSize = vertex.w;

    vs_WorldPos = vertex.xyz;

}

// #version 330 core

// layout(location = 1) in vec4 color;

// uniform float pointRadius;  // point size in world space
// uniform mat4 viewMatrix;
// uniform mat4 projectionMatrix;
// uniform float pointScale;

// in vec4 vertex;

// out vec4 fs_Color;
// out vec3 fs_PosEye;
// out mat4 u_Persp;

// void main(void) {
//     vec3 posEye = (viewMatrix * vec4(vertex.xyz, 1.0f)).xyz;
//     float dist = length(posEye);

//     gl_PointSize = pointRadius * pointScale / dist;
    
//     fs_PosEye = posEye;
//     fs_Color = color;
//     u_Persp = projectionMatrix;

//     gl_Position = projectionMatrix * viewMatrix * vec4(vertex.xyz, 1.0f);
// }
