#version 330 core

uniform float pointRadius;  // point size in world space
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform float pointScale;

in vec4 vertex;
in vec4 color;

out vec4 fs_Color;
out float fs_pointSize;

out vec3 fs_PosEye;     // center of the viewpoint space
out mat4 u_Persp;

void main(void) {

    vec3 posEye = (viewMatrix * vec4(vertex.xyz, 1.0f)).xyz;
    float dist = length(posEye);

    gl_PointSize = pointRadius * pointScale / dist;

    fs_PosEye = posEye;

    fs_Color = color;

    u_Persp = projectionMatrix;

    gl_Position = projectionMatrix * viewMatrix * vec4(vertex.xyz, 1.0f);
    fs_pointSize = vertex.w;
}
