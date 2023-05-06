#version 330 core
in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;

out vec4 FragColor;

void main() {
    // Simple directional light
    vec3 norm = normalize(Normal);
    vec3 light = normalize(-lightPos);
    float diff = max(dot(norm, light), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

    // Ambient lighting
    vec3 ambient = 0.2 * vec3(1.0, 1.0, 1.0);

    FragColor = vec4(ambient + diffuse, 0.5); // No transparency
}