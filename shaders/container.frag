#version 330 core

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoords;

uniform vec3 lightPos;
uniform sampler2D texture_diffuse;
uniform sampler2D texture_specular_map;


out vec4 FragColor;

void main() {

    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
    // Simple directional light
    vec3 norm = normalize(Normal);

     vec3 lightDir = normalize(lightPos - FragPos); // Calculate direction from the fragment to the light source
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

    // Specular lighting
    vec3 viewDir = normalize(lightPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    vec3 spec = vec3(texture(texture_specular_map, TexCoords)); // Sample the specular map in the fragment's UV coordinates
    float specStrength = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);
    vec3 specular = specStrength * spec;  


    vec4 texColor = texture(texture_diffuse, TexCoords);

    FragColor = texColor * vec4(ambient + diffuse + vec3(specular.r), 1.0);

}