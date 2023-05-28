#version 330 core

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoords;

uniform vec3 lightPos;
uniform vec3 spotDir;
uniform float spotCutOff;
uniform float spotOuterCutOff;

uniform sampler2D texture_diffuse;
uniform sampler2D texture_specular_map;

out vec4 FragColor;

void main() {

    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);

    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir= normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

    float theta = dot(lightDir, normalize(-spotDir));
    float intensity;
    if(theta > cos(radians(spotOuterCutOff))){
        intensity = diff * (1.0 - smoothstep(cos(radians(spotCutOff)), cos(radians(spotOuterCutOff)), theta));
    }else{
        intensity = 0.1 * diff;
    }



    // Specular lighting
    vec3 viewDir = normalize(lightPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    vec3 spec = vec3(texture(texture_specular_map, TexCoords));
    float specStrength = pow(max(dot(viewDir, reflectDir), 0.0), 8.0);
    vec3 specular = specStrength * spec;

    vec3 result = (ambient + diffuse + specular.r * 0.25) * intensity;
    vec4 texColor = texture(texture_diffuse, TexCoords);
    
    FragColor = vec4(result, 1.0) * texColor;
}
