#version 330 core

in vec4 fs_Color;
in float fs_pointSize;
in mat4 u_Persp;
out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 spotDir; // the direction in which the spotlight is pointing
uniform float spotCutOff; // the cutoff angle for the spotlight
uniform float spotOuterCutOff; // the outer cutoff angle for the spotlight, for soft edges

in vec3 vs_WorldPos;
uniform mat4 viewMatrix;
uniform vec3 viewPos; // the viewer's position

void main(void) {

    vec3 FragPos = vec3(viewMatrix * vec4(vs_WorldPos, 1.0));

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside sphere
    N.z = sqrt(1.0 - mag);

    // calculate depth
    vec4 pixelPos = vec4(FragPos + normalize(N) * fs_pointSize, 1.0f);
    vec4 clipSpacePos = u_Persp * pixelPos;

    // calculate light direction
    vec3 lightDir = normalize(lightPos - FragPos);

    // calculate diffuse light intensity
    float diff = 0.7 * max(dot(N, lightDir), 0.0);

    // calculate specular lighting
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, N); 
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = spec * fs_Color.rgb;

    // calculate spotlight intensity
    float theta = dot(lightDir, normalize(-spotDir)); 

    // calculate ambient light
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * fs_Color.rgb;
    
    vec3 colorWithLight;
    // check if point is inside the spotlight
    if(theta > cos(radians(spotOuterCutOff))) {
        float intensity = diff * (1.0 - smoothstep(cos(radians(spotCutOff)), cos(radians(spotOuterCutOff)), theta));
        colorWithLight = ambient + (intensity + specular) * fs_Color.rgb;
    } else {
        colorWithLight = ambient; // point is outside the spotlight, apply only ambient light
    }
    
    // Using diff for the alpha value, but this could be any function or value that suits your needs
    float alpha = 0.75;
    
    FragColor = vec4(exp(-mag * mag) * colorWithLight, alpha);
}
