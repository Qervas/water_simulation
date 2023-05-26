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

void main(void) {

    vec3 fs_PosEye = vec3(viewMatrix * vec4(vs_WorldPos, 1.0));


    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside sphere
    N.z = sqrt(1.0 - mag);

    // calculate depth
    vec4 pixelPos = vec4(fs_PosEye + normalize(N) * fs_pointSize, 1.0f);
    vec4 clipSpacePos = u_Persp * pixelPos;

    // calculate light direction
    vec3 lightDir = normalize(lightPos - fs_PosEye);

    // calculate diffuse light intensity
    float diff = max(dot(N, lightDir), 0.0);

    // calculate spotlight intensity
    // float theta = abs(dot(lightDir, normalize(spotDir))); // angle between light direction and spotlight direction
    float theta = dot(lightDir, normalize(-spotDir)); 

    // check if point is inside the spotlight
    if(theta > cos(radians(spotCutOff))) {
        float intensity = diff * (1.0 - smoothstep(cos(radians(spotCutOff)), cos(radians(spotOuterCutOff)), theta));
        vec3 colorWithLight = intensity * fs_Color.rgb;
        FragColor = vec4(exp(-mag * mag) * colorWithLight, 1.0f);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // point is outside the spotlight, set to black or any other color
    }
}

// #version 330 core

// in vec4 fs_Color;
// in vec3 fs_PosEye;
// in mat4 u_Persp;

// out vec4 FragColor;

// uniform vec3 lightPos;
// uniform vec3 spotDir;
// uniform vec3 camPos;

// void main(void) {
//     vec3 N;
//     N.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
//     float mag = dot(N.xy, N.xy);
//     if (mag > 1.0) discard;
//     N.z = sqrt(1.0 - mag);

//     vec3 pixelPos = vec3(u_Persp * vec4(fs_PosEye + N, 1.0));
//     vec3 lightVec = lightPos - pixelPos;

//     float outerCone = 10.90f;
//     float innerCone = 15.95f;

//     float ambient = 0.20f;
//     vec3 lightDirection = normalize(lightVec);
//     float diffuse = max(dot(N, lightDirection), 0.0f);

//     vec3 viewDirection = normalize(camPos - pixelPos);
//     vec3 reflectionDirection = reflect(-lightDirection, N);
//     float specularLight = 0.50f;
//     float specAmount = pow(max(dot(viewDirection, reflectionDirection), 0.0f), 16);
//     float specular = specAmount * specularLight;

//     float angle = dot(spotDir, -lightDirection);
//     float inten = clamp((angle - outerCone) / (innerCone - outerCone), 0.0f, 1.0f);

//     FragColor = (fs_Color * (diffuse * inten + ambient) + specular * inten) * vec4(1.0f, 1.0f, 1.0f, 1.0f);
// }
