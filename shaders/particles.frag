// #version 330 core

// in vec4 fs_Color;
// in float fs_pointSize;

// in vec3 fs_PosEye;
// in mat4 u_Persp;

// out vec4 FragColor;

// void main(void){
//     // calculate normal from texture coordinates
//     vec3 N;

//     N.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);

//     float mag = dot(N.xy, N.xy);
//     if (mag > 1.0) discard;   // kill pixels outside sphere
//     N.z = sqrt(1.0 - mag);

//     // calculate depth
//     vec4 pixelPos = vec4(fs_PosEye + normalize(N) * fs_pointSize, 1.0f);
//     vec4 clipSpacePos = u_Persp * pixelPos;

//     FragColor = vec4(exp(-mag * mag) * fs_Color.rgb, 1.0f);
// }


#version 330 core

in vec4 fs_Color;
in float fs_pointSize;

in vec3 fs_PosEye;
in mat4 u_Persp;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 spotDir; // the direction in which the spotlight is pointing
uniform float spotCutOff; // the cutoff angle for the spotlight
uniform float spotOuterCutOff; // the outer cutoff angle for the spotlight, for soft edges

void main(void) {

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
    float theta = dot(lightDir, normalize(-spotDir)); // angle between light direction and spotlight direction

    // check if point is inside the spotlight
    if(theta > cos(radians(spotCutOff))) {
        float intensity = diff * (1.0 - smoothstep(cos(radians(spotCutOff)), cos(radians(spotOuterCutOff)), theta));
        vec3 colorWithLight = intensity * fs_Color.rgb;
        FragColor = vec4(exp(-mag * mag) * colorWithLight, 1.0f);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // point is outside the spotlight, set to black or any other color
    }
}
