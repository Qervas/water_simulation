#version 330 core

in vec4 fs_Color;
in float fs_pointSize;

in vec3 fs_PosEye;
in mat4 u_Persp;

out vec4 FragColor;

void main(void)
{
    // calculate normal from texture coordinates
    vec3 N;

    N.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0 - mag);

    // calculate depth
    vec4 pixelPos = vec4(fs_PosEye + normalize(N) * fs_pointSize, 1.0f);
    vec4 clipSpacePos = u_Persp * pixelPos;

    FragColor = vec4(exp(-mag * mag) * fs_Color.rgb, 1.0f);
}
