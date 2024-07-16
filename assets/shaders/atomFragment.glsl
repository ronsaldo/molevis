#version 450

layout(location=0) flat in vec4 inColor;
layout(location=1) in vec3 inViewPosition;

layout(location=0) out vec4 fragColor;

void main()
{
    // Premultiplied alpha
    vec4 color = vec4(inColor.rgb * inColor.a, inColor.a);

    fragColor = color;
}
