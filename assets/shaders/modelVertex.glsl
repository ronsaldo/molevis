#line 2

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexcoord;

void main()
{
    vec3 viewPosition = (CameraState.viewMatrix*(ModelState.modelMatrix*vec4(inPosition, 1.0))).xyz;
    vec3 viewNormal = (vec4(inNormal, 0.0)*ModelState.inverseModelMatrix*CameraState.inverseViewMatrix).xyz;

    outPosition = viewPosition;
    outNormal = viewNormal;
    outTexcoord = inTexcoord;
    gl_Position = CameraState.projectionMatrix*vec4(viewPosition, 1.0);

}