#line 2

layout(location = 0) in vec3 inPosition;

void main()
{
    vec3 viewPosition = (CameraState.viewMatrix*(ModelState.modelMatrix*vec4(inPosition, 1.0))).xyz;
    gl_Position = CameraState.projectionMatrix*vec4(viewPosition, 1.0);

}