#line 2

layout(location = 0) flat out uint outBondIndex;
layout(location = 1) out vec3 outViewPosition;

const vec2 quadVertices[4] = vec2[4](
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0)
);

void main()
{
    AtomBondDesc desc = AtomBondDescriptionBuffer[gl_InstanceIndex];
    vec3 firstAtomWorldPosition = (CameraState.atomModelMatrix * vec4(AtomRenderingStateBuffer[desc.firstAtomIndex].position, 1.0)).xyz;
    vec3 firstAtomWorldRadius = (CameraState.atomModelMatrix * (
        vec4(AtomRenderingStateBuffer[desc.firstAtomIndex].position, 1.0)
        + vec4(desc.thickness, 0.0, 0.0, 1.0))).xyz;
    vec3 secondAtomWorldPosition = (CameraState.atomModelMatrix * vec4(AtomRenderingStateBuffer[desc.secondAtomIndex].position, 1.0)).xyz;

    float radius = length(firstAtomWorldRadius - firstAtomWorldPosition);
    vec3 firstAtomViewPosition = (CameraState.viewMatrix * vec4(firstAtomWorldPosition, 1.0)).xyz;
    vec3 secondAtomViewPosition = (CameraState.viewMatrix * vec4(secondAtomWorldPosition, 1.0)).xyz;

    vec2 lineTangent = normalize(secondAtomViewPosition.xy - firstAtomViewPosition.xy);
    vec2 lineBitangent = vec2(-lineTangent.y, lineTangent.x);

    vec3 firstAtomWidthStart = firstAtomViewPosition - vec3(lineBitangent*radius, 0.0);
    vec3 firstAtomWidthEnd = firstAtomViewPosition + vec3(lineBitangent*radius, 0.0);

    vec3 secondAtomWidthStart = secondAtomViewPosition - vec3(lineBitangent*radius, 0.0);
    vec3 secondAtomWidthEnd = secondAtomViewPosition + vec3(lineBitangent*radius, 0.0);

    // Pass the instance index
    outBondIndex = gl_InstanceIndex;

    // Compute the quad view position,
    vec2 quadCoord = quadVertices[gl_VertexIndex];
    vec3 firstAtomWidth = mix(firstAtomWidthStart, firstAtomWidthEnd, quadCoord.y);
    vec3 secondAtomWidth = mix(secondAtomWidthStart, secondAtomWidthEnd, quadCoord.y);
    vec3 viewPosition = mix(firstAtomWidth, secondAtomWidth, quadCoord.x);

    outViewPosition = viewPosition;

    // Perspective projection.
    gl_Position = CameraState.projectionMatrix * vec4(viewPosition, 1.0);
}