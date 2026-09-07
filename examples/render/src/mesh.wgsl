struct Camera {
    view_proj: mat4x4f,
    eye: vec4f,
}

struct Lights {
    direction: vec4f,
    color: vec4f,
    ambient: vec4f,
}

struct Instance {
    model: mat4x4f,
    normal: mat4x4f,
    base_color: vec4f,
    emissive: vec4f,
    params: vec4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lights: Lights;
@group(0) @binding(2) var<uniform> instance: Instance;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) normal: vec3f,
    @location(1) world_position: vec3f,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = instance.model * vec4f(input.position, 1.0);
    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_position;
    output.normal = (instance.normal * vec4f(input.normal, 0.0)).xyz;
    output.world_position = world_position.xyz;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let base_color = instance.base_color;
    if instance.emissive.w > 0.5 {
        return vec4f(base_color.rgb + instance.emissive.rgb, base_color.a);
    }
    let normal = normalize(input.normal);
    let light_dir = normalize(-lights.direction.xyz);
    let diffuse = max(dot(normal, light_dir), 0.0);
    let view_dir = normalize(camera.eye.xyz - input.world_position);
    let half_dir = normalize(light_dir + view_dir);
    let shininess = 2.0 / (instance.params.x * instance.params.x);
    let specular = pow(max(dot(normal, half_dir), 0.0), shininess) * 0.5;
    let lit = base_color.rgb * (lights.ambient.rgb + diffuse * lights.color.rgb)
        + specular * lights.color.rgb;
    return vec4f(lit, base_color.a);
}
