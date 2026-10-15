#version 450
layout(set = 0, binding = 0) uniform Scene { mat4 view_projection; } scene;
layout(set = 1, binding = 0) uniform Model { mat4 model; vec4 color; } model_data;
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 0) out vec3 frag_color;
void main() {
    gl_Position = scene.view_projection * model_data.model * vec4(in_position, 1.0);
    frag_color = in_color * model_data.color.rgb;
}
