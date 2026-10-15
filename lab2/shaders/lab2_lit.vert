#version 450

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 directional_direction;
	vec4 directional_color;
	ivec4 counts;
} scene;

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
	float shininess;
} model_data;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 frag_position;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_color;

void main() {
	vec4 world_position = model_data.model * vec4(in_position, 1.0);
	gl_Position = scene.view_projection * world_position;
	frag_position = world_position.xyz;
	frag_normal = normalize((model_data.normal_matrix * vec4(in_normal, 0.0)).xyz);
	frag_color = in_color;
}
