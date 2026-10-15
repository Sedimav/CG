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

layout(location = 0) out vec3 frag_color;

void main() {
	gl_Position = scene.view_projection * model_data.model * vec4(in_position, 1.0);
	frag_color = model_data.albedo.rgb * in_color;
}
