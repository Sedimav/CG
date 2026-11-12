#version 450

struct PointLight {
	vec4 position_radius;
	vec4 color_intensity;
};

struct SpotLight {
	vec4 position_radius;
	vec4 direction_inner;
	vec4 color_intensity;
	vec4 outer_padding;
};

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 directional_direction;
	vec4 directional_color;
	ivec4 counts;
} scene;

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
	float shininess;
} model_data;

layout(location = 0) in vec3 frag_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;

layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 1) uniform sampler2D albedo_texture;
layout(set = 1, binding = 2) uniform sampler2D specular_texture;
layout(set = 1, binding = 3) uniform sampler2D emissive_texture;

layout(push_constant) uniform Sampling {
	uint flags;
	float time;
	float strength;
	float uv_scale;
} sampling;

vec3 sampleAlbedo(vec2 uv) {
	vec3 base = texture(albedo_texture, uv).rgb;
	if ((sampling.flags & 1u) == 0u) return base;
	float t = sampling.time;
	vec2 flow = vec2(sin(uv.y * 8.0 + t), cos(uv.x * 8.0 - t)) * 0.06;
	vec3 a = texture(albedo_texture, uv + flow + vec2(t * 0.06, 0.0)).rgb;
	vec3 b = texture(albedo_texture, uv * 1.7 - flow + vec2(0.0, -t * 0.04)).rgb;
	vec3 tint = 0.75 + 0.25 * cos(t + vec3(0.0, 2.1, 4.2));
	return mix(base, mix(a, b, 0.4) * tint, sampling.strength);
}

vec3 blinnPhong(vec3 light_dir, vec3 light_color, float intensity, float attenuation,
                vec3 normal, vec3 view_dir, vec3 base_color, vec3 specular_color) {
	float diffuse_factor = max(dot(normal, light_dir), 0.0);
	vec3 halfway_sum = light_dir + view_dir;
	vec3 halfway = halfway_sum / max(length(halfway_sum), 1e-6);
	float specular_factor = diffuse_factor > 0.0
		? pow(max(dot(normal, halfway), 0.0), model_data.shininess) : 0.0;
	vec3 diffuse = base_color * diffuse_factor;
	vec3 specular = specular_color * specular_factor;
	return (diffuse + specular) * light_color * intensity * attenuation;
}

void main() {
	vec3 normal = normalize(frag_normal);
	vec3 view_dir = normalize(scene.camera_position.xyz - frag_position);
	vec2 uv = frag_uv * sampling.uv_scale;
	vec3 base_color = sampleAlbedo(uv);
	vec3 specular_color = (sampling.flags & 2u) != 0u
		? texture(specular_texture, uv).rgb * model_data.specular.rgb : vec3(0.0);
	vec3 emissive = (sampling.flags & 4u) != 0u
		? texture(emissive_texture, uv).rgb : vec3(0.0);

	vec3 color = scene.ambient.rgb * base_color;

	vec3 directional_dir = normalize(-scene.directional_direction.xyz);
	color += blinnPhong(directional_dir, scene.directional_color.rgb,
	                    scene.directional_color.a, 1.0,
	                    normal, view_dir, base_color, specular_color);

	for (int i = 0; i < scene.counts.x; ++i) {
		vec3 to_light = point_lights[i].position_radius.xyz - frag_position;
		float distance_sq = max(dot(to_light, to_light), 0.01);
		vec3 light_dir = normalize(to_light);
		float attenuation = 1.0 / distance_sq;
		color += blinnPhong(light_dir, point_lights[i].color_intensity.rgb,
		                    point_lights[i].color_intensity.a, attenuation,
		                    normal, view_dir, base_color, specular_color);
	}

	for (int i = 0; i < scene.counts.y; ++i) {
		vec3 to_light = spot_lights[i].position_radius.xyz - frag_position;
		float distance_sq = max(dot(to_light, to_light), 0.01);
		vec3 light_dir = normalize(to_light);
		vec3 spot_direction = normalize(spot_lights[i].direction_inner.xyz);
		float theta = dot(normalize(-light_dir), spot_direction);
		float inner_cos = spot_lights[i].direction_inner.a;
		float outer_cos = spot_lights[i].outer_padding.x;
		float spot_factor = inner_cos > outer_cos
			? smoothstep(outer_cos, inner_cos, theta) : step(outer_cos, theta);
		float attenuation = spot_factor / distance_sq;
		color += blinnPhong(light_dir, spot_lights[i].color_intensity.rgb,
		                    spot_lights[i].color_intensity.a, attenuation,
		                    normal, view_dir, base_color, specular_color);
	}

	out_color = vec4(color + emissive, 1.0);
}
