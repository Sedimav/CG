#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

namespace application {
namespace {

constexpr float pi = 3.14159265358979323846f;
constexpr uint32_t max_point_lights = 8;
constexpr uint32_t max_spot_lights = 8;
// Two models, one marker per point, and a marker plus cone per spot.
constexpr uint32_t max_draws = 2 + max_point_lights + 2 * max_spot_lights;

struct Vec3 {
	float x;
	float y;
	float z;
};

struct alignas(16) Vec4 {
	float x;
	float y;
	float z;
	float w;
};

struct Mat4 {
	std::array<float, 16> v{};

	float& at(int row, int col) {
		return v[size_t(col * 4 + row)];
	}

	const float& at(int row, int col) const {
		return v[size_t(col * 4 + row)];
	}
};

struct Vertex {
	float position[3];
	float normal[3];
	float color[3];
};

struct SceneUniforms {
	float view_projection[16];
	Vec4 camera_position;
	Vec4 ambient;
	Vec4 directional_direction;
	Vec4 directional_color;
	int32_t counts[4];
};

struct ModelUniforms {
	float model[16];
	float normal_matrix[16];
	Vec4 albedo;
	Vec4 specular;
	float shininess;
	float padding[3];
};

struct PointLightGpu {
	Vec4 position_radius;
	Vec4 color_intensity;
};

struct SpotLightGpu {
	Vec4 position_radius;
	Vec4 direction_inner;
	Vec4 color_intensity;
	Vec4 outer_padding;
};

static_assert(sizeof(SceneUniforms) == 144 && offsetof(SceneUniforms, counts) == 128);
static_assert(sizeof(ModelUniforms) == 176 && offsetof(ModelUniforms, shininess) == 160);
static_assert(sizeof(PointLightGpu) == 32 && sizeof(SpotLightGpu) == 64);
static_assert(offsetof(SpotLightGpu, direction_inner) == 16);
static_assert(offsetof(SpotLightGpu, color_intensity) == 32);
static_assert(offsetof(SpotLightGpu, outer_padding) == 48);

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VmaAllocationInfo info{};
};

struct MeshRange {
	uint32_t first_index = 0;
	uint32_t index_count = 0;
};

struct ModelSlot {
	Buffer uniform_buffer;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct PointLight {
	Vec3 position;
	float radius;
	float color[3];
	float intensity;
};

struct SpotLight {
	Vec3 position;
	Vec3 direction;
	float radius;
	float color[3];
	float intensity;
	float inner_angle;
	float outer_angle;
};

enum class ProjectionMode {
	Perspective = 0,
	Orthographic = 1,
};

struct AppState {
	VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipeline lit_pipeline = VK_NULL_HANDLE;
	VkPipeline unlit_pipeline = VK_NULL_HANDLE;
	VkPipeline cone_pipeline = VK_NULL_HANDLE;

	Buffer vertex_buffer;
	Buffer index_buffer;
	Buffer scene_uniform_buffer;
	Buffer point_lights_buffer;
	Buffer spot_lights_buffer;
	std::array<ModelSlot, max_draws> model_slots;

	MeshRange octahedron;
	MeshRange cube;
	MeshRange cone;

	ProjectionMode projection = ProjectionMode::Perspective;
	bool show_second_object = true;
	Vec3 second_position{2.6f, 0.0f, 0.0f};
	Vec3 second_rotation{0.0f, -20.0f, 0.0f};
	Vec3 second_scale{0.7f, 0.7f, 0.7f};
	float second_color[4]{0.4f, 0.8f, 1.0f, 1.0f};
	bool animate_object = false;
	float object_phase = 0.0f;
	float object_speed = 1.0f;
	float trajectory_radius = 1.5f;
	float trajectory_height = 0.6f;
	float spin_speed = 45.0f;
	Vec3 camera_position{0.0f, 1.2f, 7.0f};
	Vec3 camera_rotation{-8.0f, 0.0f, 0.0f};
	float camera_fov = 60.0f;

	Vec3 object_position{0.0f, 0.0f, 0.0f};
	Vec3 object_rotation{0.0f, 25.0f, 0.0f};
	Vec3 object_scale{1.3f, 1.3f, 1.3f};
	float albedo[4]{0.95f, 0.72f, 0.38f, 1.0f};
	float specular[4]{1.0f, 1.0f, 1.0f, 1.0f};
	float shininess = 64.0f;

	float ambient[4]{0.08f, 0.08f, 0.10f, 1.0f};
	Vec3 directional_direction{-0.4f, -1.0f, -0.3f};
	float directional_color[4]{1.0f, 0.95f, 0.82f, 1.0f};
	float directional_intensity = 1.1f;

	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;
	bool show_light_visualization = true;
	bool animate_lights = true;
	float animation_phase = 0.0f;
	double previous_time = 0.0;

};

AppState app;

float radians(float degrees) {
	return degrees * pi / 180.0f;
}

Vec3 add(Vec3 a, Vec3 b) {
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 sub(Vec3 a, Vec3 b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 mul(Vec3 value, float scale) {
	return {value.x * scale, value.y * scale, value.z * scale};
}

float dot(Vec3 a, Vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(Vec3 a, Vec3 b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

float length(Vec3 value) {
	return std::sqrt(std::max(dot(value, value), 0.0f));
}

Vec3 normalize(Vec3 value) {
	const float len = length(value);
	if (len <= 0.00001f) {
		return {0.0f, 1.0f, 0.0f};
	}
	return mul(value, 1.0f / len);
}

Mat4 identity() {
	Mat4 result{};
	result.at(0, 0) = 1.0f;
	result.at(1, 1) = 1.0f;
	result.at(2, 2) = 1.0f;
	result.at(3, 3) = 1.0f;
	return result;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
	Mat4 result{};
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			float value = 0.0f;
			for (int i = 0; i < 4; ++i) {
				value += a.at(row, i) * b.at(i, col);
			}
			result.at(row, col) = value;
		}
	}
	return result;
}

Mat4 translation(Vec3 value) {
	Mat4 result = identity();
	result.at(0, 3) = value.x;
	result.at(1, 3) = value.y;
	result.at(2, 3) = value.z;
	return result;
}

Mat4 scaling(Vec3 value) {
	Mat4 result{};
	result.at(0, 0) = value.x;
	result.at(1, 1) = value.y;
	result.at(2, 2) = value.z;
	result.at(3, 3) = 1.0f;
	return result;
}

Mat4 rotationX(float angle) {
	Mat4 result = identity();
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	result.at(1, 1) = c;
	result.at(1, 2) = -s;
	result.at(2, 1) = s;
	result.at(2, 2) = c;
	return result;
}

Mat4 rotationY(float angle) {
	Mat4 result = identity();
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	result.at(0, 0) = c;
	result.at(0, 2) = s;
	result.at(2, 0) = -s;
	result.at(2, 2) = c;
	return result;
}

Mat4 rotationZ(float angle) {
	Mat4 result = identity();
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	result.at(0, 0) = c;
	result.at(0, 1) = -s;
	result.at(1, 0) = s;
	result.at(1, 1) = c;
	return result;
}

Mat4 rotationAxisAngle(Vec3 axis, float angle) {
	axis = normalize(axis);
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	const float t = 1.0f - c;

	Mat4 result = identity();
	result.at(0, 0) = t * axis.x * axis.x + c;
	result.at(0, 1) = t * axis.x * axis.y - s * axis.z;
	result.at(0, 2) = t * axis.x * axis.z + s * axis.y;
	result.at(1, 0) = t * axis.x * axis.y + s * axis.z;
	result.at(1, 1) = t * axis.y * axis.y + c;
	result.at(1, 2) = t * axis.y * axis.z - s * axis.x;
	result.at(2, 0) = t * axis.x * axis.z - s * axis.y;
	result.at(2, 1) = t * axis.y * axis.z + s * axis.x;
	result.at(2, 2) = t * axis.z * axis.z + c;
	return result;
}

Mat4 rotationFromMinusZ(Vec3 direction) {
	const Vec3 from{0.0f, 0.0f, -1.0f};
	const Vec3 to = normalize(direction);
	const float c = std::clamp(dot(from, to), -1.0f, 1.0f);
	if (c > 0.999f) {
		return identity();
	}
	if (c < -0.999f) {
		return rotationAxisAngle({0.0f, 1.0f, 0.0f}, pi);
	}
	const Vec3 axis = cross(from, to);
	return rotationAxisAngle(axis, std::acos(c));
}

Mat4 perspective(float fov_y, float aspect, float near_plane, float far_plane) {
	Mat4 result{};
	const float f = 1.0f / std::tan(fov_y * 0.5f);
	result.at(0, 0) = f / aspect;
	result.at(1, 1) = -f;
	result.at(2, 2) = far_plane / (far_plane - near_plane);
	result.at(2, 3) = -(far_plane * near_plane) / (far_plane - near_plane);
	result.at(3, 2) = 1.0f;
	return result;
}

Mat4 orthographic(float width, float height, float near_plane, float far_plane) {
	Mat4 result = identity();
	result.at(0, 0) = 2.0f / width;
	result.at(1, 1) = -2.0f / height;
	result.at(2, 2) = 1.0f / (far_plane - near_plane);
	result.at(2, 3) = -near_plane / (far_plane - near_plane);
	return result;
}

Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
	const Vec3 f = normalize(sub(center, eye));
	const Vec3 s = normalize(cross(f, up));
	const Vec3 u = cross(s, f);

	Mat4 result = identity();
	result.at(0, 0) = s.x;
	result.at(0, 1) = s.y;
	result.at(0, 2) = s.z;
	result.at(0, 3) = -dot(s, eye);
	result.at(1, 0) = u.x;
	result.at(1, 1) = u.y;
	result.at(1, 2) = u.z;
	result.at(1, 3) = -dot(u, eye);
	result.at(2, 0) = f.x;
	result.at(2, 1) = f.y;
	result.at(2, 2) = f.z;
	result.at(2, 3) = -dot(f, eye);
	return result;
}

// Inverse transpose of the linear 3x3 part. Translation does not affect normals.
Mat4 normalMatrix(const Mat4& m) {
	const Vec3 a{m.at(0, 0), m.at(1, 0), m.at(2, 0)};
	const Vec3 b{m.at(0, 1), m.at(1, 1), m.at(2, 1)};
	const Vec3 c{m.at(0, 2), m.at(1, 2), m.at(2, 2)};
	const float determinant = dot(a, cross(b, c));
	if (std::abs(determinant) < 1e-8f) {
		return identity();
	}
	const Vec3 columns[] = {mul(cross(b, c), 1.0f / determinant),
	                        mul(cross(c, a), 1.0f / determinant),
	                        mul(cross(a, b), 1.0f / determinant)};
	Mat4 result = identity();
	for (int col = 0; col < 3; ++col) {
		result.at(0, col) = columns[col].x;
		result.at(1, col) = columns[col].y;
		result.at(2, col) = columns[col].z;
	}
	return result;
}

Mat4 modelMatrix(Vec3 position, Vec3 rotation, Vec3 scale) {
	return multiply(
		translation(position),
		multiply(
			rotationZ(radians(rotation.z)),
			multiply(rotationY(radians(rotation.y)),
			         multiply(rotationX(radians(rotation.x)), scaling(scale)))));
}

Mat4 cameraOrientation() {
	return multiply(rotationY(-radians(app.camera_rotation.y)),
	                multiply(rotationX(radians(app.camera_rotation.x)),
	                         rotationZ(radians(app.camera_rotation.z))));
}

Vec3 cameraForward() {
	const Mat4 rotation = cameraOrientation();
	return {-rotation.at(0, 2), -rotation.at(1, 2), -rotation.at(2, 2)};
}

Vec3 cameraUp() {
	const Mat4 rotation = cameraOrientation();
	return {rotation.at(0, 1), rotation.at(1, 1), rotation.at(2, 1)};
}

Mat4 animatedObjectMatrix() {
	Vec3 position = app.object_position;
	Vec3 rotation = app.object_rotation;
	const float t = app.object_phase;
	position = add(position, {app.trajectory_radius * std::sin(t),
	                         app.trajectory_height * std::sin(2.0f * t),
	                         0.5f * app.trajectory_radius * std::sin(2.0f * t)});
	rotation.y += t * app.spin_speed;
	rotation.x += t * app.spin_speed * 0.4f;
	return modelMatrix(position, rotation, app.object_scale);
}

std::vector<char> readFile(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		return {};
	}
	const std::streamsize size = file.tellg();
	if (size <= 0 || size % sizeof(uint32_t) != 0) {
		return {};
	}
	std::vector<char> data(static_cast<size_t>(size));
	file.seekg(0);
	if (!file.read(data.data(), size)) {
		return {};
	}
	return data;
}

VkShaderModule createShaderModule(const char* path) {
	const std::string full_path = std::string(CG_RESOURCE_ROOT) + "/" + path;
	const std::vector<char> code = readFile(full_path.c_str());
	if (code.empty()) {
		std::cerr << "Failed to read shader file: " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const VkShaderModuleCreateInfo shader_module = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &shader_module, nullptr,
	                         &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return VK_NULL_HANDLE;
	}
	return module;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	return vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info,
	                       &allocation_info, &buffer.handle, &buffer.allocation,
	                       &buffer.info) == VK_SUCCESS;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle,
		                 buffer.allocation);
		buffer = {};
	}
}

void copyToBuffer(const Buffer& buffer, const void* data, size_t size) {
	std::memcpy(buffer.info.pMappedData, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void pushVertex(std::vector<Vertex>& vertices, Vec3 position, Vec3 normal, Vec3 color) {
	vertices.push_back({
		{position.x, position.y, position.z},
		{normal.x, normal.y, normal.z},
		{color.x, color.y, color.z},
	});
}

MeshRange appendOctahedron(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const uint16_t base = uint16_t(vertices.size());
	const std::array<Vec3, 6> positions = {{
		{ 0.0f,  1.0f,  0.0f},
		{ 1.0f,  0.0f,  0.0f},
		{ 0.0f,  0.0f,  1.0f},
		{-1.0f,  0.0f,  0.0f},
		{ 0.0f,  0.0f, -1.0f},
		{ 0.0f, -1.0f,  0.0f},
	}};
	for (Vec3 position : positions) {
		const Vec3 color = mul(add(position, {1.0f, 1.0f, 1.0f}), 0.5f);
		pushVertex(vertices, position, normalize(position), color);
	}

	const uint32_t first_index = uint32_t(indices.size());
	const std::array<uint16_t, 24> local_indices = {{
		0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1,
		5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4,
	}};
	for (uint16_t index : local_indices) {
		indices.push_back(uint16_t(base + index));
	}
	return {first_index, uint32_t(local_indices.size())};
}

MeshRange appendCube(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const uint16_t base = uint16_t(vertices.size());
	const std::array<Vec3, 8> positions = {{
		{-1.0f, -1.0f, -1.0f},
		{ 1.0f, -1.0f, -1.0f},
		{ 1.0f,  1.0f, -1.0f},
		{-1.0f,  1.0f, -1.0f},
		{-1.0f, -1.0f,  1.0f},
		{ 1.0f, -1.0f,  1.0f},
		{ 1.0f,  1.0f,  1.0f},
		{-1.0f,  1.0f,  1.0f},
	}};
	for (Vec3 position : positions) {
		pushVertex(vertices, position, normalize(position), {1.0f, 1.0f, 1.0f});
	}

	const uint32_t first_index = uint32_t(indices.size());
	const std::array<uint16_t, 36> local_indices = {{
		0, 2, 1, 0, 3, 2,
		4, 5, 6, 4, 6, 7,
		0, 1, 5, 0, 5, 4,
		3, 6, 2, 3, 7, 6,
		1, 2, 6, 1, 6, 5,
		0, 4, 7, 0, 7, 3,
	}};
	for (uint16_t index : local_indices) {
		indices.push_back(uint16_t(base + index));
	}
	return {first_index, uint32_t(local_indices.size())};
}

MeshRange appendCone(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const uint16_t base = uint16_t(vertices.size());
	const uint32_t segments = 32;
	pushVertex(vertices, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
	pushVertex(vertices, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f});

	for (uint32_t i = 0; i < segments; ++i) {
		const float a = float(i) / float(segments) * 2.0f * pi;
		pushVertex(vertices, {std::cos(a), std::sin(a), -1.0f},
		           {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f});
	}

	const uint32_t first_index = uint32_t(indices.size());
	for (uint32_t i = 0; i < segments; ++i) {
		const uint16_t a = uint16_t(base + 2 + i);
		const uint16_t b = uint16_t(base + 2 + ((i + 1) % segments));
		indices.push_back(a);
		indices.push_back(b);
		if (i % 8 == 0) {
			indices.push_back(base);
			indices.push_back(a);
		}
	}
	return {first_index, uint32_t(indices.size() - first_index)};
}

bool createGeometryBuffers() {
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
	app.octahedron = appendOctahedron(vertices, indices);
	app.cube = appendCube(vertices, indices);
	app.cone = appendCone(vertices, indices);

	if (!createBuffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  app.vertex_buffer)) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}
	copyToBuffer(app.vertex_buffer, vertices.data(), sizeof(Vertex) * vertices.size());

	if (!createBuffer(sizeof(uint16_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                  app.index_buffer)) {
		std::cerr << "Failed to create index buffer\n";
		return false;
	}
	copyToBuffer(app.index_buffer, indices.data(), sizeof(uint16_t) * indices.size());

	if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  app.scene_uniform_buffer)) {
		std::cerr << "Failed to create scene uniform buffer\n";
		return false;
	}
	if (!createBuffer(sizeof(PointLightGpu) * max_point_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  app.point_lights_buffer)) {
		std::cerr << "Failed to create point light storage buffer\n";
		return false;
	}
	if (!createBuffer(sizeof(SpotLightGpu) * max_spot_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  app.spot_lights_buffer)) {
		std::cerr << "Failed to create spot light storage buffer\n";
		return false;
	}
	for (ModelSlot& slot : app.model_slots) {
		if (!createBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  slot.uniform_buffer)) {
			std::cerr << "Failed to create model uniform buffer\n";
			return false;
		}
	}
	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const std::array<VkDescriptorSetLayoutBinding, 3> scene_bindings = {{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	}};
	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(scene_bindings.size()),
		.pBindings = scene_bindings.data(),
	};
	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &app.scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &app.model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	const std::array<VkDescriptorPoolSize, 2> pool_sizes = {{
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + max_draws },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	}};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + max_draws,
		.poolSizeCount = uint32_t(pool_sizes.size()),
		.pPoolSizes = pool_sizes.data(),
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &app.descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = app.descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &app.scene_set_layout,
	};
	if (vkAllocateDescriptorSets(context.device, &scene_allocate_info,
	                             &app.scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	std::array<VkDescriptorSetLayout, max_draws> model_layouts{};
	model_layouts.fill(app.model_set_layout);
	std::array<VkDescriptorSet, max_draws> model_sets{};
	const VkDescriptorSetAllocateInfo model_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = app.descriptor_pool,
		.descriptorSetCount = max_draws,
		.pSetLayouts = model_layouts.data(),
	};
	if (vkAllocateDescriptorSets(context.device, &model_allocate_info, model_sets.data()) !=
	    VK_SUCCESS) {
		std::cerr << "Failed to allocate model descriptor sets\n";
		return false;
	}
	for (uint32_t i = 0; i < max_draws; ++i) {
		app.model_slots[i].descriptor_set = model_sets[i];
	}

	const VkDescriptorBufferInfo scene_buffer = {
		.buffer = app.scene_uniform_buffer.handle,
		.offset = 0,
		.range = sizeof(SceneUniforms),
	};
	const VkDescriptorBufferInfo point_buffer = {
		.buffer = app.point_lights_buffer.handle,
		.offset = 0,
		.range = sizeof(PointLightGpu) * max_point_lights,
	};
	const VkDescriptorBufferInfo spot_buffer = {
		.buffer = app.spot_lights_buffer.handle,
		.offset = 0,
		.range = sizeof(SpotLightGpu) * max_spot_lights,
	};
	std::array<VkWriteDescriptorSet, 3 + max_draws> writes{};
	writes[0] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = app.scene_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &scene_buffer,
	};
	writes[1] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = app.scene_descriptor_set,
		.dstBinding = 1,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &point_buffer,
	};
	writes[2] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = app.scene_descriptor_set,
		.dstBinding = 2,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.pBufferInfo = &spot_buffer,
	};

	std::array<VkDescriptorBufferInfo, max_draws> model_buffers{};
	for (uint32_t i = 0; i < max_draws; ++i) {
		model_buffers[i] = {
			.buffer = app.model_slots[i].uniform_buffer.handle,
			.offset = 0,
			.range = sizeof(ModelUniforms),
		};
		writes[3 + i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = app.model_slots[i].descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &model_buffers[i],
		};
	}
	vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
	return true;
}

bool createPipeline(const char* vertex_path, const char* fragment_path, VkPipeline& pipeline,
                    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
	auto& context = graphics::internal::context;
	VkShaderModule vertex_shader = createShaderModule(vertex_path);
	VkShaderModule fragment_shader = createShaderModule(fragment_path);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		if (vertex_shader != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		if (fragment_shader != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages = {{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	}};
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const std::array<VkVertexInputAttributeDescription, 3> vertex_attributes = {{
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, normal) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, color) },
	}};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = uint32_t(vertex_attributes.size()),
		.pVertexAttributeDescriptions = vertex_attributes.data(),
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};
	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
		                  VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};
	const std::array<VkDynamicState, 2> dynamic_states = {{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	}};
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = uint32_t(dynamic_states.size()),
		.pDynamicStates = dynamic_states.data(),
	};

	if (app.pipeline_layout == VK_NULL_HANDLE) {
		const std::array<VkDescriptorSetLayout, 2> set_layouts = {{
			app.scene_set_layout,
			app.model_set_layout,
		}};
		const VkPipelineLayoutCreateInfo pipeline_layout = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = uint32_t(set_layouts.size()),
			.pSetLayouts = set_layouts.data(),
		};
		if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
		                           &app.pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create pipeline layout\n";
			return false;
		}
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = uint32_t(shader_stages.size()),
		.pStages = shader_stages.data(),
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = app.pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE,
	                                                  1, &pipeline_info, nullptr,
	                                                  &pipeline);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}
	return true;
}

void initializeLights() {
	app.point_lights.push_back({
		.position = {-2.4f, 1.4f, 1.8f},
		.radius = 0.16f,
		.color = {1.0f, 0.26f, 0.18f},
		.intensity = 7.0f,
	});
	app.point_lights.push_back({
		.position = {2.2f, 1.1f, -1.3f},
		.radius = 0.16f,
		.color = {0.15f, 0.55f, 1.0f},
		.intensity = 6.0f,
	});
	app.spot_lights.push_back({
		.position = {0.0f, 3.0f, 2.6f},
		.direction = {0.0f, -0.75f, -1.0f},
		.radius = 0.18f,
		.color = {0.75f, 1.0f, 0.55f},
		.intensity = 12.0f,
		.inner_angle = 14.0f,
		.outer_angle = 24.0f,
	});
}

void updateLightAnimation() {
	if (!app.animate_lights || app.point_lights.empty()) {
		return;
	}
	app.point_lights[0].position.x = std::cos(app.animation_phase) * 2.5f;
	app.point_lights[0].position.z = std::sin(app.animation_phase) * 2.0f;
	if (app.point_lights.size() > 1) {
		app.point_lights[1].position.y = 1.1f + std::sin(app.animation_phase * 1.7f) * 0.6f;
	}
}

void updateSceneBuffers() {
	auto& context = graphics::internal::context;
	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);
	const float aspect = width / height;

	const Mat4 view = lookAt(app.camera_position, add(app.camera_position, cameraForward()),
	                         cameraUp());
	const Mat4 projection =
		app.projection == ProjectionMode::Perspective
			? perspective(radians(app.camera_fov), aspect, 0.1f, 100.0f)
			: orthographic(7.0f * aspect, 7.0f, 0.1f, 100.0f);
	const Mat4 view_projection = multiply(projection, view);

	SceneUniforms scene{};
	std::memcpy(scene.view_projection, view_projection.v.data(), sizeof(scene.view_projection));
	scene.camera_position = {app.camera_position.x, app.camera_position.y,
	                         app.camera_position.z, 1.0f};
	scene.ambient = {app.ambient[0], app.ambient[1], app.ambient[2], 1.0f};
	const Vec3 dir = normalize(app.directional_direction);
	scene.directional_direction = {dir.x, dir.y, dir.z, 0.0f};
	scene.directional_color = {
		app.directional_color[0], app.directional_color[1],
		app.directional_color[2], app.directional_intensity,
	};
	scene.counts[0] = int(app.point_lights.size());
	scene.counts[1] = int(app.spot_lights.size());
	copyToBuffer(app.scene_uniform_buffer, &scene, sizeof(scene));

	std::array<PointLightGpu, max_point_lights> points{};
	for (size_t i = 0; i < app.point_lights.size() && i < points.size(); ++i) {
		const PointLight& light = app.point_lights[i];
		points[i].position_radius = {light.position.x, light.position.y, light.position.z,
		                             light.radius};
		points[i].color_intensity = {light.color[0], light.color[1], light.color[2],
		                             light.intensity};
	}
	copyToBuffer(app.point_lights_buffer, points.data(), sizeof(points));

	std::array<SpotLightGpu, max_spot_lights> spots{};
	for (size_t i = 0; i < app.spot_lights.size() && i < spots.size(); ++i) {
		const SpotLight& light = app.spot_lights[i];
		const Vec3 light_dir = normalize(light.direction);
		spots[i].position_radius = {light.position.x, light.position.y, light.position.z,
		                            light.radius};
		spots[i].direction_inner = {light_dir.x, light_dir.y, light_dir.z,
		                            std::cos(radians(light.inner_angle))};
		spots[i].color_intensity = {light.color[0], light.color[1], light.color[2],
		                            light.intensity};
		spots[i].outer_padding = {std::cos(radians(light.outer_angle)), 0.0f, 0.0f, 0.0f};
	}
	copyToBuffer(app.spot_lights_buffer, spots.data(), sizeof(spots));
}

void writeModel(uint32_t slot, Mat4 model, const float* albedo, const float* specular,
                float shininess) {
	ModelUniforms model_uniform{};
	std::memcpy(model_uniform.model, model.v.data(), sizeof(model_uniform.model));
	const Mat4 normal = normalMatrix(model);
	std::memcpy(model_uniform.normal_matrix, normal.v.data(), sizeof(model_uniform.normal_matrix));
	model_uniform.albedo = {albedo[0], albedo[1], albedo[2], 1.0f};
	model_uniform.specular = {specular[0], specular[1], specular[2], 1.0f};
	model_uniform.shininess = shininess;
	copyToBuffer(app.model_slots[slot].uniform_buffer, &model_uniform, sizeof(model_uniform));
}

void updateModelBuffers() {
	uint32_t slot = 0;
	writeModel(slot, animatedObjectMatrix(),
	           app.albedo, app.specular, app.shininess);
	++slot;
	if (app.show_second_object) {
		writeModel(slot, modelMatrix(app.second_position, app.second_rotation, app.second_scale),
		           app.second_color, app.specular, app.shininess);
		++slot;
	}

	if (app.show_light_visualization) {
		const float white[4]{1.0f, 1.0f, 1.0f, 1.0f};
		for (const PointLight& light : app.point_lights) {
			if (slot >= max_draws) {
				break;
			}
			const float albedo[4]{light.color[0], light.color[1], light.color[2], 1.0f};
			writeModel(slot, modelMatrix(light.position, {0.0f, 0.0f, 0.0f},
			                            {light.radius, light.radius, light.radius}),
			           albedo, white, 1.0f);
			++slot;
		}
		for (const SpotLight& light : app.spot_lights) {
			if (slot + 1 >= max_draws) {
				break;
			}
			const float albedo[4]{light.color[0], light.color[1], light.color[2], 1.0f};
			writeModel(slot, modelMatrix(light.position, {0.0f, 0.0f, 0.0f},
			                            {light.radius, light.radius, light.radius}),
			           albedo, white, 1.0f);
			++slot;

			const float cone_scale = 1.8f;
			const float cone_radius = std::tan(radians(light.outer_angle)) * cone_scale;
			const Mat4 cone_model = multiply(
				translation(light.position),
				multiply(rotationFromMinusZ(light.direction),
				         scaling({cone_radius, cone_radius, cone_scale})));
			writeModel(slot, cone_model, white, white, 1.0f);
			++slot;
		}
	}
}

void drawMesh(VkCommandBuffer command_buffer, VkPipeline pipeline, uint32_t model_slot,
              MeshRange mesh) {
	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	std::array<VkDescriptorSet, 2> sets = {{
		app.scene_descriptor_set,
		app.model_slots[model_slot].descriptor_set,
	}};
	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        app.pipeline_layout, 0, uint32_t(sets.size()), sets.data(),
	                        0, nullptr);
	vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, mesh.first_index, 0, 0);
}

void drawModelUi() {
	ImGui::SeparatorText("Model");
	ImGui::Checkbox("Show second octahedron", &app.show_second_object);
	ImGui::DragFloat3("Object position", &app.object_position.x, 0.05f, -8.0f, 8.0f);
	ImGui::DragFloat3("Object rotation", &app.object_rotation.x, 1.0f, -360.0f, 360.0f);
	ImGui::DragFloat3("Object scale", &app.object_scale.x, 0.02f, 0.1f, 4.0f);
	ImGui::ColorEdit3("Material albedo", app.albedo);
	ImGui::ColorEdit3("Specular color", app.specular);
	ImGui::SliderFloat("Shininess", &app.shininess, 2.0f, 256.0f);
	app.object_scale.x = std::clamp(app.object_scale.x, 0.1f, 4.0f);
	app.object_scale.y = std::clamp(app.object_scale.y, 0.1f, 4.0f);
	app.object_scale.z = std::clamp(app.object_scale.z, 0.1f, 4.0f);
	if (app.show_second_object && ImGui::TreeNode("Second octahedron")) {
		ImGui::DragFloat3("Position", &app.second_position.x, 0.05f);
		ImGui::DragFloat3("Rotation", &app.second_rotation.x, 1.0f);
		ImGui::SliderFloat3("Scale", &app.second_scale.x, 0.1f, 4.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::ColorEdit3("Color", app.second_color);
		ImGui::TreePop();
	}
	ImGui::SeparatorText("Object animation");
	ImGui::Checkbox("Play object animation", &app.animate_object);
	ImGui::SameLine();
	if (ImGui::Button("Reset animation")) app.object_phase = 0.0f;
	ImGui::SliderFloat("Animation speed", &app.object_speed, 0.0f, 3.0f);
	ImGui::SliderFloat("Trajectory radius", &app.trajectory_radius, 0.0f, 3.0f);
	ImGui::SliderFloat("Trajectory height", &app.trajectory_height, 0.0f, 2.0f);
	ImGui::SliderFloat("Spin (degrees per phase)", &app.spin_speed, 0.0f, 180.0f);
}

void drawCameraUi() {
	ImGui::SeparatorText("Camera");
	const char* projection_items[] = { "Perspective", "Orthographic" };
	int projection_index = int(app.projection);
	if (ImGui::Combo("Projection", &projection_index, projection_items, 2)) {
		app.projection = ProjectionMode(projection_index);
	}
	ImGui::DragFloat3("Camera position", &app.camera_position.x, 0.05f, -20.0f, 20.0f);
	ImGui::DragFloat3("Camera rotation", &app.camera_rotation.x, 0.5f, -180.0f, 180.0f);
	ImGui::SliderFloat("Field of view", &app.camera_fov, 20.0f, 100.0f);
}

void drawLightUi() {
	ImGui::SeparatorText("Lighting");
	ImGui::ColorEdit3("Ambient light", app.ambient);
	ImGui::DragFloat3("Directional direction", &app.directional_direction.x, 0.02f, -1.0f, 1.0f);
	ImGui::ColorEdit3("Directional color", app.directional_color);
	ImGui::SliderFloat("Directional intensity", &app.directional_intensity, 0.0f, 5.0f);
	ImGui::Checkbox("Animate lights", &app.animate_lights);
	ImGui::Checkbox("Show light visualization", &app.show_light_visualization);

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Add point light") && app.point_lights.size() < max_point_lights) {
			app.point_lights.push_back({
				.position = {0.0f, 1.5f, 2.0f},
				.radius = 0.16f,
				.color = {1.0f, 0.8f, 0.45f},
				.intensity = 5.0f,
			});
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove point light") && !app.point_lights.empty()) {
			app.point_lights.pop_back();
		}
		for (size_t i = 0; i < app.point_lights.size(); ++i) {
			ImGui::PushID(int(i));
			if (ImGui::TreeNode("Point light")) {
				PointLight& light = app.point_lights[i];
				ImGui::DragFloat3("Position", &light.position.x, 0.05f, -8.0f, 8.0f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 30.0f);
				ImGui::SliderFloat("Marker radius", &light.radius, 0.05f, 0.4f);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Add spot light") && app.spot_lights.size() < max_spot_lights) {
			app.spot_lights.push_back({
				.position = {0.0f, 3.0f, 2.0f},
				.direction = {0.0f, -1.0f, -1.0f},
				.radius = 0.18f,
				.color = {0.7f, 1.0f, 0.6f},
				.intensity = 8.0f,
				.inner_angle = 12.0f,
				.outer_angle = 22.0f,
			});
		}
		ImGui::SameLine();
		if (ImGui::Button("Remove spot light") && !app.spot_lights.empty()) {
			app.spot_lights.pop_back();
		}
		for (size_t i = 0; i < app.spot_lights.size(); ++i) {
			ImGui::PushID(int(i));
			if (ImGui::TreeNode("Spot light")) {
				SpotLight& light = app.spot_lights[i];
				ImGui::DragFloat3("Position", &light.position.x, 0.05f, -8.0f, 8.0f);
				ImGui::DragFloat3("Direction", &light.direction.x, 0.02f, -1.0f, 1.0f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 40.0f);
				ImGui::SliderFloat("Inner angle", &light.inner_angle, 1.0f, 45.0f);
				ImGui::SliderFloat("Outer angle", &light.outer_angle, 1.0f, 60.0f);
				light.outer_angle = std::max(light.outer_angle, light.inner_angle);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}
}

} // namespace

bool initialize() {
	initializeLights();
	if (!createGeometryBuffers()) {
		return false;
	}
	if (!createDescriptors()) {
		return false;
	}
	if (!createPipeline("shaders/lab2_lit.vert.spv", "shaders/lab2_lit.frag.spv",
	                    app.lit_pipeline)) {
		return false;
	}
	if (!createPipeline("shaders/lab2_unlit.vert.spv", "shaders/lab2_unlit.frag.spv",
	                    app.unlit_pipeline)) {
		return false;
	}
	if (!createPipeline("shaders/lab2_unlit.vert.spv", "shaders/lab2_unlit.frag.spv",
	                    app.cone_pipeline, VK_PRIMITIVE_TOPOLOGY_LINE_LIST)) {
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	if (app.cone_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, app.cone_pipeline, nullptr);
	}
	if (app.unlit_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, app.unlit_pipeline, nullptr);
	}
	if (app.lit_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, app.lit_pipeline, nullptr);
	}
	if (app.pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, app.pipeline_layout, nullptr);
	}
	if (app.descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, app.descriptor_pool, nullptr);
	}
	if (app.model_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, app.model_set_layout, nullptr);
	}
	if (app.scene_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, app.scene_set_layout, nullptr);
	}

	for (ModelSlot& slot : app.model_slots) {
		destroyBuffer(slot.uniform_buffer);
	}
	destroyBuffer(app.spot_lights_buffer);
	destroyBuffer(app.point_lights_buffer);
	destroyBuffer(app.scene_uniform_buffer);
	destroyBuffer(app.index_buffer);
	destroyBuffer(app.vertex_buffer);
	app = {};
}

void update(double time) {
	const double delta = app.previous_time == 0.0 ? 0.0 : time - app.previous_time;
	app.previous_time = time;
	if (app.animate_lights) app.animation_phase += float(delta);
	if (app.animate_object) app.object_phase += float(delta) * app.object_speed;
	updateLightAnimation();

	ImGui::Begin("Lab 2: Camera and Lighting");
	ImGui::TextUnformatted("Variant 2: regular octahedron");

	drawCameraUi();
	drawModelUi();
	drawLightUi();
	ImGui::End();

	updateSceneBuffers();
	updateModelBuffers();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	vkResetCommandBuffer(fd.command_buffer, 0);
	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin);

	const VkClearValue clear_values[] = {
		{ .color = { { 0.015f, 0.017f, 0.022f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};
	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = {
		.extent = context.swapchain_extent,
	};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &app.vertex_buffer.handle, &vertex_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, app.index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);

	uint32_t slot = 0;
	const VkPipeline object_pipeline = app.lit_pipeline;
	drawMesh(fd.command_buffer, object_pipeline, slot++, app.octahedron);
	if (app.show_second_object) drawMesh(fd.command_buffer, object_pipeline, slot++, app.octahedron);
	if (app.show_light_visualization) {
		for (size_t i = 0; i < app.point_lights.size(); ++i) {
			drawMesh(fd.command_buffer, app.unlit_pipeline, slot++, app.cube);
		}
		for (size_t i = 0; i < app.spot_lights.size(); ++i) {
			drawMesh(fd.command_buffer, app.unlit_pipeline, slot++, app.cube);
			drawMesh(fd.command_buffer, app.cone_pipeline, slot++, app.cone);
		}
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
