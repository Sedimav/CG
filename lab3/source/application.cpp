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

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

namespace application {
namespace {

constexpr float pi = 3.14159265358979323846f;
constexpr uint32_t material_count = 3;
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
	float uv[2];
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

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
};

struct Material {
	const char* name = nullptr;
	Texture albedo;
	Texture specular;
	Texture emissive;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct SamplingPush {
	uint32_t flags;
	float time;
	float strength;
	float uv_scale;
};
static_assert(sizeof(SamplingPush) == 16 && offsetof(SamplingPush, time) == 4);
static_assert(sizeof(Vertex) == 32 && offsetof(Vertex, uv) == 24);

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
	Buffer model_uniform_buffer;
	VkDeviceSize model_stride = 0;
	VkCommandPool upload_pool = VK_NULL_HANDLE;
	VkSampler sampler = VK_NULL_HANDLE;
	Texture black_texture;
	std::array<Material, material_count> materials;
	int object_material = 0;
	int second_material = 1;
	bool object_effect = false;
	bool second_effect = true;
	bool use_specular_map = true;
	bool use_emissive_map = true;
	bool animate_sampling = true;
	float sampling_time = 0.0f;
	float sampling_speed = 1.0f;
	float sampling_strength = 0.7f;
	float uv_scale = 1.0f;

	MeshRange octahedron;
	MeshRange cube;
	MeshRange cone;

	ProjectionMode projection = ProjectionMode::Perspective;
	bool show_second_object = true;
	Vec3 second_position{2.6f, 0.0f, 0.0f};
	Vec3 second_rotation{0.0f, -20.0f, 0.0f};
	Vec3 second_scale{0.7f, 0.7f, 0.7f};
	float second_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
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
	float albedo[4]{1.0f, 1.0f, 1.0f, 1.0f};
	float specular[4]{1.0f, 1.0f, 1.0f, 1.0f};
	float shininess = 64.0f;

	float ambient[4]{0.08f, 0.08f, 0.10f, 1.0f};
	Vec3 directional_direction{-0.4f, -1.0f, -0.3f};
	float directional_color[4]{1.0f, 0.95f, 0.82f, 1.0f};
	float directional_intensity = 1.1f;

	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;
	bool show_light_visualization = true;
	bool animate_lights = false;
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

bool createUploadPool() {
	auto& context = graphics::internal::context;
	const VkCommandPoolCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};
	return vkCreateCommandPool(context.device, &info, nullptr, &app.upload_pool) == VK_SUCCESS;
}

VkCommandBuffer beginUpload() {
	auto& context = graphics::internal::context;
	const VkCommandBufferAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = app.upload_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(context.device, &allocate, &cmd) != VK_SUCCESS) return cmd;
	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
		vkFreeCommandBuffers(context.device, app.upload_pool, 1, &cmd);
		return VK_NULL_HANDLE;
	}
	return cmd;
}

bool endUpload(VkCommandBuffer cmd) {
	auto& context = graphics::internal::context;
	bool success = vkEndCommandBuffer(cmd) == VK_SUCCESS;
	VkFence fence = VK_NULL_HANDLE;
	const VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	if (success) success = vkCreateFence(context.device, &fence_info, nullptr, &fence) == VK_SUCCESS;
	if (success) {
		const VkSubmitInfo submit = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1,
			.pCommandBuffers = &cmd,
		};
		success = vkQueueSubmit(context.graphics_queue, 1, &submit, fence) == VK_SUCCESS;
		if (success) success = vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
	}
	if (fence) vkDestroyFence(context.device, fence, nullptr);
	vkFreeCommandBuffers(context.device, app.upload_pool, 1, &cmd);
	return success;
}

bool uploadGeometry(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                    Buffer& destination) {
	auto& context = graphics::internal::context;
	const VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	};
	if (vmaCreateBuffer(context.allocator, &info, &allocation, &destination.handle,
	                    &destination.allocation, &destination.info) != VK_SUCCESS) return false;
	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
	copyToBuffer(staging, data, size);
	VkCommandBuffer cmd = beginUpload();
	if (!cmd) { destroyBuffer(staging); return false; }
	const VkBufferCopy copy{0, 0, size};
	vkCmdCopyBuffer(cmd, staging.handle, destination.handle, 1, &copy);
	const VkBufferMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = usage == VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
			? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : VK_ACCESS_INDEX_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = destination.handle,
		.size = size,
	};
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
	                     0, 0, nullptr, 1, &barrier, 0, nullptr);
	const bool success = endUpload(cmd);
	destroyBuffer(staging);
	return success;
}

void destroyTexture(Texture& texture) {
	auto& context = graphics::internal::context;
	if (texture.view) vkDestroyImageView(context.device, texture.view, nullptr);
	if (texture.image) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	texture = {};
}

bool loadTexture(const char* filename, VkFormat format, Texture& texture) {
	const std::string path = std::string(CG_RESOURCE_ROOT) + "/textures/" + filename;
	int width = 0, height = 0, channels = 0;
	stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
	if (!pixels) {
		std::cerr << "Failed to load texture " << path << ": " << stbi_failure_reason() << '\n';
		return false;
	}
	auto& context = graphics::internal::context;
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);
	if (width <= 0 || height <= 0 || uint32_t(width) > properties.limits.maxImageDimension2D ||
	    uint32_t(height) > properties.limits.maxImageDimension2D) {
		stbi_image_free(pixels);
		return false;
	}
	const VkDeviceSize size = VkDeviceSize(width) * VkDeviceSize(height) * 4;
	Buffer staging;
	if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
		stbi_image_free(pixels);
		return false;
	}
	copyToBuffer(staging, pixels, size);
	stbi_image_free(pixels);
	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {uint32_t(width), uint32_t(height), 1},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	};
	if (vmaCreateImage(context.allocator, &image_info, &allocation, &texture.image,
	                   &texture.allocation, nullptr) != VK_SUCCESS) {
		destroyBuffer(staging);
		return false;
	}
	VkCommandBuffer cmd = beginUpload();
	if (!cmd) { destroyBuffer(staging); return false; }
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = texture.image,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);
	const VkBufferImageCopy copy = {
		.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
		.imageExtent = image_info.extent,
	};
	vkCmdCopyBufferToImage(cmd, staging.handle, texture.image,
	                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);
	const bool success = endUpload(cmd);
	destroyBuffer(staging);
	if (!success) return false;
	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	return vkCreateImageView(context.device, &view_info, nullptr, &texture.view) == VK_SUCCESS;
}

bool createTextures() {
	const VkSamplerCreateInfo sampler = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = 0.0f,
	};
	if (vkCreateSampler(graphics::internal::context.device, &sampler, nullptr,
	                    &app.sampler) != VK_SUCCESS) return false;
	if (!loadTexture("black.png", VK_FORMAT_R8G8B8A8_UNORM, app.black_texture)) return false;
	const char* names[] = {"Ceramic tiles", "Circuit", "Stone"};
	const char* albedo[] = {"tiles_albedo.png", "circuit_albedo.png", "stone_albedo.png"};
	const char* specular[] = {"tiles_specular.png", "circuit_specular.png", nullptr};
	const char* emissive[] = {nullptr, "circuit_emissive.png", nullptr};
	for (uint32_t i = 0; i < material_count; ++i) {
		Material& material = app.materials[i];
		material.name = names[i];
		if (!loadTexture(albedo[i], VK_FORMAT_R8G8B8A8_SRGB, material.albedo)) return false;
		if (specular[i] && !loadTexture(specular[i], VK_FORMAT_R8G8B8A8_UNORM, material.specular)) return false;
		if (emissive[i] && !loadTexture(emissive[i], VK_FORMAT_R8G8B8A8_SRGB, material.emissive)) return false;
	}
	return true;
}

void pushVertex(std::vector<Vertex>& vertices, Vec3 position, Vec3 normal, Vec3 /* color */) {
	vertices.push_back({
		{position.x, position.y, position.z},
		{normal.x, normal.y, normal.z},
		{0.0f, 0.0f},
	});
}

MeshRange appendOctahedron(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	const std::array<Vec3, 6> positions = {{
		{0, 1, 0}, {1, 0, 0}, {0, 0, 1}, {-1, 0, 0}, {0, 0, -1}, {0, -1, 0},
	}};
	const std::array<uint16_t, 24> faces = {{
		0,1,2, 0,2,3, 0,3,4, 0,4,1,
		5,2,1, 5,3,2, 5,4,3, 5,1,4,
	}};
	const float uv[3][2] = {{0.5f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
	const uint32_t first_index = uint32_t(indices.size());
	// Separate copies at UV seams; preserve the smooth normals from lab 2.
	for (size_t i = 0; i < faces.size(); ++i) {
		const Vec3 position = positions[faces[i]];
		const Vec3 normal = normalize(position);
		indices.push_back(uint16_t(vertices.size()));
		vertices.push_back({{position.x, position.y, position.z},
		                    {normal.x, normal.y, normal.z}, {uv[i % 3][0], uv[i % 3][1]}});
	}
	return {first_index, uint32_t(faces.size())};
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

	if (!uploadGeometry(vertices.data(), sizeof(Vertex) * vertices.size(),
	                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, app.vertex_buffer)) return false;
	if (!uploadGeometry(indices.data(), sizeof(uint16_t) * indices.size(),
	                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT, app.index_buffer)) return false;

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
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	app.model_stride = (sizeof(ModelUniforms) + alignment - 1) / alignment * alignment;
	if (!createBuffer(app.model_stride * max_draws, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  app.model_uniform_buffer)) return false;
	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;
	const std::array<VkDescriptorSetLayoutBinding, 3> scene_bindings = {{
		{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
	}};
	const VkDescriptorSetLayoutCreateInfo scene_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(scene_bindings.size()),
		.pBindings = scene_bindings.data(),
	};
	if (vkCreateDescriptorSetLayout(context.device, &scene_layout, nullptr, &app.scene_set_layout) != VK_SUCCESS) return false;
	const std::array<VkDescriptorSetLayoutBinding, 4> material_bindings = {{
		{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
	}};
	const VkDescriptorSetLayoutCreateInfo material_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(material_bindings.size()),
		.pBindings = material_bindings.data(),
	};
	if (vkCreateDescriptorSetLayout(context.device, &material_layout, nullptr, &app.model_set_layout) != VK_SUCCESS) return false;
	const VkDescriptorPoolSize sizes[] = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, material_count},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, material_count * 3},
	};
	const VkDescriptorPoolCreateInfo pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + material_count,
		.poolSizeCount = uint32_t(std::size(sizes)),
		.pPoolSizes = sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool, nullptr, &app.descriptor_pool) != VK_SUCCESS) return false;
	std::array<VkDescriptorSetLayout, 1 + material_count> layouts{};
	layouts.fill(app.model_set_layout);
	layouts[0] = app.scene_set_layout;
	std::array<VkDescriptorSet, 1 + material_count> sets{};
	const VkDescriptorSetAllocateInfo allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = app.descriptor_pool,
		.descriptorSetCount = uint32_t(layouts.size()),
		.pSetLayouts = layouts.data(),
	};
	if (vkAllocateDescriptorSets(context.device, &allocate, sets.data()) != VK_SUCCESS) return false;
	app.scene_descriptor_set = sets[0];
	const VkDescriptorBufferInfo scene_buffers[] = {
		{app.scene_uniform_buffer.handle, 0, sizeof(SceneUniforms)},
		{app.point_lights_buffer.handle, 0, sizeof(PointLightGpu) * max_point_lights},
		{app.spot_lights_buffer.handle, 0, sizeof(SpotLightGpu) * max_spot_lights},
	};
	for (uint32_t i = 0; i < 3; ++i) {
		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = app.scene_descriptor_set,
			.dstBinding = i,
			.descriptorCount = 1,
			.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &scene_buffers[i],
		};
		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}
	for (uint32_t i = 0; i < material_count; ++i) {
		Material& material = app.materials[i];
		material.descriptor_set = sets[i + 1];
		// range describes one model; dynamic offset chooses its aligned slot at draw time.
		const VkDescriptorBufferInfo model_buffer{app.model_uniform_buffer.handle, 0, sizeof(ModelUniforms)};
		const VkDescriptorImageInfo images[] = {
			{app.sampler, material.albedo.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
			{app.sampler, material.specular.view ? material.specular.view : app.black_texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
			{app.sampler, material.emissive.view ? material.emissive.view : app.black_texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
		};
		std::array<VkWriteDescriptorSet, 4> writes{};
		for (uint32_t binding = 0; binding < writes.size(); ++binding) {
			writes[binding] = {
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = material.descriptor_set,
				.dstBinding = binding,
				.descriptorCount = 1,
				.descriptorType = binding == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = binding == 0 ? nullptr : &images[binding - 1],
				.pBufferInfo = binding == 0 ? &model_buffer : nullptr,
			};
		}
		vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
	}
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
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
		  .offset = offsetof(Vertex, uv) },
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
		const VkPushConstantRange push_range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SamplingPush)};
		const VkPipelineLayoutCreateInfo pipeline_layout = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = uint32_t(set_layouts.size()),
			.pSetLayouts = set_layouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_range,
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
	const VkDeviceSize offset = app.model_stride * slot;
	std::memcpy(static_cast<char*>(app.model_uniform_buffer.info.pMappedData) + offset,
	            &model_uniform, sizeof(model_uniform));
	vmaFlushAllocation(graphics::internal::context.allocator, app.model_uniform_buffer.allocation,
	                   offset, sizeof(model_uniform));
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
              MeshRange mesh, int material_index = 0, bool effect = false) {
	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	const VkDescriptorSet sets[] = {
		app.scene_descriptor_set,
		app.materials[size_t(material_index)].descriptor_set,
	};
	const uint32_t offset = uint32_t(app.model_stride * model_slot);
	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        app.pipeline_layout, 0, 2, sets, 1, &offset);
	const uint32_t flags = (effect ? 1u : 0u) | (app.use_specular_map ? 2u : 0u) |
	                       (app.use_emissive_map ? 4u : 0u);
	const SamplingPush push{flags, app.sampling_time, app.sampling_strength, app.uv_scale};
	vkCmdPushConstants(command_buffer, app.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
	                   0, sizeof(push), &push);
	vkCmdDrawIndexed(command_buffer, mesh.index_count, 1, mesh.first_index, 0, 0);
}

void drawTextureUi() {
	const char* names[material_count];
	for (size_t i = 0; i < material_count; ++i) names[i] = app.materials[i].name;
	ImGui::SeparatorText("Textures");
	ImGui::Combo("First material", &app.object_material, names, material_count);
	ImGui::Checkbox("First sampling effect", &app.object_effect);
	if (app.show_second_object) {
		ImGui::Combo("Second material", &app.second_material, names, material_count);
		ImGui::Checkbox("Second sampling effect", &app.second_effect);
	}
	ImGui::Checkbox("Specular map", &app.use_specular_map);
	ImGui::Checkbox("Emissive map", &app.use_emissive_map);
	ImGui::Checkbox("Animate sampling", &app.animate_sampling);
	ImGui::SliderFloat("Sampling speed", &app.sampling_speed, 0.0f, 3.0f);
	ImGui::SliderFloat("Effect strength", &app.sampling_strength, 0.0f, 1.0f);
	ImGui::SliderFloat("UV scale", &app.uv_scale, 1.0f, 6.0f);
}

void drawModelUi() {
	ImGui::SeparatorText("Model");
	ImGui::Checkbox("Show second octahedron", &app.show_second_object);
	ImGui::DragFloat3("Object position", &app.object_position.x, 0.05f, -8.0f, 8.0f);
	ImGui::DragFloat3("Object rotation", &app.object_rotation.x, 1.0f, -360.0f, 360.0f);
	ImGui::DragFloat3("Object scale", &app.object_scale.x, 0.02f, 0.1f, 4.0f);
	ImGui::ColorEdit3("Specular color", app.specular);
	ImGui::SliderFloat("Shininess", &app.shininess, 2.0f, 256.0f);
	app.object_scale.x = std::clamp(app.object_scale.x, 0.1f, 4.0f);
	app.object_scale.y = std::clamp(app.object_scale.y, 0.1f, 4.0f);
	app.object_scale.z = std::clamp(app.object_scale.z, 0.1f, 4.0f);
	if (app.show_second_object && ImGui::TreeNode("Second octahedron")) {
		ImGui::DragFloat3("Position", &app.second_position.x, 0.05f);
		ImGui::DragFloat3("Rotation", &app.second_rotation.x, 1.0f);
		ImGui::SliderFloat3("Scale", &app.second_scale.x, 0.1f, 4.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
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
	if (!createUploadPool() || !createTextures()) return false;
	initializeLights();
	if (!createGeometryBuffers()) {
		return false;
	}
	if (!createDescriptors()) {
		return false;
	}
	if (!createPipeline("shaders/lab3_lit.vert.spv", "shaders/lab3_lit.frag.spv",
	                    app.lit_pipeline)) {
		return false;
	}
	if (!createPipeline("shaders/lab3_unlit.vert.spv", "shaders/lab3_unlit.frag.spv",
	                    app.unlit_pipeline)) {
		return false;
	}
	if (!createPipeline("shaders/lab3_unlit.vert.spv", "shaders/lab3_unlit.frag.spv",
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

	destroyBuffer(app.model_uniform_buffer);
	for (Material& material : app.materials) {
		destroyTexture(material.albedo);
		destroyTexture(material.specular);
		destroyTexture(material.emissive);
	}
	destroyTexture(app.black_texture);
	if (app.sampler) vkDestroySampler(context.device, app.sampler, nullptr);
	if (app.upload_pool) vkDestroyCommandPool(context.device, app.upload_pool, nullptr);
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
	if (app.animate_sampling) app.sampling_time += float(delta) * app.sampling_speed;
	if (app.animate_object) app.object_phase += float(delta) * app.object_speed;
	updateLightAnimation();

	ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(420, 680), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 3: Texture Mapping");
	ImGui::TextUnformatted("Variant 2: regular octahedron");

	drawTextureUi();
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
	drawMesh(fd.command_buffer, object_pipeline, slot++, app.octahedron, app.object_material, app.object_effect);
	if (app.show_second_object) drawMesh(fd.command_buffer, object_pipeline, slot++, app.octahedron, app.second_material, app.second_effect);
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
