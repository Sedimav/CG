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
constexpr uint32_t max_draws = 2;

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
	float color[3];
};

struct SceneUniforms {
	float view_projection[16];
};

struct ModelUniforms {
	float model[16];
	Vec4 color;
};

static_assert(sizeof(SceneUniforms) == 64);
static_assert(sizeof(ModelUniforms) == 80 && offsetof(ModelUniforms, color) == 64);

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
	VkPipeline unlit_pipeline = VK_NULL_HANDLE;

	Buffer vertex_buffer;
	Buffer index_buffer;
	Buffer scene_uniform_buffer;
	std::array<ModelSlot, max_draws> model_slots;

	MeshRange octahedron;

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
	float object_color[4]{0.95f, 0.72f, 0.38f, 1.0f};
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

void pushVertex(std::vector<Vertex>& vertices, Vec3 position, Vec3 color) {
	vertices.push_back({
		{position.x, position.y, position.z},
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
		pushVertex(vertices, position, color);
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

bool createGeometryBuffers() {
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
	app.octahedron = appendOctahedron(vertices, indices);

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

	const std::array<VkDescriptorSetLayoutBinding, 1> scene_bindings = {{
		{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		 .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT},
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

	const std::array<VkDescriptorPoolSize, 1> pool_sizes = {{
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + max_draws },
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
	std::array<VkWriteDescriptorSet, 1 + max_draws> writes{};
	writes[0] = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = app.scene_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &scene_buffer,
	};
	std::array<VkDescriptorBufferInfo, max_draws> model_buffers{};
	for (uint32_t i = 0; i < max_draws; ++i) {
		model_buffers[i] = {
			.buffer = app.model_slots[i].uniform_buffer.handle,
			.offset = 0,
			.range = sizeof(ModelUniforms),
		};
		writes[1 + i] = {
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
	const std::array<VkVertexInputAttributeDescription, 2> vertex_attributes = {{
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
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
	copyToBuffer(app.scene_uniform_buffer, &scene, sizeof(scene));
}

void writeModel(uint32_t slot, Mat4 model, const float* color) {
	ModelUniforms data{};
	std::memcpy(data.model, model.v.data(), sizeof(data.model));
	data.color = {color[0], color[1], color[2], 1.0f};
	copyToBuffer(app.model_slots[slot].uniform_buffer, &data, sizeof(data));
}

void updateModelBuffers() {
	uint32_t slot = 0;
	writeModel(slot, animatedObjectMatrix(),
	           app.object_color);
	++slot;
	if (app.show_second_object) {
		writeModel(slot, modelMatrix(app.second_position, app.second_rotation, app.second_scale),
		           app.second_color);
		++slot;
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
	ImGui::ColorEdit3("Object color", app.object_color);
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

} // namespace

bool initialize() {
	if (!createGeometryBuffers()) {
		return false;
	}
	if (!createDescriptors()) {
		return false;
	}
	if (!createPipeline("shaders/lab1.vert.spv", "shaders/lab1.frag.spv", app.unlit_pipeline)) {
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	if (app.unlit_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, app.unlit_pipeline, nullptr);
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
	destroyBuffer(app.scene_uniform_buffer);
	destroyBuffer(app.index_buffer);
	destroyBuffer(app.vertex_buffer);
	app = {};
}

void update(double time) {
	const double delta = app.previous_time == 0.0 ? 0.0 : time - app.previous_time;
	app.previous_time = time;
	if (app.animate_object) app.object_phase += float(delta) * app.object_speed;

	ImGui::Begin("Lab 1: Transformations");
	ImGui::TextUnformatted("Variant 2: regular octahedron");

	drawCameraUi();
	drawModelUi();
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
	const VkPipeline object_pipeline = app.unlit_pipeline;
	drawMesh(fd.command_buffer, object_pipeline, slot++, app.octahedron);
	if (app.show_second_object) drawMesh(fd.command_buffer, object_pipeline, slot++, app.octahedron);
	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
