#include "water/renderer/vk_device.h"
#include <stdexcept>
#include <vector>
#include <cstring>
#include <iostream>

namespace water::renderer {

namespace {

const std::vector<const char*> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
};

bool device_supports_extensions(VkPhysicalDevice dev,
                                 const std::vector<const char*>& required) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> props(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, props.data());

    for (const char* name : required) {
        bool found = false;
        for (const auto& p : props) {
            if (std::strcmp(p.extensionName, name) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

void check_vk(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan failure: ") + what
                                  + " (VkResult=" + std::to_string(r) + ")");
    }
}

} // namespace

VulkanDevice::VulkanDevice() {
    create_instance();
    pick_physical_device();
    create_logical_device();
}

VulkanDevice::~VulkanDevice() {
    if (device_)   vkDestroyDevice(device_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void VulkanDevice::create_instance() {
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "water_sim";
    app.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
    app.pEngineName        = "water_sim";
    app.engineVersion      = VK_MAKE_VERSION(2, 0, 0);
    app.apiVersion         = VK_API_VERSION_1_4;

    VkInstanceCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;

    check_vk(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
}

void VulkanDevice::pick_physical_device() {
    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (n == 0) throw std::runtime_error("No Vulkan-capable physical devices");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());

    // Prefer a discrete GPU that supports our required extensions.
    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (auto d : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        bool ext_ok = device_supports_extensions(d, kRequiredDeviceExtensions);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && ext_ok) {
            physical_ = d;
            rt_supported_ = true;
            return;
        }
        if (ext_ok && fallback == VK_NULL_HANDLE) fallback = d;
    }
    if (fallback != VK_NULL_HANDLE) {
        physical_ = fallback;
        rt_supported_ = true;
        return;
    }
    throw std::runtime_error("No Vulkan device found with required RT extensions");
}

void VulkanDevice::create_logical_device() {
    // Find a queue family with graphics+compute+transfer.
    std::uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qfc, nullptr);
    std::vector<VkQueueFamilyProperties> qfp(qfc);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qfc, qfp.data());
    queue_family_ = UINT32_MAX;
    for (std::uint32_t i = 0; i < qfc; ++i) {
        const auto flags = qfp[i].queueFlags;
        if ((flags & VK_QUEUE_GRAPHICS_BIT) &&
            (flags & VK_QUEUE_COMPUTE_BIT)  &&
            (flags & VK_QUEUE_TRANSFER_BIT)) {
            queue_family_ = i; break;
        }
    }
    if (queue_family_ == UINT32_MAX) {
        throw std::runtime_error("No queue family with graphics+compute+transfer");
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family_;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    // Required feature chain for ray tracing.
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as_features.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_features{};
    rt_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rt_features.rayTracingPipeline = VK_TRUE;
    rt_features.pNext = &as_features;

    VkPhysicalDeviceBufferDeviceAddressFeatures bda_features{};
    bda_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bda_features.bufferDeviceAddress = VK_TRUE;
    bda_features.pNext = &rt_features;

    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.descriptorIndexing  = VK_TRUE;
    v12.runtimeDescriptorArray = VK_TRUE;
    v12.pNext = &bda_features;

    VkPhysicalDeviceVulkan13Features v13{};
    v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13.synchronization2 = VK_TRUE;
    v13.dynamicRendering = VK_TRUE;
    v13.pNext = &v12;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &v13;

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = static_cast<std::uint32_t>(kRequiredDeviceExtensions.size());
    dci.ppEnabledExtensionNames = kRequiredDeviceExtensions.data();
    dci.pNext                   = &features2;

    check_vk(vkCreateDevice(physical_, &dci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

VkDeviceInfo VulkanDevice::info() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_, &props);
    VkDeviceInfo i;
    i.device_name           = props.deviceName;
    i.api_version           = props.apiVersion;
    i.driver_version        = props.driverVersion;
    i.ray_tracing_supported = rt_supported_;
    return i;
}

} // namespace water::renderer
