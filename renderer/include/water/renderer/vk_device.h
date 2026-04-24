#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace water::renderer {

struct VkDeviceInfo {
    std::string device_name;
    std::uint32_t api_version = 0;
    std::uint32_t driver_version = 0;
    bool          ray_tracing_supported = false;
};

// Owns a Vulkan 1.4 instance and physical/logical device pair, with the
// extensions required by Phase 4 ray tracing already enabled. Phase 1 only
// verifies that creation succeeds and reports the device info; no
// command-buffer recording yet.
class VulkanDevice {
public:
    VulkanDevice();
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;

    VkInstance       instance()        const noexcept { return instance_; }
    VkPhysicalDevice physical_device() const noexcept { return physical_; }
    VkDevice         device()          const noexcept { return device_;   }
    std::uint32_t    queue_family_idx()const noexcept { return queue_family_; }
    VkQueue          graphics_queue()  const noexcept { return queue_;    }

    VkDeviceInfo info() const;

private:
    void create_instance();
    void pick_physical_device();
    void create_logical_device();

    VkInstance       instance_     = VK_NULL_HANDLE;
    VkPhysicalDevice physical_     = VK_NULL_HANDLE;
    VkDevice         device_       = VK_NULL_HANDLE;
    VkQueue          queue_        = VK_NULL_HANDLE;
    std::uint32_t    queue_family_ = 0;
    bool             rt_supported_ = false;
};

} // namespace water::renderer
