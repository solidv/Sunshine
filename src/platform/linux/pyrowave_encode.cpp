/**
 * @file src/platform/linux/pyrowave_encode.cpp
 * @brief PyroWave (Vulkan wavelet, intra-only) host encoder for Linux.
 */
#ifdef SUNSHINE_ENABLE_PYROWAVE

  // The include order below is deliberate: pyrowave.h fails to compile unless the Vulkan headers
  // were included first, and clang-format would otherwise sort the quoted includes ahead of them.
  // clang-format off
  #include <algorithm>
  #include <array>
  #include <vector>

  #include <unistd.h>

  #include <drm_fourcc.h>
  #include <vulkan/vulkan.h>

  #include "pyrowave.h"

  #include "graphics.h"
  #include "pyrowave_encode.h"
  #include "src/logging.h"
  #include "src/platform/common.h"
// clang-format on

namespace platf::pyrowave {

  namespace {

    // RGB->YUV plane conversion (SPIR-V), compiled from
    // src_assets/linux/assets/shaders/vulkan/pyrowave_rgb2yuv.comp by the pyrowave_vulkan_shaders target.
    const uint32_t rgb2yuv_spv[] =
  #include "shaders/pyrowave_rgb2yuv.spv.inc"
      ;

    /// PyroWave packet size in bytes. Each packet is independently decodable, so this is a
    /// loss-resilience knob as much as a network one (Moonlight ships one packet per UDP datagram).
    constexpr size_t PACKET_BOUNDARY = 1024;

    /**
     * @brief Wavelet bands (counted from the coarsest level) covered by the FEC-protected head.
     *
     * PyroWave numbers blocks coarsest-first, and `bands = 3` covers the two coarsest
     * decomposition levels (4 and 3) — the sequence header plus the 1/32- and 1/16-scale
     * coefficients. Loss there is catastrophic, while loss in the finer tail degrades to blur
     * that the client's partial-frame delivery can still show, so only this prefix is protected.
     */
    constexpr int FEC_PROTECTED_BANDS = 3;

    /// Scratch/bitstream buffer sizing, see the comment in encoder_t::encode().
    constexpr size_t MAX_BLOCK_BYTES = 64 * 1024;

    /**
     * @brief Map a DRM fourcc to a Vulkan format + component swizzle (matches vulkan_encode.cpp).
     */
    struct drm_format_info {
      VkFormat format;  ///< Vulkan format matching the DRM fourcc.
      VkComponentMapping swizzle;  ///< Component mapping needed to present the format as RGB.
    };

    /**
     * @brief Resolve the Vulkan format and swizzle used to sample a captured DRM buffer.
     *
     * @param fourcc DRM fourcc of the captured buffer.
     * @return Matching Vulkan format and component swizzle; BGRA is assumed for unknown formats.
     */
    drm_format_info drm_fourcc_to_vk_format(uint32_t fourcc) {
      constexpr VkComponentMapping identity = {
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY,
        VK_COMPONENT_SWIZZLE_IDENTITY
      };
      switch (fourcc) {
        case DRM_FORMAT_XRGB8888:
        case DRM_FORMAT_ARGB8888:
          return {VK_FORMAT_B8G8R8A8_UNORM, identity};
        case DRM_FORMAT_XBGR8888:
        case DRM_FORMAT_ABGR8888:
          return {VK_FORMAT_R8G8B8A8_UNORM, identity};
        case DRM_FORMAT_XRGB2101010:
        case DRM_FORMAT_ARGB2101010:
          return {VK_FORMAT_A2R10G10B10_UNORM_PACK32, identity};
        case DRM_FORMAT_XBGR2101010:
        case DRM_FORMAT_ABGR2101010:
          return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, identity};
        default:
          BOOST_LOG(warning) << "PyroWave: unknown DRM fourcc 0x" << std::hex << fourcc << std::dec << ", assuming B8G8R8A8";
          return {VK_FORMAT_B8G8R8A8_UNORM, identity};
      }
    }

    /**
     * @brief Query how many DRM format modifier planes a format/device pair uses.
     *
     * @param phys Physical device the image will be created on.
     * @param format Vulkan format of the image.
     * @param modifier DRM format modifier describing the buffer layout.
     * @return Plane count reported for the modifier, or 0 when the modifier is unsupported.
     */
    int query_modifier_plane_count(VkPhysicalDevice phys, VkFormat format, uint64_t modifier) {
      VkDrmFormatModifierPropertiesListEXT mod_list = {VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
      VkFormatProperties2 fp = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
      fp.pNext = &mod_list;
      vkGetPhysicalDeviceFormatProperties2(phys, format, &fp);
      std::vector<VkDrmFormatModifierPropertiesEXT> props(mod_list.drmFormatModifierCount);
      mod_list.pDrmFormatModifierProperties = props.data();
      vkGetPhysicalDeviceFormatProperties2(phys, format, &fp);
      for (const auto &mp : props) {
        if (mp.drmFormatModifier == modifier) {
          return mp.drmFormatModifierPlaneCount;
        }
      }
      return 0;
    }

    /**
     * @brief Features PyroWave's encoder requires, per pyrowave.h.
     */
    struct required_features_t {
      bool shader_int16 = false;  ///< 16-bit integer shader operations.
      bool storage_buffer_8bit = false;  ///< 8-bit storage buffer access.
      bool subgroup_size_control = false;  ///< Vulkan 1.3 subgroup size control.

      /**
       * @brief Report whether every required feature is present.
       *
       * @return True when PyroWave's encoder can run on the device.
       */
      bool ok() const {
        return shader_int16 && storage_buffer_8bit && subgroup_size_control;
      }
    };

    /**
     * @brief Query the compute features PyroWave's encoder needs from a physical device.
     *
     * @param phys Physical device to inspect.
     * @return Feature set found on the device.
     */
    required_features_t query_device_features(VkPhysicalDevice phys) {
      required_features_t out;

      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(phys, &props);
      if (props.apiVersion < VK_API_VERSION_1_3) {
        return out;  // not ok
      }

      VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
      VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      f12.pNext = &f13;
      VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &f12;
      vkGetPhysicalDeviceFeatures2(phys, &f2);

      out.shader_int16 = f2.features.shaderInt16 == VK_TRUE;
      out.storage_buffer_8bit = f12.storageBuffer8BitAccess == VK_TRUE;
      out.subgroup_size_control = f13.subgroupSizeControl == VK_TRUE;
      return out;
    }

    /**
     * @brief Find a queue family for PyroWave.
     *
     * Prefer a universal graphics+compute queue, which is what PyroWave's reference adoption uses:
     * Granite maps its graphics/compute/transfer queues onto it and its internal cross-queue sync
     * works. A compute-only family caused a self-deadlock in Granite's submission path.
     *
     * @param phys Physical device to inspect.
     * @param family_out Receives the selected queue family index.
     * @return True when a suitable queue family was found.
     */
    bool find_compute_queue_family(VkPhysicalDevice phys, uint32_t &family_out) {
      uint32_t count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
      std::vector<VkQueueFamilyProperties> families(count);
      vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, families.data());
      for (uint32_t i = 0; i < count; i++) {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
          family_out = i;
          return true;
        }
      }
      for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          family_out = i;
          return true;
        }
      }
      return false;
    }

    /**
     * @brief Find a memory type satisfying the requested properties.
     *
     * @param pd Physical device that owns the memory.
     * @param bits Memory type bits required by the resource.
     * @param props Memory properties that must all be present.
     * @return Selected memory type index, or UINT32_MAX when none matches.
     */
    uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags props) {
      VkPhysicalDeviceMemoryProperties mp;
      vkGetPhysicalDeviceMemoryProperties(pd, &mp);
      for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
          return i;
        }
      }
      return UINT32_MAX;
    }

    /**
     * @brief A single R8/R16_UNORM plane image on PyroWave's device.
     *
     * Written by the RGB->YUV compute pass and read by the GPU encode.
     */
    struct plane_image_t {
      VkDevice dev = VK_NULL_HANDLE;  ///< Device owning the image.
      VkImage image = VK_NULL_HANDLE;  ///< Plane image.
      VkDeviceMemory mem = VK_NULL_HANDLE;  ///< Backing memory for the image.
      VkImageView view = VK_NULL_HANDLE;  ///< View used for storage-image writes and sampling.
      int w = 0;  ///< Plane width in pixels.
      int h = 0;  ///< Plane height in pixels.
      VkFormat format = VK_FORMAT_R8_UNORM;  ///< Plane container format.

      /**
       * @brief Create the image, memory, and view.
       *
       * @param device Device to create the resources on.
       * @param pd Physical device used for memory type selection.
       * @param width Plane width in pixels.
       * @param height Plane height in pixels.
       * @param fmt Plane container format.
       * @return True on success.
       */
      bool init(VkDevice device, VkPhysicalDevice pd, int width, int height, VkFormat fmt) {
        dev = device;
        w = width;
        h = height;
        format = fmt;

        VkImageCreateInfo ci {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {(uint32_t) w, (uint32_t) h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(dev, &ci, nullptr, &image) != VK_SUCCESS) {
          return false;
        }
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, image, &mr);
        VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = find_memory_type(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(dev, &ai, nullptr, &mem) != VK_SUCCESS) {
          return false;
        }
        vkBindImageMemory(dev, image, mem, 0);

        VkImageViewCreateInfo vi {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return vkCreateImageView(dev, &vi, nullptr, &view) == VK_SUCCESS;
      }

      /**
       * @brief Describe the plane as a PyroWave encode input view.
       *
       * @return View descriptor; the plane stays in VK_IMAGE_LAYOUT_GENERAL, which PyroWave
       *         expects for GPU buffers it does not transition itself.
       */
      pyrowave_image_view image_view() const {
        pyrowave_image_view v {};
        v.image = image;
        v.width = (uint32_t) w;
        v.height = (uint32_t) h;
        v.image_format = format;
        v.view_format = format;
        v.mip_level = 0;
        v.layer = 0;
        v.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        v.swizzle = VK_COMPONENT_SWIZZLE_IDENTITY;
        v.layout = VK_IMAGE_LAYOUT_GENERAL;
        return v;
      }

      /**
       * @brief Destroy the plane image, view, and memory.
       */
      void destroy() {
        if (view) {
          vkDestroyImageView(dev, view, nullptr);
        }
        if (image) {
          vkDestroyImage(dev, image, nullptr);
        }
        if (mem) {
          vkFreeMemory(dev, mem, nullptr);
        }
      }
    };

  }  // namespace

  bool validate() {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) {
      return false;
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());

    bool ok = false;
    for (auto phys : devs) {
      uint32_t fam;
      if (query_device_features(phys).ok() && find_compute_queue_family(phys, fam)) {
        ok = true;
        break;
      }
    }
    vkDestroyInstance(inst, nullptr);
    return ok;
  }

  struct encoder_t::impl_t {
    pyrowave_device pdev = nullptr;  ///< PyroWave device owning the encoder.
    pyrowave_encoder enc = nullptr;  ///< PyroWave encoder for this session's resolution.
    int width = 0;  ///< Encoded luma width.
    int height = 0;  ///< Encoded luma height.
    bool yuv444 = false;  ///< True for full-resolution chroma.
    bool ten_bit = false;  ///< True when planes use R16_UNORM containers.
    bool hdr = false;  ///< True to use the BT.2020 PQ color math.
    size_t max_bitstream = 0;  ///< Per-frame bitstream budget in bytes.

    /// Whether the capture-failure diagnostic was already logged for this session.
    bool logged_capture_failure = false;

    // Packetization buffers, reused across frames (see the sizing comment in encode()).
    std::vector<uint8_t> scratch;  ///< Bitstream output of pyrowave_encoder_packetize().
    std::vector<pyrowave_packet> packets;  ///< Packet table returned by pyrowave_encoder_packetize().

    // GPU encode resources on PyroWave's own VkDevice.
    VkDevice vk_dev = VK_NULL_HANDLE;  ///< Device PyroWave created and owns.
    VkPhysicalDevice vk_phys = VK_NULL_HANDLE;  ///< Physical device backing vk_dev.
    VkQueue vk_queue = VK_NULL_HANDLE;  ///< Queue used for the RGB->YUV conversion.
    uint32_t vk_family = 0;  ///< Queue family of vk_queue.
    VkCommandPool vk_pool = VK_NULL_HANDLE;  ///< Command pool for the conversion pass.
    VkCommandBuffer vk_cmd = VK_NULL_HANDLE;  ///< Command buffer recording the conversion pass.
    VkFence vk_fence = VK_NULL_HANDLE;  ///< Fence waited on after the conversion pass.
    plane_image_t plane_y;  ///< Luma plane.
    plane_image_t plane_cb;  ///< Cb plane.
    plane_image_t plane_cr;  ///< Cr plane.

    // RGB->YUV compute-convert pipeline.
    VkShaderModule conv_shader = VK_NULL_HANDLE;  ///< Conversion compute shader.
    VkDescriptorSetLayout conv_dsl = VK_NULL_HANDLE;  ///< Descriptor set layout for the conversion.
    VkPipelineLayout conv_pl = VK_NULL_HANDLE;  ///< Pipeline layout for the conversion.
    VkPipeline conv_pipe = VK_NULL_HANDLE;  ///< Conversion compute pipeline.
    VkSampler conv_sampler = VK_NULL_HANDLE;  ///< Linear sampler used for the captured RGB image.
    VkDescriptorPool conv_pool = VK_NULL_HANDLE;  ///< Descriptor pool holding conv_set.
    VkDescriptorSet conv_set = VK_NULL_HANDLE;  ///< Descriptor set bound by the conversion.
    PFN_vkGetMemoryFdPropertiesKHR getMemoryFdProperties = nullptr;  ///< External-memory entry point.

    // Per-frame imported dmabuf image (RGB), destroyed after each convert.
    VkImage imp_image = VK_NULL_HANDLE;  ///< Imported captured buffer.
    VkDeviceMemory imp_mem = VK_NULL_HANDLE;  ///< Memory exported by the captured buffer.
    VkImageView imp_view = VK_NULL_HANDLE;  ///< View of the imported capture.

    /**
     * @brief Push constants matching the layout declared in pyrowave_rgb2yuv.comp.
     */
    struct convert_pc_t {
      int32_t dst[2];  ///< Luma output size.
      int32_t scaled[2];  ///< Aspect-fit content size inside dst.
      int32_t offset[2];  ///< Letterbox offset.
      int32_t flip;  ///< 1 to sample the source bottom-up.
      int32_t chroma444;  ///< 1 for full-resolution chroma.
      int32_t hdr;  ///< 1 for BT.2020 PQ full range.
      int32_t scrgb;  ///< 1 when the source is linear scRGB (always 0 on Linux).
    };

    ~impl_t() {
      if (enc) {
        pyrowave_encoder_destroy(enc);
      }
      // Destroy our GPU resources before the pyrowave device (they live on its VkDevice).
      if (vk_dev) {
        vkDeviceWaitIdle(vk_dev);
        destroy_import();
        plane_y.destroy();
        plane_cb.destroy();
        plane_cr.destroy();
        if (conv_pipe) {
          vkDestroyPipeline(vk_dev, conv_pipe, nullptr);
        }
        if (conv_pl) {
          vkDestroyPipelineLayout(vk_dev, conv_pl, nullptr);
        }
        if (conv_dsl) {
          vkDestroyDescriptorSetLayout(vk_dev, conv_dsl, nullptr);
        }
        if (conv_pool) {
          vkDestroyDescriptorPool(vk_dev, conv_pool, nullptr);
        }
        if (conv_sampler) {
          vkDestroySampler(vk_dev, conv_sampler, nullptr);
        }
        if (conv_shader) {
          vkDestroyShaderModule(vk_dev, conv_shader, nullptr);
        }
        if (vk_fence) {
          vkDestroyFence(vk_dev, vk_fence, nullptr);
        }
        if (vk_pool) {
          vkDestroyCommandPool(vk_dev, vk_pool, nullptr);
        }
      }
      if (pdev) {
        pyrowave_device_destroy(pdev);
      }
    }

    /**
     * @brief Adopt PyroWave's Vulkan device and create the conversion resources.
     *
     * @return True on success.
     */
    bool init_gpu() {
      pyrowave_device_get_vk_device_handles(pdev, nullptr, &vk_phys, &vk_dev);
      if (!vk_dev || !vk_phys) {
        return false;
      }
      if (!find_compute_queue_family(vk_phys, vk_family)) {
        return false;
      }
      vkGetDeviceQueue(vk_dev, vk_family, 0, &vk_queue);
      if (!vk_queue) {
        return false;
      }

      VkCommandPoolCreateInfo pci {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      pci.queueFamilyIndex = vk_family;
      pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      if (vkCreateCommandPool(vk_dev, &pci, nullptr, &vk_pool) != VK_SUCCESS) {
        return false;
      }
      VkCommandBufferAllocateInfo cai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      cai.commandPool = vk_pool;
      cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      cai.commandBufferCount = 1;
      if (vkAllocateCommandBuffers(vk_dev, &cai, &vk_cmd) != VK_SUCCESS) {
        return false;
      }
      VkFenceCreateInfo fci {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      if (vkCreateFence(vk_dev, &fci, nullptr, &vk_fence) != VK_SUCCESS) {
        return false;
      }

      int chroma_w = yuv444 ? width : width / 2;
      int chroma_h = yuv444 ? height : height / 2;
      // PyroWave is depth-agnostic (normalized-float wavelet); the plane container depth just
      // has to match what the client decodes into.
      VkFormat plane_fmt = ten_bit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
      if (!plane_y.init(vk_dev, vk_phys, width, height, plane_fmt) || !plane_cb.init(vk_dev, vk_phys, chroma_w, chroma_h, plane_fmt) || !plane_cr.init(vk_dev, vk_phys, chroma_w, chroma_h, plane_fmt)) {
        return false;
      }

      getMemoryFdProperties = (PFN_vkGetMemoryFdPropertiesKHR) vkGetDeviceProcAddr(vk_dev, "vkGetMemoryFdPropertiesKHR");
      return init_convert();
    }

    /**
     * @brief Build the RGB->YUV compute pipeline and bind the persistent plane storage images.
     *
     * @return True on success.
     */
    bool init_convert() {
      VkSamplerCreateInfo sci {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
      sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
      sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      if (vkCreateSampler(vk_dev, &sci, nullptr, &conv_sampler) != VK_SUCCESS) {
        return false;
      }

      VkDescriptorSetLayoutBinding b[4] = {};
      b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      for (int i = 1; i < 4; i++) {
        b[i] = {(uint32_t) i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      }
      VkDescriptorSetLayoutCreateInfo dli {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      dli.bindingCount = 4;
      dli.pBindings = b;
      if (vkCreateDescriptorSetLayout(vk_dev, &dli, nullptr, &conv_dsl) != VK_SUCCESS) {
        return false;
      }

      VkPushConstantRange pcr {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(convert_pc_t)};
      VkPipelineLayoutCreateInfo pli {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pli.setLayoutCount = 1;
      pli.pSetLayouts = &conv_dsl;
      pli.pushConstantRangeCount = 1;
      pli.pPushConstantRanges = &pcr;
      if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &conv_pl) != VK_SUCCESS) {
        return false;
      }

      VkShaderModuleCreateInfo smi {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      smi.codeSize = sizeof(rgb2yuv_spv);
      smi.pCode = rgb2yuv_spv;
      if (vkCreateShaderModule(vk_dev, &smi, nullptr, &conv_shader) != VK_SUCCESS) {
        return false;
      }

      VkComputePipelineCreateInfo cpi {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      cpi.stage.module = conv_shader;
      cpi.stage.pName = "main";
      cpi.layout = conv_pl;
      if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &conv_pipe) != VK_SUCCESS) {
        return false;
      }

      VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3}
      };
      VkDescriptorPoolCreateInfo dpi {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      dpi.maxSets = 1;
      dpi.poolSizeCount = 2;
      dpi.pPoolSizes = ps;
      if (vkCreateDescriptorPool(vk_dev, &dpi, nullptr, &conv_pool) != VK_SUCCESS) {
        return false;
      }
      VkDescriptorSetAllocateInfo dsa {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      dsa.descriptorPool = conv_pool;
      dsa.descriptorSetCount = 1;
      dsa.pSetLayouts = &conv_dsl;
      if (vkAllocateDescriptorSets(vk_dev, &dsa, &conv_set) != VK_SUCCESS) {
        return false;
      }

      // Bind the persistent plane storage images once (binding 0 / RGB is updated per frame).
      VkDescriptorImageInfo si1 {VK_NULL_HANDLE, plane_y.view, VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo si2 {VK_NULL_HANDLE, plane_cb.view, VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo si3 {VK_NULL_HANDLE, plane_cr.view, VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet w[3] = {};
      w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w[0].dstSet = conv_set;
      w[0].dstBinding = 1;
      w[0].descriptorCount = 1;
      w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      w[0].pImageInfo = &si1;
      w[1] = w[0];
      w[1].dstBinding = 2;
      w[1].pImageInfo = &si2;
      w[2] = w[0];
      w[2].dstBinding = 3;
      w[2].pImageInfo = &si3;
      vkUpdateDescriptorSets(vk_dev, 3, w, 0, nullptr);
      return true;
    }

    /**
     * @brief Release the per-frame imported capture image.
     */
    void destroy_import() {
      if (imp_view) {
        vkDestroyImageView(vk_dev, imp_view, nullptr);
        imp_view = VK_NULL_HANDLE;
      }
      if (imp_image) {
        vkDestroyImage(vk_dev, imp_image, nullptr);
        imp_image = VK_NULL_HANDLE;
      }
      if (imp_mem) {
        vkFreeMemory(vk_dev, imp_mem, nullptr);
        imp_mem = VK_NULL_HANDLE;
      }
    }

    /**
     * @brief Import a captured DMA-BUF as an RGB VkImage on PyroWave's device.
     *
     * @param sd Surface descriptor of the captured buffer.
     * @return True on success.
     */
    bool import_dmabuf(const egl::surface_descriptor_t &sd) {
      destroy_import();

      // The capture owns its file descriptors, so duplicate the one that is imported.
      int fd = dup(sd.fds[0]);
      if (fd < 0) {
        return false;
      }

      VkMemoryFdPropertiesKHR fd_props = {VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
      if (getMemoryFdProperties) {
        getMemoryFdProperties(vk_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fd_props);
      }

      VkExternalMemoryImageCreateInfo ext_ci = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
      ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

      std::array<VkSubresourceLayout, 4> drm_layouts = {};
      VkImageDrmFormatModifierExplicitCreateInfoEXT drm_ci = {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
      VkImageTiling tiling;
      auto [vk_format, vk_swizzle] = drm_fourcc_to_vk_format(sd.fourcc);

      if (sd.modifier != DRM_FORMAT_MOD_INVALID) {
        int dmabuf_planes = 0;
        for (int i = 0; i < 4 && sd.fds[i] >= 0; ++i) {
          dmabuf_planes++;
        }
        int expected = query_modifier_plane_count(vk_phys, vk_format, sd.modifier);
        int plane_count = (expected > 0 && expected <= dmabuf_planes) ? expected : dmabuf_planes;
        for (int i = 0; i < plane_count; ++i) {
          drm_layouts[i].offset = sd.offsets[i];
          drm_layouts[i].rowPitch = sd.pitches[i];
        }
        drm_ci.drmFormatModifier = sd.modifier;
        drm_ci.drmFormatModifierPlaneCount = plane_count;
        drm_ci.pPlaneLayouts = drm_layouts.data();
        ext_ci.pNext = &drm_ci;
        tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
      } else {
        tiling = VK_IMAGE_TILING_LINEAR;
      }

      VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      img_ci.pNext = &ext_ci;
      img_ci.imageType = VK_IMAGE_TYPE_2D;
      img_ci.format = vk_format;
      img_ci.extent = {(uint32_t) sd.width, (uint32_t) sd.height, 1};
      img_ci.mipLevels = 1;
      img_ci.arrayLayers = 1;
      img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
      img_ci.tiling = tiling;
      img_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      img_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      if (vkCreateImage(vk_dev, &img_ci, nullptr, &imp_image) != VK_SUCCESS) {
        close(fd);
        imp_image = VK_NULL_HANDLE;
        BOOST_LOG(error) << "PyroWave: dmabuf vkCreateImage failed (modifier=0x" << std::hex << sd.modifier << std::dec << ")";
        return false;
      }

      VkMemoryRequirements mem_req;
      vkGetImageMemoryRequirements(vk_dev, imp_image, &mem_req);
      VkImportMemoryFdInfoKHR import_fd = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
      import_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
      import_fd.fd = fd;
      VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      ai.pNext = &import_fd;
      ai.allocationSize = mem_req.size;
      ai.memoryTypeIndex = find_memory_type(vk_phys, fd_props.memoryTypeBits ? fd_props.memoryTypeBits : mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(vk_dev, &ai, nullptr, &imp_mem) != VK_SUCCESS) {
        BOOST_LOG(error) << "PyroWave: dmabuf import vkAllocateMemory failed";
        vkDestroyImage(vk_dev, imp_image, nullptr);
        imp_image = VK_NULL_HANDLE;
        return false;
      }
      vkBindImageMemory(vk_dev, imp_image, imp_mem, 0);

      VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      vi.image = imp_image;
      vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vi.format = vk_format;
      vi.components = vk_swizzle;
      vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      if (vkCreateImageView(vk_dev, &vi, nullptr, &imp_view) != VK_SUCCESS) {
        destroy_import();
        return false;
      }
      return true;
    }

    /**
     * @brief Import the captured DMA-BUF and convert RGB->YUV into the plane images on the GPU.
     *
     * @param desc Captured image descriptor.
     * @return True on success.
     */
    bool convert_dmabuf(const egl::img_descriptor_t &desc) {
      const auto &sd = desc.sd;
      if (!import_dmabuf(sd)) {
        return false;
      }

      // Update the RGB sampler binding for this frame's imported image.
      VkDescriptorImageInfo rgb {conv_sampler, imp_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet w {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = conv_set;
      w.dstBinding = 0;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w.pImageInfo = &rgb;
      vkUpdateDescriptorSets(vk_dev, 1, &w, 0, nullptr);

      // Aspect-preserving fit + even-aligned letterbox.
      float scalar = std::min((float) width / sd.width, (float) height / sd.height);
      int scaled_w = ((int) (sd.width * scalar)) & ~1;
      int scaled_h = ((int) (sd.height * scalar)) & ~1;
      convert_pc_t pc {};
      pc.dst[0] = width;
      pc.dst[1] = height;
      pc.scaled[0] = scaled_w > 0 ? scaled_w : 2;
      pc.scaled[1] = scaled_h > 0 ? scaled_h : 2;
      pc.offset[0] = ((width - pc.scaled[0]) / 2) & ~1;
      pc.offset[1] = ((height - pc.scaled[1]) / 2) & ~1;
      pc.flip = desc.y_invert ? 1 : 0;
      pc.chroma444 = yuv444 ? 1 : 0;
      pc.hdr = hdr ? 1 : 0;
      // Linux captures arrive display-referred (sRGB- or PQ-encoded), never linear scRGB.
      pc.scrgb = 0;

      vkResetCommandBuffer(vk_cmd, 0);
      VkCommandBufferBeginInfo beg {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(vk_cmd, &beg);

      auto img_barrier = [&](VkImage im, VkImageLayout o, VkImageLayout n, VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
        VkImageMemoryBarrier mb {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        mb.oldLayout = o;
        mb.newLayout = n;
        mb.srcQueueFamilyIndex = mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.image = im;
        mb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        mb.srcAccessMask = sa;
        mb.dstAccessMask = da;
        vkCmdPipelineBarrier(vk_cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &mb);
      };

      // The imported dmabuf is externally owned and freshly written by the capture, so its contents
      // may be discarded (UNDEFINED) and only the layout needs to become readable.
      img_barrier(imp_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      img_barrier(plane_y.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      img_barrier(plane_cb.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      img_barrier(plane_cr.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

      vkCmdBindPipeline(vk_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, conv_pipe);
      vkCmdBindDescriptorSets(vk_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, conv_pl, 0, 1, &conv_set, 0, nullptr);
      vkCmdPushConstants(vk_cmd, conv_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
      vkCmdDispatch(vk_cmd, (width + 7) / 8, (height + 7) / 8, 1);

      // Make the plane writes visible; PyroWave's encode reads them (in GENERAL) after our fence.
      img_barrier(plane_y.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      img_barrier(plane_cb.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      img_barrier(plane_cr.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      vkEndCommandBuffer(vk_cmd);

      VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &vk_cmd;
      vkResetFences(vk_dev, 1, &vk_fence);
      if (vkQueueSubmit(vk_queue, 1, &si, vk_fence) != VK_SUCCESS) {
        destroy_import();
        return false;
      }
      vkWaitForFences(vk_dev, 1, &vk_fence, VK_TRUE, UINT64_MAX);
      destroy_import();
      return true;
    }
  };

  std::unique_ptr<encoder_t> encoder_t::create(int width, int height, int bitrate_kbps, int frame_rate, bool yuv444, bool ten_bit, bool hdr) {
    // 4:2:0 requires even dimensions (harmless for 4:4:4; keeps the letterbox math shared).
    width &= ~1;
    height &= ~1;
    if (width <= 0 || height <= 0) {
      return nullptr;
    }

    auto self = std::unique_ptr<encoder_t>(new encoder_t());
    self->impl = std::make_unique<impl_t>();
    auto &impl = *self->impl;

    impl.width = width;
    impl.height = height;
    impl.yuv444 = yuv444;
    impl.ten_bit = ten_bit;
    impl.hdr = hdr && ten_bit;  // HDR color math only makes sense in 10-bit containers

    // Per-frame byte budget from bitrate. Intra-only: bitrate / fps bytes per frame.
    if (frame_rate <= 0) {
      frame_rate = 60;
    }
    int64_t bits_per_frame = (int64_t) bitrate_kbps * 1000 / frame_rate;
    impl.max_bitstream = (size_t) (bits_per_frame / 8);
    if (impl.max_bitstream < 4096) {
      impl.max_bitstream = 4096;
    }

    // Let PyroWave create and own its Vulkan (Granite) device. The encoder does not need to share
    // Sunshine's device, and this avoids the raw-device adoption path, which deadlocked inside
    // Granite's cross-queue submission sync when handed a single externally-created queue. vid/pid 0
    // and null UUIDs select the default GPU (the same entry point PyroWave's own tests use).
    if (pyrowave_create_device_by_compat(0, 0, nullptr, nullptr, nullptr, &impl.pdev) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: pyrowave_create_device_by_compat failed";
      return nullptr;
    }

    pyrowave_encoder_create_info eci {};
    eci.device = impl.pdev;
    eci.width = width;
    eci.height = height;
    eci.chroma = yuv444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
    if (pyrowave_encoder_create(&eci, &impl.enc) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: pyrowave_encoder_create failed";
      return nullptr;
    }

    if (!impl.init_gpu()) {
      BOOST_LOG(error) << "PyroWave: GPU encode resource init failed";
      return nullptr;
    }

    BOOST_LOG(info) << "PyroWave encoder ready (GPU): " << width << "x" << height
                    << (yuv444 ? " 4:4:4" : " 4:2:0")
                    << (ten_bit ? (impl.hdr ? " 10-bit HDR" : " 10-bit SDR") : " 8-bit")
                    << " budget " << impl.max_bitstream << " bytes/frame";
    return self;
  }

  int encoder_t::encode(const platf::img_t &img, std::vector<uint8_t> &out, size_t &head_bytes) {
    auto &impl = *this->impl;

    // Never leave a previous frame's boundary behind on an early failure return.
    head_bytes = 0;

    // The capture path is fixed for the lifetime of a session, so a frame that cannot be imported
    // is a persistent condition: report it once rather than at stream framerates.
    auto log_capture_failure = [&impl](const char *message) {
      if (impl.logged_capture_failure) {
        BOOST_LOG(debug) << message;
      } else {
        impl.logged_capture_failure = true;
        BOOST_LOG(error) << message;
      }
    };

    // PyroWave requires a GPU DMA-BUF capture (e.g. encoder=vulkan): it imports the dmabuf and does
    // the RGB->YUV conversion on the GPU (zero-copy). Host-mapped (software-capture) frames, which
    // carry a non-null img.data, are not supported.
    if (img.data != nullptr) {
      log_capture_failure("PyroWave requires a DMA-BUF capture (use encoder=vulkan)");
      return -1;
    }
    const auto *desc = dynamic_cast<const egl::img_descriptor_t *>(&img);
    if (!desc) {
      log_capture_failure("PyroWave requires a DMA-BUF capture (use encoder=vulkan)");
      return -1;
    }
    if (!impl.convert_dmabuf(*desc)) {
      log_capture_failure("PyroWave: dmabuf convert failed");
      return -1;
    }

    pyrowave_gpu_buffers gb {};
    gb.planes[0] = impl.plane_y.image_view();
    gb.planes[1] = impl.plane_cb.image_view();
    gb.planes[2] = impl.plane_cr.image_view();

    pyrowave_rate_control rc {};
    rc.maximum_bitstream_size = impl.max_bitstream;

    if (pyrowave_encoder_encode_gpu_synchronous(impl.enc, nullptr, nullptr, &gb, &rc) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: encode_gpu_synchronous failed";
      return -1;
    }

    // Number of leading packets that carry the coarsest wavelet bands, for the RTP layer's
    // asymmetric FEC. Computed before packetizing; the packing is deterministic and identical.
    size_t critical_packets = 0;
    if (pyrowave_encoder_compute_num_critical_packets(impl.enc, FEC_PROTECTED_BANDS, PACKET_BOUNDARY, 0, &critical_packets) != PYROWAVE_SUCCESS) {
      critical_packets = 0;
    }

    size_t num_packets = 0;
    if (pyrowave_encoder_compute_num_packets(impl.enc, PACKET_BOUNDARY, &num_packets) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: compute_num_packets failed";
      return -1;
    }

    // Size the bitstream buffer for the worst case, NOT num_packets * PACKET_BOUNDARY:
    // pyrowave's packetize() only closes a packet after the block that overflows it, so a packet
    // can exceed PACKET_BOUNDARY by one block (up to 4097 words; payload_words is a 12-bit field),
    // and packetize() does not bounds-check the output buffer in release builds. High-entropy
    // frames (e.g. full-range 10-bit HDR) actually hit this. Buffers are reused across frames
    // (grow-only).
    size_t scratch_bound = 4096 + num_packets * (PACKET_BOUNDARY + MAX_BLOCK_BYTES);
    if (impl.scratch.size() < scratch_bound) {
      impl.scratch.resize(scratch_bound);
    }
    if (impl.packets.size() < num_packets) {
      impl.packets.resize(num_packets);
    }
    size_t out_packets = 0;
    if (pyrowave_encoder_packetize(impl.enc, impl.packets.data(), PACKET_BOUNDARY, &out_packets, impl.scratch.data(), impl.scratch.size()) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: packetize failed";
      return -1;
    }

    // PyroWave splits a frame into several independently-decodable packets, each of which must be
    // pushed separately on the decode side. Sunshine's RTP layer, however, ships one opaque payload
    // per frame and reassembles it as a single buffer. So we frame the PyroWave packets ourselves:
    //
    //   [u32 packet_count] { [u32 size] [size bytes] } * packet_count
    //
    // The Moonlight PyroWave decoder parses this framing and re-pushes each PyroWave packet.
    //
    // The loss-critical head ends where the coarsest bands end. A frame that is entirely critical
    // (or too small to split) has no tail to leave bare, so it keeps the even split (head_bytes 0).
    head_bytes = 0;
    if (critical_packets > 0 && critical_packets < out_packets) {
      head_bytes = 4 + critical_packets * 4;  // Packet count field and the packet size fields.
      for (size_t i = 0; i < critical_packets; i++) {
        head_bytes += impl.packets[i].size;
      }
    }

    auto put_u32 = [&out](uint32_t v) {
      out.push_back((uint8_t) (v & 0xff));
      out.push_back((uint8_t) ((v >> 8) & 0xff));
      out.push_back((uint8_t) ((v >> 16) & 0xff));
      out.push_back((uint8_t) ((v >> 24) & 0xff));
    };

    out.clear();
    put_u32((uint32_t) out_packets);
    for (size_t i = 0; i < out_packets; i++) {
      put_u32((uint32_t) impl.packets[i].size);
      const uint8_t *src = impl.scratch.data() + impl.packets[i].offset;
      out.insert(out.end(), src, src + impl.packets[i].size);
    }
    return 0;
  }

  encoder_t::~encoder_t() = default;

}  // namespace platf::pyrowave

#endif  // SUNSHINE_ENABLE_PYROWAVE
