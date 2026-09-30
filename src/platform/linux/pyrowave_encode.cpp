/**
 * @file src/platform/linux/pyrowave_encode.cpp
 * @brief PyroWave (Vulkan wavelet, intra-only) host encoder for Linux.
 */
#ifdef SUNSHINE_ENABLE_PYROWAVE

  // The include order below is deliberate: pyrowave.h fails to compile unless the Vulkan headers
  // were included first, and clang-format would otherwise sort the quoted includes ahead of them.
  // clang-format off
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
     * @brief Resolve the Vulkan format used to sample a captured DRM buffer.
     *
     * The component mapping stays the identity: Vulkan's format swizzle already presents B8G8R8A8
     * as (R, G, B, A) to the shader, so PyroWave's scaler samples the expected RGB channels.
     *
     * @param fourcc DRM fourcc of the captured buffer.
     * @return Matching Vulkan format; BGRA8 is assumed for unknown formats.
     */
    VkFormat drm_fourcc_to_vk_format(uint32_t fourcc) {
      switch (fourcc) {
        case DRM_FORMAT_XRGB8888:
        case DRM_FORMAT_ARGB8888:
          return VK_FORMAT_B8G8R8A8_UNORM;
        case DRM_FORMAT_XBGR8888:
        case DRM_FORMAT_ABGR8888:
          return VK_FORMAT_R8G8B8A8_UNORM;
        case DRM_FORMAT_XRGB2101010:
        case DRM_FORMAT_ARGB2101010:
          return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
        case DRM_FORMAT_XBGR2101010:
        case DRM_FORMAT_ABGR2101010:
          return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        default:
          BOOST_LOG(warning) << "PyroWave: unknown DRM fourcc 0x" << std::hex << fourcc << std::dec << ", assuming B8G8R8A8";
          return VK_FORMAT_B8G8R8A8_UNORM;
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

  }  // namespace

  crop_rect_t compute_crop_rect(int src_width, int src_height, int width, int height) {
    crop_rect_t crop {};
    if (src_width <= 0 || src_height <= 0 || width <= 0 || height <= 0) {
      return crop;
    }
    if ((int64_t) src_width * height == (int64_t) src_height * width) {
      return crop;  // Same aspect ratio; scale the whole frame.
    }

    int crop_w = src_width;
    int crop_h = src_height;
    if ((int64_t) src_width * height > (int64_t) src_height * width) {
      // The source is wider than the stream: keep the full height and trim the sides.
      crop_w = (int) ((int64_t) src_height * width / height);
    } else {
      crop_h = (int) ((int64_t) src_width * height / width);
    }

    // Even alignment keeps 4:2:0 chroma siting clean.
    crop.width = crop_w & ~1;
    crop.height = crop_h & ~1;
    if (crop.width <= 0 || crop.height <= 0) {
      return crop;
    }
    crop.x = ((src_width - crop.width) / 2) & ~1;
    crop.y = ((src_height - crop.height) / 2) & ~1;
    crop.needed = true;
    return crop;
  }

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
    bool ten_bit = false;  ///< True when planes use R16_UNORM containers.
    bool hdr = false;  ///< True to use the BT.2020 PQ color math.
    size_t max_bitstream = 0;  ///< Per-frame bitstream budget in bytes.

    /// Whether the capture-failure diagnostic was already logged for this session.
    bool logged_capture_failure = false;

    // Packetization buffers, reused across frames (see the sizing comment in encode()).
    std::vector<uint8_t> scratch;  ///< Bitstream output of pyrowave_encoder_packetize().
    std::vector<pyrowave_packet> packets;  ///< Packet table returned by pyrowave_encoder_packetize().

    // Import synchronization on PyroWave's own VkDevice: a one-shot command buffer and fence
    // used to move the freshly captured DMA-BUF into a layout the GPU encode can sample.
    VkDevice vk_dev = VK_NULL_HANDLE;  ///< Device PyroWave created and owns.
    VkPhysicalDevice vk_phys = VK_NULL_HANDLE;  ///< Physical device backing vk_dev.
    VkQueue vk_queue = VK_NULL_HANDLE;  ///< Queue used for the import barrier.
    uint32_t vk_family = 0;  ///< Queue family of vk_queue.
    VkCommandPool vk_pool = VK_NULL_HANDLE;  ///< Command pool for the import barrier.
    VkCommandBuffer vk_cmd = VK_NULL_HANDLE;  ///< Command buffer recording the import barrier.
    VkFence vk_fence = VK_NULL_HANDLE;  ///< Fence waited on after the import barrier.
    PFN_vkGetMemoryFdPropertiesKHR getMemoryFdProperties = nullptr;  ///< External-memory entry point.

    // Per-frame imported dmabuf image (RGB), kept alive until the GPU encode has sampled it.
    VkImage imp_image = VK_NULL_HANDLE;  ///< Imported captured buffer.
    VkDeviceMemory imp_mem = VK_NULL_HANDLE;  ///< Memory exported by the captured buffer.

    ~impl_t() {
      if (enc) {
        pyrowave_encoder_destroy(enc);
      }
      // Destroy our GPU resources before the pyrowave device (they live on its VkDevice).
      if (vk_dev) {
        vkDeviceWaitIdle(vk_dev);
        destroy_import();
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
     * @brief Adopt PyroWave's Vulkan device and create the import-sync resources.
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

      getMemoryFdProperties = (PFN_vkGetMemoryFdPropertiesKHR) vkGetDeviceProcAddr(vk_dev, "vkGetMemoryFdPropertiesKHR");
      return true;
    }

    /**
     * @brief Release the per-frame imported capture image.
     */
    void destroy_import() {
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
      VkFormat vk_format = drm_fourcc_to_vk_format(sd.fourcc);

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
      return true;
    }

    /**
     * @brief Import the captured DMA-BUF and make it readable by the GPU encode.
     *
     * The buffer is imported as an RGB VkImage on PyroWave's device and transitioned to
     * VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL (discarding its externally owned contents), then
     * the transition is waited on. PyroWave samples the image later, when its encode commands run,
     * so the caller must keep the import alive until the encoder's fence has been waited on and
     * release it with destroy_import().
     *
     * @param desc Captured image descriptor.
     * @return True on success.
     */
    bool prepare_dmabuf(const egl::img_descriptor_t &desc) {
      if (!import_dmabuf(desc.sd)) {
        return false;
      }

      vkResetCommandBuffer(vk_cmd, 0);
      VkCommandBufferBeginInfo beg {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(vk_cmd, &beg);

      VkImageMemoryBarrier mb {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      mb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      mb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      mb.srcQueueFamilyIndex = mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      mb.image = imp_image;
      mb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      mb.srcAccessMask = 0;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(vk_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &mb);
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
      return true;
    }

    /**
     * @brief Fill the PyroWave scaled-encode descriptor for the imported capture.
     *
     * PyroWave's scaler owns the RGB->YUV conversion and the scaling into the encoder's resolution,
     * so the captured image is handed to it directly. The scaler always fills the encoder frame,
     * which is why a source with a different aspect ratio is center-cropped rather than letterboxed.
     *
     * @param desc Captured image descriptor.
     * @param crop Receives the center-crop rectangle when one is needed; stays untouched otherwise.
     *             It must outlive the encode call, which keeps a pointer to it.
     * @param info Receives the encode descriptor.
     */
    void fill_scaled_info(const egl::img_descriptor_t &desc, VkRect2D &crop, pyrowave_scaled_encode_info &info) {
      const auto &sd = desc.sd;
      info = {};
      info.view.image = imp_image;
      info.view.width = (uint32_t) sd.width;
      info.view.height = (uint32_t) sd.height;
      info.view.image_format = drm_fourcc_to_vk_format(sd.fourcc);
      info.view.view_format = info.view.image_format;
      info.view.mip_level = 0;
      info.view.layer = 0;
      info.view.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
      info.view.swizzle = VK_COMPONENT_SWIZZLE_IDENTITY;
      info.view.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      // Linux captures arrive display-referred: sRGB for SDR, PQ/BT.2020 for HDR. The scaler
      // re-emits the same encoding, so the full-range BT.709 (SDR) or BT.2020 NCL (HDR) YCbCr
      // matches the sequence header PyroWave writes.
      info.input_color_space = hdr ? VK_COLOR_SPACE_HDR10_ST2084_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      info.output_color_space = info.input_color_space;
      // PyroWave is depth-agnostic (normalized-float wavelet); the plane container depth just has
      // to match what the client decodes into.
      info.intermediate_plane_format = ten_bit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
      // Neutral chroma code point for the container depth (the codec itself is float).
      info.ycbcr_chroma_midpoint = ten_bit ? 512.0f / 1023.0f : 128.0f / 255.0f;
      info.force_linear_filtering = false;
      info.skip_dither = false;

      auto crop_rect = compute_crop_rect(sd.width, sd.height, width, height);
      if (crop_rect.needed) {
        crop.extent = {(uint32_t) crop_rect.width, (uint32_t) crop_rect.height};
        crop.offset = {(int32_t) crop_rect.x, (int32_t) crop_rect.y};
        info.crop_rect = &crop;
      }
    }
  };

  std::unique_ptr<encoder_t> encoder_t::create(int width, int height, int bitrate_kbps, int frame_rate, bool yuv444, bool ten_bit, bool hdr) {
    // 4:2:0 requires even dimensions (harmless for 4:4:4; keeps the scaler/crop math shared).
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

    // PyroWave requires a GPU DMA-BUF capture (e.g. encoder=vulkan): the captured buffer is
    // imported straight into PyroWave's Vulkan device and handed to its GPU scaler, which owns the
    // RGB->YUV conversion and the scaling into the encoder's resolution. Host-mapped
    // (software-capture) frames, which carry a non-null img.data, are not supported.
    if (img.data != nullptr) {
      log_capture_failure("PyroWave requires a DMA-BUF capture (use encoder=vulkan)");
      return -1;
    }
    const auto *desc = dynamic_cast<const egl::img_descriptor_t *>(&img);
    if (!desc) {
      log_capture_failure("PyroWave requires a DMA-BUF capture (use encoder=vulkan)");
      return -1;
    }
    if (!impl.prepare_dmabuf(*desc)) {
      log_capture_failure("PyroWave: dmabuf import failed");
      return -1;
    }

    VkRect2D crop {};
    pyrowave_scaled_encode_info scaling {};
    impl.fill_scaled_info(*desc, crop, scaling);

    pyrowave_rate_control rc {};
    rc.maximum_bitstream_size = impl.max_bitstream;

    if (pyrowave_encoder_encode_gpu_scaled_synchronous(impl.enc, nullptr, nullptr, &scaling, &rc) != PYROWAVE_SUCCESS) {
      impl.destroy_import();
      BOOST_LOG(error) << "PyroWave: encode_gpu_scaled_synchronous failed";
      return -1;
    }

    // Number of leading packets that carry the coarsest wavelet bands, for the RTP layer's
    // asymmetric FEC. Computed before packetizing; the packing is deterministic and identical.
    // This waits for the GPU encode's fence, which also means the captured buffer is no longer in
    // use once it returns, so the import can be released.
    size_t critical_packets = 0;
    if (pyrowave_encoder_compute_num_critical_packets(impl.enc, FEC_PROTECTED_BANDS, PACKET_BOUNDARY, 0, &critical_packets) != PYROWAVE_SUCCESS) {
      critical_packets = 0;
    }
    impl.destroy_import();

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
