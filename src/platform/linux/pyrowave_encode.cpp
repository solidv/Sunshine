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
  #include <cstring>
  #include <functional>
  #include <future>
  #include <limits>
  #include <map>
  #include <memory>
  #include <optional>
  #include <span>
  #include <stdexcept>
  #include <string>
  #include <thread>
  #include <type_traits>
  #include <vector>

  #include <pthread.h>
  #include <signal.h>

  #include <sys/stat.h>
  #if defined(__FreeBSD__)
    #include <sys/types.h>
  #else
    #include <sys/capability.h>
    #include <sys/sysmacros.h>
  #endif
  #include <unistd.h>

  #include <drm_fourcc.h>
  #include <vulkan/vulkan.h>

  #include "pyrowave.h"
  #include "metal/pyrowave_bitstream.hpp"

  #include "graphics.h"
  #include "pyrowave_encode.h"
  #include "src/logging.h"
  #include "src/platform/common.h"
  #include "src/pyrowave_policy.h"
  #include "src/pyrowave_protocol.h"
// clang-format on

namespace platf::pyrowave {

  namespace {

    /// PyroWave packet size in bytes used when no negotiated packet size is available. Each packet
    /// is independently decodable, so this is a loss-resilience knob as much as a network one.
    constexpr size_t DEFAULT_PACKET_BOUNDARY = 1024;

    /// Bytes the transport framing adds in front of every chunk (`[u32 size]`).
    constexpr size_t TRANSPORT_CHUNK_PREFIX_BYTES = sizeof(uint32_t);

    /// Bytes the RTP layer adds to a negotiated packet size (matches `MAX_RTP_HEADER_SIZE`).
    constexpr int RTP_HEADER_ALLOWANCE = 16;

    /// On-wire bytes of the RTP and GameStream video headers (`video_packet_raw_t`).
    constexpr int VIDEO_PACKET_HEADER_BYTES = 32;

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

    /// Highest Vulkan core version this integration understands and enables when offered.
  #if VK_HEADER_VERSION >= 304
    constexpr uint32_t MAX_SUPPORTED_API_VERSION = VK_API_VERSION_1_4;
  #else
    constexpr uint32_t MAX_SUPPORTED_API_VERSION = VK_API_VERSION_1_3;
  #endif

    /**
     * @brief Resolve the Vulkan format used to sample a captured DRM buffer.
     *
     * The component mapping stays the identity: Vulkan's format swizzle already presents B8G8R8A8
     * as (R, G, B, A) to the shader, so PyroWave's scaler samples the expected RGB channels.
     *
     * @param fourcc DRM fourcc of the captured buffer.
     * @return Matching Vulkan format, or nullopt when PyroWave cannot sample the format.
     */
    std::optional<VkFormat> drm_fourcc_to_vk_format(uint32_t fourcc) {
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
          // DRM_FORMAT_BGRA1010102/RGBA1010102 have no core Vulkan equivalent with the alpha bits
          // in the low two positions; do not silently reinterpret them as 8-bit BGRA.
          return std::nullopt;
      }
    }

    /**
     * @brief Properties of a DRM modifier on the encode device.
     */
    struct modifier_properties_t {
      int planes = 0;  ///< Plane count reported for the modifier.
      bool sampled = false;  ///< True when the modifier supports sampled images.
    };

    /**
     * @brief Query how many planes a format/modifier pair uses and whether it is sampleable.
     *
     * @param phys Physical device to query.
     * @param format Vulkan format of the image.
     * @param modifier DRM modifier to look up.
     * @return Properties reported for the pair; zero planes when unsupported.
     */
    modifier_properties_t query_modifier_properties(VkPhysicalDevice phys, VkFormat format, uint64_t modifier) {
      VkDrmFormatModifierPropertiesListEXT mod_list = {VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
      VkFormatProperties2 fp = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
      fp.pNext = &mod_list;
      vkGetPhysicalDeviceFormatProperties2(phys, format, &fp);
      std::vector<VkDrmFormatModifierPropertiesEXT> props(mod_list.drmFormatModifierCount);
      mod_list.pDrmFormatModifierProperties = props.data();
      vkGetPhysicalDeviceFormatProperties2(phys, format, &fp);
      for (const auto &mp : props) {
        if (mp.drmFormatModifier == modifier) {
          return {
            (int) mp.drmFormatModifierPlaneCount,
            (mp.drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0,
          };
        }
      }
      return {};
    }

    /**
     * @brief Features PyroWave's encoder requires, per pyrowave.h.
     */
    struct required_features_t {
      bool shader_int16 = false;  ///< 16-bit integer shader operations.
      bool storage_buffer_8bit = false;  ///< 8-bit storage buffer access.
      bool subgroup_size_control = false;  ///< Vulkan 1.3 subgroup size control.
      uint32_t api_version = 0;  ///< Effective Vulkan version (lower of instance and device).
      bool vulkan_1_4 = false;  ///< True when Vulkan 1.4 features were queried.
      bool global_priority_query = false;  ///< Vulkan 1.4 global queue priority query support.

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
     * @param instance_api_version Vulkan version the probing instance was created with.
     * @return Feature set found on the device.
     */
    required_features_t query_device_features(VkPhysicalDevice phys, uint32_t instance_api_version) {
      required_features_t out;

      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(phys, &props);
      out.api_version = effective_vulkan_api_version(instance_api_version, props.apiVersion);
      if (out.api_version < VK_API_VERSION_1_3) {
        return out;  // not ok
      }

      VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  #if VK_HEADER_VERSION >= 304
      VkPhysicalDeviceVulkan14Features f14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
      if (out.api_version >= VK_API_VERSION_1_4) {
        f13.pNext = &f14;
      }
  #endif
      VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      f12.pNext = &f13;
      VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &f12;
      vkGetPhysicalDeviceFeatures2(phys, &f2);

      out.shader_int16 = f2.features.shaderInt16 == VK_TRUE;
      out.storage_buffer_8bit = f12.storageBuffer8BitAccess == VK_TRUE;
      out.subgroup_size_control = f13.subgroupSizeControl == VK_TRUE;
  #if VK_HEADER_VERSION >= 304
      out.vulkan_1_4 = out.api_version >= VK_API_VERSION_1_4;
      out.global_priority_query = out.vulkan_1_4 && f14.globalPriorityQuery == VK_TRUE;
  #endif
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
     * @brief The physical device PyroWave should encode on, matched to Sunshine's capture GPU.
     */
    struct selected_gpu_t {
      bool usable = false;  ///< True when PyroWave can run on the selected device.
      bool has_uuid = false;  ///< True when `uuid` identifies the selected device.
      uint32_t api_version = 0;  ///< Effective Vulkan API version of the selected device.
      bool vulkan_1_4 = false;  ///< True when Vulkan 1.4 features were queried on the device.
      bool global_priority_query = false;  ///< True when Vulkan 1.4 global priority queries are supported.
      pyrowave_uuid uuid {};  ///< Device UUID of the selected device, used to pin PyroWave to it.
      std::string name;  ///< Device name, for logging.
    };

    /**
     * @brief Create the throwaway Vulkan instance used to inspect capture devices.
     *
     * @param instance_api_version Receives the API version the instance was created with.
     * @return Instance handle, or VK_NULL_HANDLE when creation fails.
     */
    VkInstance create_probe_instance(uint32_t &instance_api_version) {
      uint32_t loader_api_version = VK_API_VERSION_1_0;
      if (vkEnumerateInstanceVersion(&loader_api_version) != VK_SUCCESS) {
        loader_api_version = VK_API_VERSION_1_0;
      }
      instance_api_version = clamp_vulkan_api_version(loader_api_version);

      VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
      app.apiVersion = instance_api_version;
      VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
      ci.pApplicationInfo = &app;
      static const char *drm_ext = VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME;
      ci.enabledExtensionCount = 1;
      ci.ppEnabledExtensionNames = &drm_ext;

      VkInstance inst = VK_NULL_HANDLE;
      if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) {
        // Retry without the extension for loaders that do not support it.
        ci.enabledExtensionCount = 0;
        ci.ppEnabledExtensionNames = nullptr;
        if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) {
          return VK_NULL_HANDLE;
        }
      }
      return inst;
    }

    /**
     * @brief Select the physical device PyroWave uses for capture.
     *
     * Prefers the device whose render node matches Sunshine's capture GPU. When no match can be
     * made, returns the first device that satisfies PyroWave's required features.
     *
     * @param inst Probe instance.
     * @param instance_api_version API version the instance was created with.
     * @param features Receives the selected device's feature set.
     * @return Selected physical device, or VK_NULL_HANDLE when none is usable.
     */
    VkPhysicalDevice find_capture_physical_device(VkInstance inst, uint32_t instance_api_version, required_features_t &features) {
      uint32_t count = 0;
      vkEnumeratePhysicalDevices(inst, &count, nullptr);
      std::vector<VkPhysicalDevice> devs(count);
      vkEnumeratePhysicalDevices(inst, &count, devs.data());

      auto usable = [instance_api_version](VkPhysicalDevice phys, required_features_t &out) {
        out = query_device_features(phys, instance_api_version);
        uint32_t family = 0;
        return out.ok() && find_compute_queue_family(phys, family);
      };

      VkPhysicalDevice matched = VK_NULL_HANDLE;
      struct stat node_stat;
      auto render_path = platf::resolve_render_device();
      if (!render_path.empty() && render_path[0] == '/' && stat(render_path.c_str(), &node_stat) == 0) {
        auto target_major = major(node_stat.st_rdev);
        auto target_minor = minor(node_stat.st_rdev);
        for (auto phys : devs) {
          VkPhysicalDeviceDrmPropertiesEXT drm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
          VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
          props2.pNext = &drm;
          vkGetPhysicalDeviceProperties2(phys, &props2);
          if (drm.hasRender && drm.renderMajor == (int64_t) target_major && drm.renderMinor == (int64_t) target_minor) {
            matched = phys;
            break;
          }
        }
      }

      if (matched) {
        features = query_device_features(matched, instance_api_version);
        return matched;
      }

      for (auto phys : devs) {
        required_features_t out;
        if (usable(phys, out)) {
          features = out;
          return phys;
        }
      }
      return VK_NULL_HANDLE;
    }

    /**
     * @brief Query the modifiers a device can sample for a format.
     *
     * @param phys Physical device to query.
     * @param format Vulkan format to query.
     * @return Modifiers whose tiling features include sampled-image support.
     */
    std::vector<uint64_t> sampled_modifiers(VkPhysicalDevice phys, VkFormat format) {
      VkDrmFormatModifierPropertiesListEXT mod_list = {VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
      VkFormatProperties2 fp = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
      fp.pNext = &mod_list;
      vkGetPhysicalDeviceFormatProperties2(phys, format, &fp);
      if (mod_list.drmFormatModifierCount == 0) {
        return {};
      }
      std::vector<VkDrmFormatModifierPropertiesEXT> props(mod_list.drmFormatModifierCount);
      mod_list.pDrmFormatModifierProperties = props.data();
      vkGetPhysicalDeviceFormatProperties2(phys, format, &fp);

      std::vector<uint64_t> out;
      out.reserve(props.size());
      for (const auto &mp : props) {
        if (mp.drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
          out.push_back(mp.drmFormatModifier);
        }
      }
      return out;
    }

    /**
     * @brief Match Sunshine's capture render node to a PyroWave-capable physical device.
     *
     * PyroWave must encode on the same GPU that produced the captured DMA-BUF: on multi-GPU
     * machines the default device may be the other one, and importing across GPUs fails or takes a
     * slow path. The render node Sunshine captures from (`platf::resolve_render_device()`) is
     * matched to a Vulkan device via VK_EXT_physical_device_drm, mirroring the Vulkan encoder.
     * When no match can be made (for example on a loader without the extension), the first device
     * that can run the encoder is used and its UUID is pinned for device creation.
     *
     * @return Selection result.
     */
    selected_gpu_t select_capture_gpu() {
      selected_gpu_t out;

      uint32_t instance_api_version = VK_API_VERSION_1_0;
      VkInstance inst = create_probe_instance(instance_api_version);
      if (!inst) {
        return out;
      }

      required_features_t features;
      auto phys = find_capture_physical_device(inst, instance_api_version, features);
      if (phys) {
        uint32_t family = 0;
        out.usable = features.ok() && find_compute_queue_family(phys, family);

        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(phys, &props);
        out.name = props.deviceName;
        out.api_version = features.api_version;
        out.vulkan_1_4 = features.vulkan_1_4;
        out.global_priority_query = features.global_priority_query;

        VkPhysicalDeviceIDProperties id = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &id;
        vkGetPhysicalDeviceProperties2(phys, &props2);
        std::memcpy(out.uuid.uuid, id.deviceUUID, VK_UUID_SIZE);
        out.has_uuid = true;
      }

      vkDestroyInstance(inst, nullptr);
      return out;
    }

    /**
     * @brief Human-readable name for a Vulkan global queue priority.
     *
     * @param priority Priority to name.
     * @return Static name of the priority level.
     */
    const char *queue_priority_name(VkQueueGlobalPriority priority) {
      switch (priority) {
        case VK_QUEUE_GLOBAL_PRIORITY_LOW_EXT:
          return "low";
        case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT:
          return "medium";
        case VK_QUEUE_GLOBAL_PRIORITY_HIGH_EXT:
          return "high";
        case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_EXT:
          return "realtime";
        default:
          return "unknown";
      }
    }

  #if !defined(__FreeBSD__)
    /**
     * @brief Temporarily owns CAP_SYS_NICE while running a privileged device-creation task.
     *
     * The capability is inherited when the worker thread is created and only needs to be raised
     * into the effective set for the duration of a task; the kernel checks it per calling thread
     * when the Vulkan driver creates its GPU context.
     */
    class cap_sys_nice {
    public:
      cap_sys_nice() {
        caps = cap_get_proc();
        if (!caps) {
          BOOST_LOG(warning) << "PyroWave: privileged worker could not read its capabilities"sv;
          return;
        }

        cap_value_t sys_nice = CAP_SYS_NICE;
        if (cap_set_flag(caps, CAP_EFFECTIVE, 1, &sys_nice, CAP_SET) || cap_set_proc(caps)) {
          // The worker is started before main() drops privileges, so a missing capability means the
          // binary was not granted one (packages install with cap_sys_admin,cap_sys_nice+p).
          BOOST_LOG(warning) << "PyroWave: privileged worker lacks CAP_SYS_NICE; the driver will refuse GPU queue priorities above medium"sv;
          cap_free(caps);
          caps = nullptr;
        }
      }

      ~cap_sys_nice() {
        if (!caps) {
          return;
        }

        cap_value_t sys_nice = CAP_SYS_NICE;
        if (cap_set_flag(caps, CAP_EFFECTIVE, 1, &sys_nice, CAP_CLEAR) || cap_set_proc(caps)) {
          BOOST_LOG(warning) << "PyroWave: failed to drop CAP_SYS_NICE on the privileged worker"sv;
        }
        cap_free(caps);
      }

      cap_sys_nice(const cap_sys_nice &) = delete;
      cap_sys_nice &operator=(const cap_sys_nice &) = delete;

    private:
      cap_t caps = nullptr;  ///< Worker-thread capabilities, or null when unavailable.
    };
  #endif

    /**
     * @brief Reports that the privileged device worker rejected a task.
     */
    class privileged_device_worker_stopped final: public std::runtime_error {
    public:
      using std::runtime_error::runtime_error;  ///< Inherit standard runtime error constructors.
    };

    /**
     * @brief Worker thread that creates PyroWave devices while holding CAP_SYS_NICE.
     *
     * Started during startup, before the rest of the process drops privileges: capabilities are
     * per-thread, so this thread keeps CAP_SYS_NICE and raises it into its effective set around
     * each task. Tasks run one at a time, which also serializes PyroWave's global device-creation
     * state across concurrent sessions.
     */
    class privileged_device_worker {
    public:
      /**
       * @brief Start the worker if it is not running yet.
       */
      static void ensure_started() {
        instance();
      }

      /**
       * @brief Run a task on the privileged worker thread.
       *
       * @tparam F Callable type.
       * @param f Task to run.
       * @return Result of the task.
       * @throws privileged_device_worker_stopped when the worker is shutting down.
       */
      template<class F>
      static auto run(F &&f) -> std::invoke_result_t<F> {
        return instance().run_impl(std::forward<F>(f));
      }

    private:
      static privileged_device_worker &instance() {
        static privileged_device_worker w;
        return w;
      }

      privileged_device_worker():
          thread_ {[this] {
            sigset_t all;
            sigfillset(&all);
            if (pthread_sigmask(SIG_BLOCK, &all, nullptr) != 0) {
              BOOST_LOG(error) << "PyroWave: failed to block signals in privileged device worker"sv;
              queue_.stop();
              return;
            }

            platf::set_thread_name("pyrowave_worker");
            for (;;) {
              auto task = queue_.pop();
              if (!task) {
                break;
              }
              (*task)();
            }
          }} {
      }

      ~privileged_device_worker() {
        queue_.stop();
      }

      template<class F>
      auto run_impl(F &&f) -> std::invoke_result_t<F> {
        using R = std::invoke_result_t<F>;
        auto task = std::make_shared<std::packaged_task<R()>>(
          [f = std::forward<F>(f)]() mutable -> R {
  #if !defined(__FreeBSD__)
            cap_sys_nice nice;
  #endif
            return f();
          }
        );
        auto fut = task->get_future();

        if (!queue_.raise([task]() mutable {
              (*task)();
            })) {
          throw privileged_device_worker_stopped {"privileged_device_worker: task rejected (worker stopping)"};
        }

        return fut.get();
      }

      safe::queue_t<std::function<void()>> queue_ {32, safe::queue_t<std::function<void()>>::overflow_policy_e::reject};  ///< Queued privileged tasks.
      std::jthread thread_;  ///< Worker thread holding CAP_SYS_NICE.
    };

    /**
     * @brief Create a PyroWave device with a high queue priority when the OS grants it.
     *
     * Requests HIGH on the privileged worker thread, where CAP_SYS_NICE lets the kernel accept
     * priorities above medium. Granite downgrades internally to whatever the actual driver allows;
     * the granted priority is read back with pyrowave_device_get_global_priority(). When the worker
     * cannot run the task the request is retried on the calling thread (medium at best), preserving
     * the unprivileged behavior.
     *
     * @param device_uuid Physical device UUID to create the device on, or nullptr for the default.
     * @param device Receives the created device.
     * @return True when a device was created.
     */
    bool create_device_elevated(const pyrowave_uuid *device_uuid, pyrowave_device *device) {
      auto attempt = [device_uuid, device] {
        return pyrowave_create_device_by_compat(0, 0, device_uuid, nullptr, nullptr, VK_QUEUE_GLOBAL_PRIORITY_HIGH_EXT, device) == PYROWAVE_SUCCESS;
      };

      try {
        return privileged_device_worker::run(attempt);
      } catch (const privileged_device_worker_stopped &e) {
        BOOST_LOG(warning) << "PyroWave: privileged device worker unavailable: "sv << e.what();
        return attempt();
      }
    }

  }  // namespace

  uint32_t max_vulkan_api_version() {
    return MAX_SUPPORTED_API_VERSION;
  }

  uint32_t clamp_vulkan_api_version(uint32_t version) {
    if (version < VK_API_VERSION_1_1) {
      return VK_API_VERSION_1_1;
    }
    if (version > MAX_SUPPORTED_API_VERSION) {
      return MAX_SUPPORTED_API_VERSION;
    }
    return version;
  }

  uint32_t effective_vulkan_api_version(uint32_t instance_version, uint32_t device_version) {
    return std::min(instance_version, device_version);
  }

  size_t packet_boundary_for_packet_size(int packet_size) {
    // One datagram carries packet_size + RTP_HEADER_ALLOWANCE - VIDEO_PACKET_HEADER_BYTES bytes of
    // video payload (see videoBroadcastThread's payload_blocksize), and the transport framing adds
    // TRANSPORT_CHUNK_PREFIX_BYTES in front of every chunk. A chunk of packet_size - 20 bytes
    // therefore never needs more than one datagram. stream.cpp static_asserts the header sizes
    // this relies on.
    constexpr int min_packet_size = RTP_HEADER_ALLOWANCE + VIDEO_PACKET_HEADER_BYTES + (int) TRANSPORT_CHUNK_PREFIX_BYTES;
    if (packet_size < min_packet_size) {
      return DEFAULT_PACKET_BOUNDARY;
    }
    return (size_t) (packet_size + RTP_HEADER_ALLOWANCE - VIDEO_PACKET_HEADER_BYTES - (int) TRANSPORT_CHUNK_PREFIX_BYTES);
  }

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
    return select_capture_gpu().usable;
  }

  bool supports_capture_fourcc(uint32_t fourcc) {
    return drm_fourcc_to_vk_format(fourcc).has_value();
  }

  std::map<uint32_t, std::vector<uint64_t>> supported_capture_modifiers() {
    static constexpr std::array<uint32_t, 8> capture_formats = {
      DRM_FORMAT_ARGB8888,
      DRM_FORMAT_XRGB8888,
      DRM_FORMAT_ABGR8888,
      DRM_FORMAT_XBGR8888,
      DRM_FORMAT_ARGB2101010,
      DRM_FORMAT_XRGB2101010,
      DRM_FORMAT_ABGR2101010,
      DRM_FORMAT_XBGR2101010,
    };

    std::map<uint32_t, std::vector<uint64_t>> out;
    uint32_t instance_api_version = VK_API_VERSION_1_0;
    VkInstance inst = create_probe_instance(instance_api_version);
    if (!inst) {
      return out;
    }

    required_features_t features;
    auto phys = find_capture_physical_device(inst, instance_api_version, features);
    uint32_t family = 0;
    if (phys && features.ok() && find_compute_queue_family(phys, family)) {
      for (auto fourcc : capture_formats) {
        auto format = drm_fourcc_to_vk_format(fourcc);
        if (!format) {
          continue;
        }
        auto modifiers = sampled_modifiers(phys, *format);
        if (!modifiers.empty()) {
          out.emplace(fourcc, std::move(modifiers));
        }
      }
    }

    vkDestroyInstance(inst, nullptr);
    return out;
  }

  void ensure_privileged_device_worker_started() {
    privileged_device_worker::ensure_started();
  }

  struct encoder_t::impl_t {
    pyrowave_device pdev = nullptr;  ///< PyroWave device owning the encoder.
    pyrowave_encoder enc = nullptr;  ///< PyroWave encoder for this session's resolution.
    int width = 0;  ///< Encoded luma width.
    int height = 0;  ///< Encoded luma height.
    std::string gpu_name;  ///< Selected PyroWave GPU, for capture-import diagnostics.
    bool ten_bit = false;  ///< True when planes use R16_UNORM containers.
    bool hdr = false;  ///< True to use the BT.2020 PQ color math.
    bool full_range = true;  ///< True for full-range YCbCr planes, false for limited (MPEG) range.
    size_t max_bitstream = 0;  ///< Per-frame bitstream budget in bytes.

    /// Whether the capture-failure diagnostic was already logged for this session.
    bool logged_capture_failure = false;

    // Packetization buffers, reused across frames (see the sizing comment in encode()).
    std::vector<uint8_t> scratch;  ///< Bitstream output of pyrowave_encoder_packetize().
    std::vector<pyrowave_packet> packets;  ///< Packet table returned by pyrowave_encoder_packetize().

    /// Transport chunk boundary in bytes, derived from the client's negotiated packet size.
    size_t packet_boundary = DEFAULT_PACKET_BOUNDARY;

    /// True when frames use record framing (see encoder_t::create()).
    bool record_framing = false;

    /// Video payload bytes per RTP shard for record alignment; 0 when the packet size cannot align.
    size_t record_shard_payload = 0;

    /// Maximum record-framed frame bytes the transport can carry in one frame.
    size_t record_frame_limit = 0;

    /// PyroWave-owned external image (fast path). When null, the manually imported linear image
    /// below is in use instead. The image is owned by the import cache (pimg_cached) or by the
    /// current frame; release_capture() ends its use for the frame either way.
    pyrowave_image pimg = nullptr;

    /// True when PyroWave was created for an async-compute encode queue (high-priority request
    /// accepted). The manual import's same-queue-family assumption does not hold then.
    bool async_encode = false;

    /// True once a capture with no DRM modifier was imported as linear, so the assumption is
    /// reported once per session rather than silently.
    bool logged_implicit_modifier = false;

    /// Identity of an external import that failed, used to retry periodically instead of disabling
    /// the fast path for the rest of the session. A different format/resolution retries at once.
    struct import_attempt_key_t {
      uint64_t fourcc = 0;  ///< DRM fourcc of the captured buffer.
      uint64_t modifier = 0;  ///< DRM modifier of the captured buffer.
      int width = 0;  ///< Buffer width.
      int height = 0;  ///< Buffer height.

      /**
       * @brief Compare two import attempts.
       *
       * @param other Attempt to compare against.
       * @return True when both describe the same capture format and size.
       */
      bool operator==(const import_attempt_key_t &other) const {
        return fourcc == other.fourcc && modifier == other.modifier &&
               width == other.width && height == other.height;
      }
    };

    std::optional<import_attempt_key_t> failed_external_import;  ///< Key of the last failed external import.
    std::optional<import_attempt_key_t> logged_external_import;  ///< Key whose failure was already logged.
    std::optional<uint64_t> logged_unsupported_modifier;  ///< Modifier whose unsupported-layout failure was logged.
    uint32_t external_import_retry_frames = 0;  ///< Frames to wait before retrying the external path.
    static constexpr uint32_t EXTERNAL_IMPORT_RETRY_FRAMES = 30;  ///< Retry interval after a failure.

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
    VkFormat imp_format = VK_FORMAT_UNDEFINED;  ///< Vulkan format of the manual import.

    /// Planes and sampled-image support reported for a (format, modifier) pair, memoized because
    /// query_modifier_properties() enumerates every modifier the device reports and the capture
    /// format is fixed per session.
    struct modifier_properties_cache_t {
      VkFormat format {};  ///< Format the properties were queried for.
      uint64_t modifier = 0;  ///< DRM format modifier the properties were queried for.
      int planes = 0;  ///< Plane count reported for the pair.
      bool sampled = false;  ///< True when the modifier supports sampled images.
    };

    /// Memoized query_modifier_properties() results, see modifier_properties().
    std::vector<modifier_properties_cache_t> modifier_properties_cache;

    /// Number of capture imports kept alive for reuse. Captures cycle through the compositor's
    /// buffer pool, which is a handful of buffers.
    static constexpr size_t import_cache_capacity = 8;

    /// Identity of a captured DMA-BUF for as long as a descriptor to it stays open.
    struct import_cache_key_t {
      uint64_t dev = 0;  ///< st_dev of the captured descriptor.
      uint64_t ino = 0;  ///< st_ino of the captured descriptor.
      uint64_t modifier = 0;  ///< DRM format modifier of the buffer.
      VkFormat format {};  ///< Vulkan format the buffer is imported as.
      int width = 0;  ///< Buffer width.
      int height = 0;  ///< Buffer height.

      /**
       * @brief Compare two capture identities.
       *
       * @param other Identity to compare against.
       * @return True when both describe the same buffer.
       */
      bool operator==(const import_cache_key_t &other) const {
        return dev == other.dev && ino == other.ino && modifier == other.modifier &&
               format == other.format && width == other.width && height == other.height;
      }
    };

    /// One reusable capture import.
    struct import_cache_entry_t {
      import_cache_key_t key {};  ///< Identity of the cached buffer.
      int held_fd = -1;  ///< Descriptor pinning the buffer's identity (dev/ino) while cached.
      pyrowave_image image = nullptr;  ///< PyroWave-owned import handed to the encoder per frame.
      uint64_t last_used = 0;  ///< Import clock tick of the most recent use, for LRU eviction.
    };

    /// Reused imports of the capture's DMA-BUFs. Importing a buffer (image creation, memory
    /// binding, and the driver's address setup) is the expensive part of the external image path,
    /// and a capture buffer comes back frame after frame, so the import is kept and reused.
    std::vector<import_cache_entry_t> import_cache;

    /// Monotonic tick stamped into import cache entries to order them by recency.
    uint64_t import_cache_clock = 0;

    /// True while pimg refers to an import owned by the cache, which must outlive this frame.
    bool pimg_cached = false;

    ~impl_t() {
      if (enc) {
        pyrowave_encoder_destroy(enc);
      }
      // Destroy our GPU resources before the pyrowave device (they live on its VkDevice).
      if (vk_dev) {
        vkDeviceWaitIdle(vk_dev);
        release_capture();
        clear_cached_imports();
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
     * @brief Import a captured DMA-BUF through PyroWave's external image API.
     *
     * This is the fast path. PyroWave creates the image on its own device and acquires queue
     * ownership at encode time, so no per-frame command submission and CPU fence wait is needed
     * (unlike the manual import below). Captures that do not report a modifier are imported as
     * DRM_FORMAT_MOD_LINEAR, the same assumption the manual linear import makes. The image is
     * always in GENERAL layout, per pyrowave.h. On success PyroWave owns the duplicated file
     * descriptor; on failure ownership stays with the caller, which closes it.
     *
     * @param sd Surface descriptor of the captured buffer.
     * @return True on success.
     */
    bool create_pyrowave_image(const egl::surface_descriptor_t &sd) {
      auto vk_format = drm_fourcc_to_vk_format(sd.fourcc);
      if (!vk_format) {
        BOOST_LOG(error) << "PyroWave: unsupported capture DRM fourcc 0x" << std::hex << sd.fourcc << std::dec;
        return false;
      }

      // A capture without an explicit modifier is implicitly laid out; the manual path already
      // treats it as linear, so import it as such through the external image API. The PipeWire
      // backend filters implicit modifiers out for PyroWave sessions, but other capture backends
      // may still hand them to us.
      if (sd.modifier == DRM_FORMAT_MOD_INVALID && !logged_implicit_modifier) {
        logged_implicit_modifier = true;
        BOOST_LOG(info) << "PyroWave: capture has no DRM modifier; assuming a linear layout"sv;
      }
      uint64_t modifier = (sd.modifier == DRM_FORMAT_MOD_INVALID) ? DRM_FORMAT_MOD_LINEAR : sd.modifier;

      int dmabuf_planes = 0;
      for (int i = 0; i < 4 && sd.fds[i] >= 0; ++i) {
        dmabuf_planes++;
      }
      const auto properties = modifier_properties(*vk_format, modifier);
      if (properties.planes <= 0 || !properties.sampled) {
        if (!logged_unsupported_modifier || *logged_unsupported_modifier != modifier) {
          logged_unsupported_modifier = modifier;
          BOOST_LOG(error) << "PyroWave: capture modifier 0x" << std::hex << modifier << std::dec
                           << " is not a sampled-image modifier for "sv << (gpu_name.empty() ? "the selected GPU" : gpu_name)
                           << "; cannot import"sv;
        }
        return false;
      }
      if (properties.planes > dmabuf_planes) {
        if (!logged_unsupported_modifier || *logged_unsupported_modifier != modifier) {
          logged_unsupported_modifier = modifier;
          BOOST_LOG(error) << "PyroWave: capture modifier 0x" << std::hex << modifier << std::dec
                           << " needs "sv << properties.planes << " planes but the buffer has "sv << dmabuf_planes;
        }
        return false;
      }
      const int plane_count = properties.planes;

      int fd = dup(sd.fds[0]);
      if (fd < 0) {
        return false;
      }

      std::array<VkSubresourceLayout, 4> drm_layouts = {};
      for (int i = 0; i < plane_count; ++i) {
        drm_layouts[i].offset = sd.offsets[i];
        drm_layouts[i].rowPitch = sd.pitches[i];
      }

      VkImageDrmFormatModifierExplicitCreateInfoEXT drm_ci = {VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
      drm_ci.drmFormatModifier = modifier;
      drm_ci.drmFormatModifierPlaneCount = plane_count;
      drm_ci.pPlaneLayouts = drm_layouts.data();

      VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      img_ci.pNext = &drm_ci;
      img_ci.imageType = VK_IMAGE_TYPE_2D;
      img_ci.format = *vk_format;
      img_ci.extent = {(uint32_t) sd.width, (uint32_t) sd.height, 1};
      img_ci.mipLevels = 1;
      img_ci.arrayLayers = 1;
      img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
      img_ci.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
      img_ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      img_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

      pyrowave_image_create_info ci {};
      ci.device = pdev;
      ci.external_handle = (pyrowave_os_handle) fd;
      ci.handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
      ci.image_create_info = &img_ci;

      if (pyrowave_image_create(&ci, &pimg) != PYROWAVE_SUCCESS) {
        // A failed import leaves the file descriptor owned by the caller.
        close(fd);
        pimg = nullptr;
        return false;
      }
      return true;
    }

    /**
     * @brief Memoized query of a format/modifier pair's plane count and sampled-image support.
     *
     * @param format Vulkan format of the image.
     * @param modifier DRM format modifier describing the buffer layout.
     * @return Properties reported for the pair; zero planes when the modifier is unsupported.
     */
    modifier_properties_cache_t modifier_properties(VkFormat format, uint64_t modifier) {
      for (const auto &entry : modifier_properties_cache) {
        if (entry.format == format && entry.modifier == modifier) {
          return entry;
        }
      }
      const auto properties = query_modifier_properties(vk_phys, format, modifier);
      modifier_properties_cache.push_back({format, modifier, properties.planes, properties.sampled});
      return modifier_properties_cache.back();
    }

    /**
     * @brief Destroy one cached import: the image and the descriptor pinning its identity.
     *
     * @param entry Entry to destroy.
     */
    static void destroy_cached_import(import_cache_entry_t &entry) {
      if (entry.image) {
        pyrowave_image_destroy(entry.image);
        entry.image = nullptr;
      }
      if (entry.held_fd >= 0) {
        close(entry.held_fd);
        entry.held_fd = -1;
      }
    }

    /**
     * @brief Destroy every cached import. The device must still be alive.
     */
    void clear_cached_imports() {
      for (auto &entry : import_cache) {
        destroy_cached_import(entry);
      }
      import_cache.clear();
    }

    /**
     * @brief Drop the least-recently-used cached import.
     */
    void evict_cached_import() {
      if (import_cache.empty()) {
        return;
      }
      auto lru = std::min_element(import_cache.begin(), import_cache.end(), [](const import_cache_entry_t &a, const import_cache_entry_t &b) {
        return a.last_used < b.last_used;
      });
      destroy_cached_import(*lru);
      import_cache.erase(lru);
    }

    /**
     * @brief Derive the identity of a captured DMA-BUF.
     *
     * The descriptor's device/inode pair identifies the buffer while a descriptor to it stays
     * open, which the cache guarantees by holding one. Captures without a descriptor, or whose
     * stat fails, cannot be cached.
     *
     * @param sd Surface descriptor of the captured buffer.
     * @param key Receives the identity.
     * @return True when the buffer can be identified.
     */
    static bool capture_import_key(const egl::surface_descriptor_t &sd, import_cache_key_t &key) {
      struct stat st {};
      if (sd.fds[0] < 0 || fstat(sd.fds[0], &st) != 0) {
        return false;
      }
      auto vk_format = drm_fourcc_to_vk_format(sd.fourcc);
      if (!vk_format) {
        return false;
      }
      key = {};
      key.dev = (uint64_t) st.st_dev;
      key.ino = (uint64_t) st.st_ino;
      key.modifier = (sd.modifier == DRM_FORMAT_MOD_INVALID) ? DRM_FORMAT_MOD_LINEAR : sd.modifier;
      key.format = *vk_format;
      key.width = sd.width;
      key.height = sd.height;
      return true;
    }

    /**
     * @brief Import the captured buffer through PyroWave, reusing the import when it is cached.
     *
     * @param sd Surface descriptor of the captured buffer.
     * @return True on success; the import is owned by the cache, or by this frame when the buffer
     *         cannot be identified, and ends with release_capture() either way.
     */
    bool import_captured(const egl::surface_descriptor_t &sd) {
      import_cache_key_t key {};
      if (!capture_import_key(sd, key)) {
        return create_pyrowave_image(sd);
      }

      import_cache_clock++;
      for (auto &entry : import_cache) {
        if (entry.key == key) {
          entry.last_used = import_cache_clock;
          pimg = entry.image;
          pimg_cached = true;
          return true;
        }
      }

      const int held_fd = dup(sd.fds[0]);
      if (!create_pyrowave_image(sd)) {
        if (held_fd >= 0) {
          close(held_fd);
        }
        return false;
      }

      if (held_fd < 0) {
        return true;  // The identity could not be pinned, so this frame owns the import.
      }

      if (import_cache.size() >= import_cache_capacity) {
        evict_cached_import();
      }
      import_cache.push_back({key, held_fd, pimg, import_cache_clock});
      pimg_cached = true;
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
      imp_format = VK_FORMAT_UNDEFINED;
    }

    /**
     * @brief Release whichever capture import is currently held.
     *
     * Imports owned by the reuse cache outlive the frame and stay cached; only one-shot imports
     * (uncacheable external imports and manual imports) are destroyed here.
     */
    void release_capture() {
      if (pimg_cached) {
        pimg_cached = false;
        pimg = nullptr;  // Still owned by the import cache.
      } else if (pimg) {
        pyrowave_image_destroy(pimg);
        pimg = nullptr;
      }
      destroy_import();
    }

    /**
     * @brief Make the captured DMA-BUF sampleable by the GPU encode.
     *
     * Captures go through PyroWave's external image path, which lets PyroWave manage the image and
     * its queue ownership. Importing a buffer is the expensive part of that path and captures cycle
     * through a small buffer pool, so imports are reused, see import_captured(). When the external
     * path fails, captures fall back to the manual import, but only when PyroWave encodes on the
     * default (graphics/compute) queue: an async-compute encode family would not own an image
     * transitioned on the manual import's queue, so the fallback is refused in that case rather
     * than sampling across queue families.
     *
     * @param desc Captured image descriptor.
     * @return True on success.
     */
    bool prepare_capture(const egl::img_descriptor_t &desc) {
      if (!supports_capture_fourcc(desc.sd.fourcc)) {
        if (!logged_external_import || logged_external_import->fourcc != desc.sd.fourcc) {
          logged_external_import = import_attempt_key_t {desc.sd.fourcc, desc.sd.modifier, desc.sd.width, desc.sd.height};
          BOOST_LOG(error) << "PyroWave: capture format 0x" << std::hex << desc.sd.fourcc << std::dec
                           << " has no Vulkan mapping PyroWave can sample"sv;
        }
        return false;
      }

      const import_attempt_key_t key {desc.sd.fourcc, desc.sd.modifier, desc.sd.width, desc.sd.height};
      const bool retry_external = !failed_external_import || !(*failed_external_import == key) || external_import_retry_frames == 0;
      if (retry_external) {
        if (import_captured(desc.sd)) {
          failed_external_import.reset();
          external_import_retry_frames = 0;
          return true;
        }

        // Do not disable the fast path for the rest of the session: an import can fail transiently
        // while the producer is still setting up, or when the compositor switches buffers. Retry
        // after a short interval, and log the failure once per capture format.
        failed_external_import = key;
        external_import_retry_frames = EXTERNAL_IMPORT_RETRY_FRAMES;
        if (!logged_external_import || !(*logged_external_import == key)) {
          logged_external_import = key;
          BOOST_LOG(error) << "PyroWave: external DMA-BUF import failed (fourcc=0x" << std::hex << desc.sd.fourcc
                           << ", modifier=0x" << desc.sd.modifier << std::dec << ", "sv << desc.sd.width << 'x' << desc.sd.height
                           << ") on "sv << (gpu_name.empty() ? "the selected GPU" : gpu_name)
                           << "; retrying every "sv << EXTERNAL_IMPORT_RETRY_FRAMES << " frames"sv;
        }
      } else {
        --external_import_retry_frames;
      }

      if (async_encode) {
        return false;
      }
      return prepare_dmabuf(desc);
    }

    /**
     * @brief Import a captured DMA-BUF as an RGB VkImage on PyroWave's device.
     *
     * @param sd Surface descriptor of the captured buffer.
     * @return True on success.
     */
    bool import_dmabuf(const egl::surface_descriptor_t &sd) {
      destroy_import();

      auto vk_format = drm_fourcc_to_vk_format(sd.fourcc);
      if (!vk_format) {
        BOOST_LOG(error) << "PyroWave: unsupported capture DRM fourcc 0x" << std::hex << sd.fourcc << std::dec;
        return false;
      }

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

      if (sd.modifier != DRM_FORMAT_MOD_INVALID) {
        int dmabuf_planes = 0;
        for (int i = 0; i < 4 && sd.fds[i] >= 0; ++i) {
          dmabuf_planes++;
        }
        const auto properties = modifier_properties(*vk_format, sd.modifier);
        if (properties.planes <= 0 || !properties.sampled) {
          if (!logged_unsupported_modifier || *logged_unsupported_modifier != sd.modifier) {
            logged_unsupported_modifier = sd.modifier;
            BOOST_LOG(error) << "PyroWave: capture modifier 0x" << std::hex << sd.modifier << std::dec
                             << " is not a sampled-image modifier for "sv << (gpu_name.empty() ? "the selected GPU" : gpu_name)
                             << "; cannot import"sv;
          }
          close(fd);
          return false;
        }
        if (properties.planes > dmabuf_planes) {
          if (!logged_unsupported_modifier || *logged_unsupported_modifier != sd.modifier) {
            logged_unsupported_modifier = sd.modifier;
            BOOST_LOG(error) << "PyroWave: capture modifier 0x" << std::hex << sd.modifier << std::dec
                             << " needs "sv << properties.planes << " planes but the buffer has "sv << dmabuf_planes;
          }
          close(fd);
          return false;
        }
        for (int i = 0; i < properties.planes; ++i) {
          drm_layouts[i].offset = sd.offsets[i];
          drm_layouts[i].rowPitch = sd.pitches[i];
        }
        drm_ci.drmFormatModifier = sd.modifier;
        drm_ci.drmFormatModifierPlaneCount = properties.planes;
        drm_ci.pPlaneLayouts = drm_layouts.data();
        ext_ci.pNext = &drm_ci;
        tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
      } else {
        tiling = VK_IMAGE_TILING_LINEAR;
      }

      VkImageCreateInfo img_ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      img_ci.pNext = &ext_ci;
      img_ci.imageType = VK_IMAGE_TYPE_2D;
      img_ci.format = *vk_format;
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
        close(fd);
        return false;
      }
      if (vkBindImageMemory(vk_dev, imp_image, imp_mem, 0) != VK_SUCCESS) {
        BOOST_LOG(error) << "PyroWave: dmabuf vkBindImageMemory failed";
        vkFreeMemory(vk_dev, imp_mem, nullptr);
        imp_mem = VK_NULL_HANDLE;
        vkDestroyImage(vk_dev, imp_image, nullptr);
        imp_image = VK_NULL_HANDLE;
        return false;
      }
      imp_format = *vk_format;
      return true;
    }

    /**
     * @brief Import the captured DMA-BUF and make it readable by the GPU encode.
     *
     * The buffer is imported as an RGB VkImage on PyroWave's device and transitioned to
     * VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, taking ownership of the externally produced
     * contents from VK_QUEUE_FAMILY_EXTERNAL (the same acquire the Vulkan encoder performs on its
     * imported captures). The transition is then waited on. PyroWave samples the image later, when
     * its encode commands run, so the caller must keep the import alive until the encoder's fence
     * has been waited on and release it with destroy_import().
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
      // Acquire the capture buffer from its external producer into the queue family the encode
      // samples on, matching the Vulkan encoder's imported captures. The barrier is submitted on
      // the queue PyroWave itself uses for this device (a MEDIUM-priority device encodes on the
      // graphics queue), so the acquisition covers the sampling that follows.
      mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
      mb.dstQueueFamilyIndex = vk_family;
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
     * @return True on success.
     */
    bool fill_scaled_info(const egl::img_descriptor_t &desc, VkRect2D &crop, pyrowave_scaled_encode_info &info) {
      const auto &sd = desc.sd;
      info = {};
      if (pimg) {
        // PyroWave's external image: the view (and its GENERAL layout) comes from the image itself.
        if (pyrowave_image_get_image_view(pimg, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_USAGE_SAMPLED_BIT, &info.view) != PYROWAVE_SUCCESS) {
          BOOST_LOG(error) << "PyroWave: external image view creation failed";
          return false;
        }
      } else {
        info.view.image = imp_image;
        info.view.width = (uint32_t) sd.width;
        info.view.height = (uint32_t) sd.height;
        info.view.image_format = imp_format;
        info.view.view_format = info.view.image_format;
        info.view.mip_level = 0;
        info.view.layer = 0;
        info.view.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        info.view.swizzle = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.view.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }

      // Linux captures arrive display-referred: sRGB for SDR, PQ/BT.2020 for HDR. The scaler
      // re-emits the same encoding, with the YCbCr range the client negotiated (the planes feed
      // the client's CSC directly).
      info.input_color_space = hdr ? VK_COLOR_SPACE_HDR10_ST2084_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      info.output_color_space = info.input_color_space;
      info.ycbcr_range = full_range ? VK_SAMPLER_YCBCR_RANGE_ITU_FULL : VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
      info.ycbcr_range_bit_depth = ten_bit ? 10 : 8;
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
      return true;
    }
  };

  std::unique_ptr<encoder_t> encoder_t::create(int width, int height, int bitrate_kbps, int frame_rate, bool yuv444, bool ten_bit, bool hdr, bool full_range, int packet_size, bool record_framing, std::optional<::pyrowave::policy::wire_budget_t> wire_budget) {
    // The v1 sequence header stores width_minus_1 and height_minus_1 in 14 bits, so dimensions
    // above 16384 cannot be represented. Reject them before the even-rounding below can hide an
    // out-of-range request.
    constexpr int MAX_DIMENSION = 1 << 14;
    if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
      BOOST_LOG(error) << "PyroWave: unsupported dimensions "sv << width << 'x' << height
                       << " (maximum "sv << MAX_DIMENSION << 'x' << MAX_DIMENSION << ')';
      return nullptr;
    }

    // 4:2:0 requires even dimensions. The bitstream and the Xbox decoder both allow odd
    // dimensions for 4:4:4, and the client initializes its block layout from the requested size,
    // so do not round there.
    if (!yuv444) {
      width &= ~1;
      height &= ~1;
    }
    if (width <= 0 || height <= 0) {
      return nullptr;
    }

    // The client controls the configured bitrate. Reject non-positive values before converting
    // them to a per-frame budget, which would otherwise become a huge size_t and reserve an
    // enormous scratch buffer.
    if (bitrate_kbps <= 0) {
      BOOST_LOG(error) << "PyroWave: invalid bitrate "sv << bitrate_kbps << " kbps"sv;
      return nullptr;
    }

    auto self = std::unique_ptr<encoder_t>(new encoder_t());
    self->impl = std::make_unique<impl_t>();
    auto &impl = *self->impl;

    impl.width = width;
    impl.height = height;
    impl.ten_bit = ten_bit;
    impl.hdr = hdr && ten_bit;  // HDR color math only makes sense in 10-bit containers
    impl.full_range = full_range;
    impl.packet_boundary = packet_boundary_for_packet_size(packet_size);
    impl.record_framing = record_framing;
    if (record_framing) {
      impl.record_shard_payload = ::pyrowave::policy::shard_payload_bytes(packet_size);
      if (impl.record_shard_payload == 0) {
        // Records can only be aligned to shards when the negotiated payload is word-aligned and
        // large enough for the frame header plus a minimal padding record. Otherwise emit the
        // length-prefixed framing, which the client can still parse without partial recovery.
        BOOST_LOG(warning) << "PyroWave: packet size "sv << packet_size
                           << " cannot align record framing; using length-prefixed framing"sv;
        impl.record_framing = false;
      }
    }

    // Per-frame byte budget from bitrate. Intra-only: bitrate / fps bytes per frame.
    if (frame_rate <= 0) {
      frame_rate = 60;
    }
    int64_t bits_per_frame = (int64_t) bitrate_kbps * 1000 / frame_rate;
    const auto wire_bytes_per_frame = (size_t) (bits_per_frame / 8);
    if (wire_budget) {
      // The client's bitrate is a wire budget: deduct packet headers and the reserved critical-block
      // parity before handing bytes to the codec. The budget is already net of audio and control.
      impl.max_bitstream = ::pyrowave::policy::frame_payload_bytes(wire_bytes_per_frame, *wire_budget);
      if (impl.max_bitstream == 0) {
        BOOST_LOG(error) << "PyroWave: " << wire_bytes_per_frame << " wire bytes/frame cannot carry a frame";
        return nullptr;
      }
    } else {
      impl.max_bitstream = wire_bytes_per_frame;
      if (impl.max_bitstream < 4096) {
        impl.max_bitstream = 4096;
      }
    }
    if (impl.record_framing) {
      // The video header can describe only four FEC blocks of 1023 data shards. Keep the codec
      // budget inside that transport capacity so record padding cannot push a frame past it.
      const auto transport_cap = ::pyrowave::policy::max_bitstream_bytes(packet_size, false, true);
      if (transport_cap) {
        impl.max_bitstream = std::min(impl.max_bitstream, transport_cap);
      }
      if (wire_budget) {
        // Padding records share the frame's wire allowance. Bound them to the codec budget so a
        // padded frame cannot grow past the budget that was reserved for it.
        impl.record_frame_limit = impl.max_bitstream;
      } else {
        impl.record_frame_limit = ::pyrowave::policy::max_frame_bytes(packet_size, true);
      }
    }

    // Let PyroWave create and own its Vulkan (Granite) device. The encoder does not need to share
    // Sunshine's device, and this avoids the raw-device adoption path, which deadlocked inside
    // Granite's cross-queue submission sync when handed a single externally-created queue.
    //
    // Pin PyroWave to the GPU the probe selected. When the capture render node can be matched,
    // that is the GPU that produced the DMA-BUF; otherwise it is the first usable device. Passing
    // its UUID keeps PyroWave from selecting a different physical device than the one the probe
    // validated (the upstream C API accepts Vulkan 1.1 devices, which the probe rejects).
    auto gpu = select_capture_gpu();
    if (!gpu.usable) {
      BOOST_LOG(error) << "PyroWave: no PyroWave-capable Vulkan device for the capture GPU";
      return nullptr;
    }
    impl.gpu_name = gpu.name;
    const pyrowave_uuid *uuid = gpu.has_uuid ? &gpu.uuid : nullptr;

    // Request a high GPU scheduling priority for the PyroWave device, so the encode keeps its
    // latency budget while a game saturates the GPU. The request runs on the privileged worker
    // thread: Linux only grants priorities above medium to a caller with CAP_SYS_NICE, and the
    // capability is per-thread, so the session thread cannot use it after startup. Granite
    // downgrades the request internally to whatever the driver actually grants (read back with
    // pyrowave_device_get_global_priority()), and falls back to the default-priority overload
    // when even the elevated request is refused outright. Requesting above medium makes PyroWave
    // encode on an async compute queue (regardless of what the OS grants), which the manual import
    // fallback cannot safely transition against.
    //
    // Granite only raises the granted priority above medium when Vulkan 1.4's globalPriorityQuery
    // feature is available, but the request above medium also selects PyroWave's async compute
    // queue, so the elevated path is used even when the feature is absent.
    VkQueueGlobalPriority priority = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT;
    if (create_device_elevated(uuid, &impl.pdev)) {
      impl.async_encode = true;
      priority = pyrowave_device_get_global_priority(impl.pdev);
    } else if (pyrowave_create_device_by_compat(0, 0, uuid, nullptr, nullptr, VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT, &impl.pdev) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: pyrowave_create_device_by_compat failed";
      return nullptr;
    }
    if (gpu.has_uuid) {
      BOOST_LOG(info) << "PyroWave: encoding on " << gpu.name << " (Vulkan "
                      << VK_VERSION_MAJOR(gpu.api_version) << '.' << VK_VERSION_MINOR(gpu.api_version)
                      << (gpu.vulkan_1_4 ? ", 1.4 features" : "")
                      << (gpu.global_priority_query ? ", global priority query" : "") << ')';
    }
    // How much of the requested priority the OS actually granted. Below high, the encode competes
    // with game GPU work at the same priority; the privileged worker logs when CAP_SYS_NICE (which
    // packaged installs apply to the binary) could not be used, and the driver may also refuse a
    // priority the queue family does not advertise.
    BOOST_LOG(info) << "PyroWave: GPU queue priority " << queue_priority_name(priority);

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
                    << (full_range ? " full-range" : " limited-range")
                    << " budget " << impl.max_bitstream << " bytes/frame"
                    << (impl.record_framing ? ", record framing" : "");
    if (wire_budget) {
      BOOST_LOG(info) << "PyroWave wire budget: " << wire_bytes_per_frame << " bytes/frame, FEC "
                      << wire_budget->critical_fec_percentage << "%, " << wire_budget->packet_overhead_bytes
                      << " overhead bytes/packet, record padding limit " << impl.record_frame_limit;
    }
    return self;
  }

  bool encoder_t::record_framing() const {
    return impl->record_framing;
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
    if (!impl.prepare_capture(*desc)) {
      log_capture_failure("PyroWave: dmabuf import failed");
      return -1;
    }

    VkRect2D crop {};
    pyrowave_scaled_encode_info scaling {};
    if (!impl.fill_scaled_info(*desc, crop, scaling)) {
      impl.release_capture();
      return -1;
    }

    // Captures imported through PyroWave's external image API are acquired and released by the
    // encode submission itself, so no explicit import sync is needed. Imports can outlive the frame
    // (the import cache reuses them), so the release matters: it hands the image back to its
    // external owner, which is the state the next frame's acquire expects (the external image
    // contract in pyrowave.h). The manual linear import path already waited on its transition, so
    // it passes neither.
    pyrowave_gpu_external_reference ext_ref {};
    pyrowave_gpu_sync_operation acquire {};
    pyrowave_gpu_sync_operation release {};
    bool external_image = impl.pimg != nullptr;
    if (external_image) {
      ext_ref.image = impl.pimg;
      ext_ref.queue_family_index = VK_QUEUE_FAMILY_EXTERNAL;
      acquire.images = &ext_ref;
      acquire.num_images = 1;
      release.images = &ext_ref;
      release.num_images = 1;
    }

    pyrowave_rate_control rc {};
    rc.maximum_bitstream_size = impl.max_bitstream;

    if (pyrowave_encoder_encode_gpu_scaled(impl.enc, external_image ? &acquire : nullptr, external_image ? &release : nullptr, &scaling, &rc) != PYROWAVE_SUCCESS) {
      impl.release_capture();
      BOOST_LOG(error) << "PyroWave: encode_gpu_scaled failed";
      return -1;
    }

    // Number of leading packets that carry the coarsest wavelet bands, for the RTP layer's
    // asymmetric FEC. Computed before packetizing; the packing is deterministic and identical.
    // This waits for the GPU encode's fence, which also means the captured buffer is no longer in
    // use once it returns, so the import can be released.
    // Record framing packetizes the whole frame into one packet, so it derives the critical head
    // from the record layout instead (see write_record_frame()).
    const size_t packetize_boundary = impl.record_framing ? std::numeric_limits<size_t>::max() : impl.packet_boundary;
    size_t critical_packets = 0;
    if (!impl.record_framing && pyrowave_encoder_compute_num_critical_packets(impl.enc, FEC_PROTECTED_BANDS, packetize_boundary, 0, &critical_packets) != PYROWAVE_SUCCESS) {
      critical_packets = 0;
    }

    impl.release_capture();

    size_t num_packets = 0;
    if (pyrowave_encoder_compute_num_packets(impl.enc, packetize_boundary, &num_packets) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: compute_num_packets failed";
      return -1;
    }

    // Size the bitstream buffer from the rate-control budget, not the per-packet worst case:
    // pyrowave's packetize() only closes a packet after the block that overflows it, so a packet
    // can exceed packet_boundary by one block (up to 4097 words; payload_words is a 12-bit field),
    // and packetize() does not bounds-check the output buffer in release builds. The whole frame
    // is still capped by maximum_bitstream_size, so the budget plus one block of slack is enough.
    // Buffers are reused across frames (grow-only). Sizing per packet instead would reserve tens
    // of MiB at 4K high bitrates.
    size_t scratch_bound = 4096 + impl.max_bitstream + MAX_BLOCK_BYTES;
    if (impl.scratch.size() < scratch_bound) {
      impl.scratch.resize(scratch_bound);
    }
    if (impl.packets.size() < num_packets) {
      impl.packets.resize(num_packets);
    }
    size_t out_packets = 0;
    if (pyrowave_encoder_packetize(impl.enc, impl.packets.data(), packetize_boundary, &out_packets, impl.scratch.data(), impl.scratch.size()) != PYROWAVE_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: packetize failed";
      return -1;
    }

    // packetize() leaves the sequence header's video-usability fields at their defaults (full-range
    // BT.709). The scaler emits the range the client negotiated, and HDR is BT.2020/PQ, so update
    // those bits to describe the planes accurately. The sequence header is the first record of the
    // first packet.
    if (out_packets > 0 && impl.packets[0].offset == 0 && impl.packets[0].size >= sizeof(PyroWave::BitstreamSequenceHeader)) {
      PyroWave::BitstreamSequenceHeader header;
      std::memcpy(&header, impl.scratch.data(), sizeof(header));
      header.color_primaries = impl.hdr;
      header.transfer_function = impl.hdr;
      header.ycbcr_transform = impl.hdr;
      header.ycbcr_range = !impl.full_range;
      std::memcpy(impl.scratch.data(), &header, sizeof(header));
    }

    if (impl.record_framing) {
      // Record framing expects one bitstream holding the sequence header and every block record.
      // write_record_frame() then orders the coarsest wavelet level first, aligns records to RTP
      // shards with padding records, and reports where the critical prefix ends.
      if (out_packets != 1 || impl.packets[0].offset != 0) {
        BOOST_LOG(error) << "PyroWave: record framing needs one bitstream packet, got " << out_packets;
        return -1;
      }
      out.clear();
      const auto stats = ::pyrowave::policy::write_record_frame(
        std::span<const uint8_t>(impl.scratch.data(), impl.packets[0].size),
        impl.record_shard_payload,
        out,
        impl.record_frame_limit
      );
      if (!stats) {
        BOOST_LOG(error) << "PyroWave: record frame layout failed";
        return -1;
      }
      head_bytes = stats->critical_bytes;
      return 0;
    }

    // PyroWave splits a frame into several independently-decodable packets, each of which must be
    // pushed separately on the decode side. Sunshine's RTP layer, however, ships one opaque payload
    // per frame and reassembles it as a single buffer. So we frame the PyroWave packets ourselves:
    //
    //   [u32 packet_count] { [u32 size] [size bytes] } * packet_count
    //
    // The Moonlight PyroWave decoder parses this framing and re-pushes each PyroWave packet.

    // The loss-critical head ends where the coarsest bands end. A frame that is entirely critical
    // (or too small to split) has no tail to leave bare, so it keeps the even split (head_bytes 0).
    head_bytes = 0;
    if (critical_packets > 0 && critical_packets < out_packets) {
      head_bytes = 4 + critical_packets * 4;  // Packet count field and packet size fields.
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

    // Reserve the exact framed size before writing. The caller hands in a fresh vector every
    // frame, so without the reservation the per-packet inserts would grow it geometrically and
    // copy the whole bitstream several times over in the process.
    size_t framed_size = 4 + out_packets * 4;
    for (size_t i = 0; i < out_packets; i++) {
      framed_size += impl.packets[i].size;
    }
    out.reserve(framed_size);

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
