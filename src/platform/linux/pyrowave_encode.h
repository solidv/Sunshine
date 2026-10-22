/**
 * @file src/platform/linux/pyrowave_encode.h
 * @brief PyroWave (Vulkan wavelet, intra-only) host encoder for Sunshine.
 *
 * PyroWave does not ride the avcodec/nvenc encoder_t abstraction: it owns its own Vulkan 1.3+
 * device and emits an already-packetized byte bitstream that Sunshine ships as packet_raw_generic.
 * When the loader, driver, and GPU expose Vulkan 1.4, Granite creates that device with 1.4
 * semantics and enables the corresponding core features.
 * Captured DMA-BUF frames are imported into PyroWave's Vulkan device and fed straight to its GPU
 * scaler/encoder, which converts RGB->YUV and scales into the stream resolution in a single pass,
 * so the codec requires a DMA-BUF capture (the `vulkan` capture/encode backend). Captures go
 * through PyroWave's external image API (no per-frame CPU import submission), with a manual linear
 * import as fallback when PyroWave is not on an async-compute queue.
 *
 * The device is pinned to Sunshine's capture GPU (matched by render node) so the DMA-BUF is
 * imported on the GPU that produced it, and a high-priority queue is requested to keep encode
 * latency low while the game saturates the GPU.
 *
 * The whole translation unit is compiled only when SUNSHINE_ENABLE_PYROWAVE is defined.
 */
#pragma once

#ifdef SUNSHINE_ENABLE_PYROWAVE

  #include <cstdint>
  #include <map>
  #include <memory>
  #include <optional>
  #include <vector>

  // local includes
  #include "src/pyrowave_policy.h"

namespace platf {
  struct img_t;
}

namespace platf::pyrowave {

  /**
   * @brief Center-crop rectangle used to match a capture to the stream's aspect ratio.
   */
  struct crop_rect_t {
    int x = 0;  ///< Left offset in source pixels (even).
    int y = 0;  ///< Top offset in source pixels (even).
    int width = 0;  ///< Cropped width in source pixels (even).
    int height = 0;  ///< Cropped height in source pixels (even).
    bool needed = false;  ///< False when the source already matches the stream aspect ratio.
  };

  /**
   * @brief Compute the center-crop that matches a capture to the stream's aspect ratio.
   *
   * PyroWave's scaler always fills the encoder frame with the (optionally cropped) source, so a
   * capture whose aspect ratio differs from the negotiated stream would otherwise be stretched.
   * Crop the source symmetrically to the stream aspect instead, aligned to even pixels for 4:2:0.
   *
   * @param src_width Captured frame width in pixels.
   * @param src_height Captured frame height in pixels.
   * @param width Stream (encoder) width in pixels.
   * @param height Stream (encoder) height in pixels.
   * @return Crop rectangle; `needed` is false when the whole frame can be scaled as-is.
   */
  crop_rect_t compute_crop_rect(int src_width, int src_height, int width, int height);

  /**
   * @brief Start the privileged worker that creates PyroWave devices.
   *
   * Linux only grants queue priorities above medium to a caller with CAP_SYS_NICE (or root), and
   * capabilities are per-thread. This must be called while the process still holds the capability,
   * i.e. during startup before the engine drops its privileges; the worker then keeps it and
   * creates each session's PyroWave device on request. Safe to call more than once.
   */
  void ensure_privileged_device_worker_started();

  /**
   * @brief Highest Vulkan core version this integration probes and understands.
   *
   * PyroWave requires Vulkan 1.3 features; when the build headers and the runtime expose Vulkan
   * 1.4, the integration also queries the 1.4 feature set so the probe matches the device Granite
   * will create.
   *
   * @return Highest supported `VK_API_VERSION_*` value.
   */
  uint32_t max_vulkan_api_version();

  /**
   * @brief Clamp a loader/instance Vulkan version to the range this integration understands.
   *
   * @param version Version reported by `vkEnumerateInstanceVersion`.
   * @return @p version clamped to [`VK_API_VERSION_1_1`, max_vulkan_api_version()].
   */
  uint32_t clamp_vulkan_api_version(uint32_t version);

  /**
   * @brief Effective Vulkan device version for an instance and physical device pair.
   *
   * A device cannot use a core version newer than the instance that created it, so this mirrors
   * Granite's `min(instance_api_version, physical_device_api_version)` selection.
   *
   * @param instance_version Version the Vulkan instance was created with.
   * @param device_version Version reported by the physical device.
   * @return The lower of the two versions.
   */
  uint32_t effective_vulkan_api_version(uint32_t instance_version, uint32_t device_version);

  /**
   * @brief Probe for a PyroWave-capable Vulkan device on Sunshine's capture GPU.
   *
   * Creates a throwaway Vulkan instance at the highest version the loader supports (up to
   * max_vulkan_api_version()), matches the render node Sunshine captures from to a physical
   * device (VK_EXT_physical_device_drm), and checks the compute features PyroWave's encoder
   * requires (subgroup size control, shaderInt16, storageBuffer8BitAccess). On a Vulkan 1.4
   * device the probe also records 1.4 capabilities such as global priority queries. When the
   * render node cannot be matched, any device satisfies the probe, matching the encoder's
   * default-device fallback. Cheap enough to call from the encoder-probe path; the result gates
   * whether SCM_PYROWAVE is advertised.
   *
   * @return True when the capture GPU (or any device, when unmatched) can run the PyroWave encoder.
   */
  bool validate();

  /**
   * @brief Report whether PyroWave can import a captured DRM fourcc.
   *
   * Formats without a direct Vulkan equivalent (notably the 10-bit formats with two low alpha
   * bits, `DRM_FORMAT_BGRA1010102` and `DRM_FORMAT_RGBA1010102`) cannot be sampled by PyroWave's
   * scaler without an explicit repack. Display backends use this to avoid offering those formats
   * to the compositor for a PyroWave session.
   *
   * @param fourcc DRM fourcc of the captured buffer.
   * @return True when the format has a Vulkan mapping PyroWave can sample.
   */
  bool supports_capture_fourcc(uint32_t fourcc);

  /**
   * @brief Query DMA-BUF modifiers the selected PyroWave GPU can sample for capture formats.
   *
   * Creates a throwaway Vulkan instance, selects the same physical device PyroWave uses for the
   * capture render node, and returns only modifiers whose tiling features include sampled-image
   * support. Display backends intersect their EGL modifier list with this result so the compositor
   * never negotiates a modifier PyroWave cannot import.
   *
   * @return Map from DRM fourcc to sampled-image modifiers; formats with no supported modifiers
   *         are absent. Empty when no PyroWave-capable device can be selected.
   */
  std::map<uint32_t, std::vector<uint64_t>> supported_capture_modifiers();

  /**
   * @brief Compute PyroWave's transport chunk boundary for a negotiated packet size.
   *
   * A PyroWave frame is framed into `[u32 size][packet]` chunks so the decoder can push each packet
   * separately. Sizing the chunk to the negotiated RTP payload keeps one chunk within one datagram,
   * which cuts the per-chunk framing overhead and decoder pushes compared with the fixed 1024-byte
   * default. Falls back to 1024 for unknown or degenerate packet sizes.
   *
   * @param packet_size Negotiated maximum packet payload (`x-nv-video[0].packetSize`), or 0.
   * @return Chunk boundary in bytes.
   */
  size_t packet_boundary_for_packet_size(int packet_size);

  /**
   * @brief One PyroWave encode session bound to a stream's width/height.
   *
   * Owns a dedicated Vulkan device plus a pyrowave_encoder. The device is pinned to the capture
   * GPU when its render node can be matched, and a high-priority queue is requested on the
   * privileged worker (the granted priority may be lower when the OS or driver refuses).
   * Not thread-safe: the caller must serialize encode() calls, matching the pyrowave encoder
   * contract.
   */
  class encoder_t {
  public:
    ~encoder_t();

    /**
     * @brief Create a session for the given frame dimensions.
     *
     * @param width Encoded luma width (made even for 4:2:0).
     * @param height Encoded luma height (made even for 4:2:0).
     * @param bitrate_kbps Negotiated target bitrate in kbps, mapped to a per-frame byte budget.
     * @param frame_rate Frames per second, used with the bitrate to size the per-frame budget.
     * @param yuv444 True for 4:4:4 chroma (full-resolution Cb/Cr), false for 4:2:0.
     * @param ten_bit True when a 10-bit profile was negotiated: planes use R16_UNORM containers
     *                (PyroWave is depth-agnostic; the container just must match the client's).
     * @param hdr True to encode HDR content: BT.2020 NCL on PQ-encoded RGB (requires ten_bit
     *            and an HDR display; else SDR BT.709 is used).
     * @param full_range True to encode full-range YCbCr, false for limited (MPEG) range. The
     *                   scaler's RGB->YUV conversion uses this, matching the client's negotiated
     *                   color range.
     * @param packet_size Negotiated maximum packet payload (`x-nv-video[0].packetSize`), used to
     *                    size the transport chunk boundary; 0 keeps the default.
     * @param record_framing True to emit record framing, which PyroWave-aware clients parse with
     *                       in-band padding records and partial-frame recovery. False emits the
     *                       length-prefixed compatibility framing.
     * @param wire_budget Transport cost model for record framing. When set, @p bitrate_kbps is
     *                    treated as the total wire allowance and the encoder budget is reduced by
     *                    packet headers and the reserved critical FEC parity. Record padding is
     *                    also bounded by the resulting frame budget.
     * @return Session on success, nullptr on failure.
     */
    static std::unique_ptr<encoder_t> create(int width, int height, int bitrate_kbps, int frame_rate, bool yuv444, bool ten_bit, bool hdr, bool full_range, int packet_size, bool record_framing = false, std::optional<::pyrowave::policy::wire_budget_t> wire_budget = std::nullopt);

    /**
     * @brief Encode one captured frame into a PyroWave bitstream.
     *
     * Requires a DMA-BUF captured image. Imports the buffer into PyroWave's Vulkan device and runs
     * the synchronous GPU encode directly on it: PyroWave's scaler converts RGB->YUV and scales the
     * (center-cropped) frame into the stream resolution, BT.709 for SDR or BT.2020 NCL on PQ for
     * HDR, in the range requested at creation, with the chroma midpoint at the container depth's
     * conventional code point.
     * The result is packetized into a single contiguous byte buffer (packet boundaries are internal
     * to PyroWave and carried in the stream). Every frame is intra (IDR). With record framing the
     * payload starts with the PyroWave sequence header followed by block and padding records laid
     * out for partial decoding. Otherwise the transport framing is
     *
     *   [u32 packet count] { [u32 size] [size bytes] } * packet count
     *
     * @param img Captured image from the display backend.
     * @param out Receives the encoded, packetized bitstream bytes.
     * @param head_bytes Receives the byte offset into @p out where the loss-critical head ends
     *                   (the sequence header and the coarsest wavelet level — PyroWave emits coarse
     *                   levels first). The RTP layer protects `[0, head_bytes)` with Reed-Solomon
     *                   and leaves the tail bare. 0 when the frame is too small to split (and on
     *                   failure), meaning "use the normal even FEC split".
     * @return 0 on success, negative on failure.
     */
    int encode(const platf::img_t &img, std::vector<uint8_t> &out, size_t &head_bytes);

    /**
     * @brief Report whether this session emits record framing.
     *
     * @return True when frames use the record framing negotiated at creation.
     */
    [[nodiscard]] bool record_framing() const;

  private:
    encoder_t() = default;

    struct impl_t;
    std::unique_ptr<impl_t> impl;
  };

}  // namespace platf::pyrowave

#endif  // SUNSHINE_ENABLE_PYROWAVE
