/**
 * @file src/platform/linux/pyrowave_encode.h
 * @brief PyroWave (Vulkan wavelet, intra-only) host encoder for Sunshine.
 *
 * PyroWave does not ride the avcodec/nvenc encoder_t abstraction: it owns its own Vulkan 1.3
 * device and emits an already-packetized byte bitstream that Sunshine ships as packet_raw_generic.
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
  #include <memory>
  #include <vector>

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
   * @brief Probe for a PyroWave-capable Vulkan device on Sunshine's capture GPU.
   *
   * Creates a throwaway Vulkan 1.1 instance, matches the render node Sunshine captures from to a
   * physical device (VK_EXT_physical_device_drm), and checks the compute features PyroWave's
   * encoder requires (subgroup size control, shaderInt16, storageBuffer8BitAccess). When the
   * render node cannot be matched, any device satisfies the probe, matching the encoder's
   * default-device fallback. Cheap enough to call from the encoder-probe path; the result gates
   * whether SCM_PYROWAVE is advertised.
   *
   * @return True when the capture GPU (or any device, when unmatched) can run the PyroWave encoder.
   */
  bool validate();

  /**
   * @brief One PyroWave encode session bound to a stream's width/height.
   *
   * Owns a dedicated Vulkan device plus a pyrowave_encoder. The device is pinned to the capture
   * GPU when its render node can be matched, and a high-priority queue is requested (the granted
   * priority may be lower when the OS refuses). Not thread-safe: the caller must serialize
   * encode() calls, matching the pyrowave encoder contract.
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
     * @param hdr True to encode HDR content: BT.2020 NCL full-range on PQ-encoded RGB
     *            (requires ten_bit and an HDR display; else SDR full-range BT.709 is used).
     * @return Session on success, nullptr on failure.
     */
    static std::unique_ptr<encoder_t> create(int width, int height, int bitrate_kbps, int frame_rate, bool yuv444, bool ten_bit, bool hdr);

    /**
     * @brief Encode one captured frame into a PyroWave bitstream.
     *
     * Requires a DMA-BUF captured image. Imports the buffer into PyroWave's Vulkan device and runs
     * the synchronous GPU encode directly on it: PyroWave's scaler converts RGB->YUV and scales the
     * (center-cropped) frame into the stream resolution, full-range BT.709 for SDR or BT.2020 NCL
     * on PQ for HDR, with the chroma midpoint at the container depth's conventional code point.
     * The result is packetized into a single contiguous byte buffer (packet boundaries are internal
     * to PyroWave and carried in the stream). Every frame is intra (IDR).
     *
     * @param img Captured image from the display backend.
     * @param out Receives the encoded, packetized bitstream bytes.
     * @param head_bytes Receives the byte offset into @p out where the loss-critical head ends
     *                   (the sequence header plus wavelet levels 4 and 3 — PyroWave emits coarse
     *                   levels first). The RTP layer protects `[0, head_bytes)` with Reed-Solomon
     *                   and leaves the tail bare. 0 when the frame is too small to split (and on
     *                   failure), meaning "use the normal even FEC split".
     * @return 0 on success, negative on failure.
     */
    int encode(const platf::img_t &img, std::vector<uint8_t> &out, size_t &head_bytes);

  private:
    encoder_t() = default;

    struct impl_t;
    std::unique_ptr<impl_t> impl;
  };

}  // namespace platf::pyrowave

#endif  // SUNSHINE_ENABLE_PYROWAVE
