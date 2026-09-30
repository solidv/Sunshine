/**
 * @file tests/unit/test_pyrowave_encode.cpp
 * @brief Tests the PyroWave encoder's DMA-BUF import path (src/platform/linux/pyrowave_encode.*).
 *
 * These tests mint a real DMA-BUF with a known colour, feed it through the encoder, decode the
 * resulting bitstream with PyroWave's own decoder, and check the reconstructed YCbCr values. That
 * is what catches import mistakes a working stream hides: a wrong Vulkan format mapping or source
 * swizzle still streams, but with swapped or shifted colours.
 *
 * They need a GPU with the Vulkan features PyroWave requires plus a GBM render node, and are
 * skipped otherwise (as in CI containers), matching how the other hardware-dependent tests behave.
 *
 * The fixtures mint their DMA-BUF on the first usable render node while the encoder picks the
 * default GPU, so on a multi-GPU machine where those differ the imported buffer would not be
 * reachable and these tests would report a false failure.
 */
#ifdef SUNSHINE_ENABLE_PYROWAVE

  // test includes
  #include "../tests_common.h"

  // standard includes
  #include <array>
  #include <cstdint>
  #include <cstring>
  #include <optional>
  #include <vector>

  // lib includes
  #include <drm_fourcc.h>
  #include <fcntl.h>
  #include <gbm.h>
  #include <sys/mman.h>
  #include <unistd.h>

  // PyroWave requires the Vulkan headers to be included before it, and clang-format would otherwise
  // sort the quoted includes ahead of them.
  // clang-format off
  #include <vulkan/vulkan.h>
  #include "pyrowave.h"
  // clang-format on

  // local includes
  #include "src/platform/linux/graphics.h"
  #include "src/platform/linux/pyrowave_encode.h"

namespace {
  constexpr int image_width = 256;  ///< Test image width; 4:2:0 needs it to be even.
  constexpr int image_height = 256;  ///< Test image height; 4:2:0 needs it to be even.
  constexpr int encode_bitrate_kbps = 20000;  ///< Generous per-frame budget keeps a flat image near lossless.
  constexpr int encode_frame_rate = 60;  ///< Framerate used with the bitrate to size the frame budget.
  constexpr int colour_tolerance = 12;  ///< Wavelet quantization drift tolerated on a flat image.

  /**
   * @brief A GBM buffer created on a render node and the GBM/DRM handles keeping it alive.
   *
   * The DMA-BUF file descriptor is not owned here: it is handed to the image descriptor, which
   * closes it, matching how the capture backends hand buffers to the encoder.
   */
  struct gbm_buffer_t {
    gbm_device *device = nullptr;  ///< GBM device the buffer was created on.
    gbm_bo *bo = nullptr;  ///< The single-plane buffer itself.
    int drm_fd = -1;  ///< Render node the GBM device belongs to.

    gbm_buffer_t() = default;
    gbm_buffer_t(const gbm_buffer_t &) = delete;
    gbm_buffer_t &operator=(const gbm_buffer_t &) = delete;

    /**
     * @brief Destroy the buffer, GBM device, and render-node file descriptor.
     */
    ~gbm_buffer_t() {
      if (bo) {
        gbm_bo_destroy(bo);
      }
      if (device) {
        gbm_device_destroy(device);
      }
      if (drm_fd >= 0) {
        close(drm_fd);
      }
    }

    /**
     * @brief Create a linear, CPU-mappable buffer on the first usable render node.
     *
     * @param fourcc DRM fourcc to allocate.
     * @return True on success; false when the machine has no usable render node.
     */
    bool create(uint32_t fourcc) {
      for (int index = 128; index < 136; ++index) {
        auto path = std::string {"/dev/dri/renderD"} + std::to_string(index);
        drm_fd = open(path.c_str(), O_RDWR);
        if (drm_fd >= 0) {
          break;
        }
      }
      if (drm_fd < 0) {
        return false;
      }
      device = gbm_create_device(drm_fd);
      if (!device) {
        return false;
      }
      bo = gbm_bo_create(device, image_width, image_height, fourcc, GBM_BO_USE_LINEAR);
      return bo != nullptr;
    }
  };

  /**
   * @brief Fill an entire buffer with one 32-bit pixel value.
   *
   * @param bo Buffer to fill.
   * @param pixel Pixel value in the buffer's native (native-endian word) representation.
   * @return True on success.
   */
  bool fill_solid(gbm_bo *bo, uint32_t pixel) {
    const uint32_t stride = gbm_bo_get_stride(bo);
    const int fd = gbm_bo_get_fd(bo);
    if (fd < 0) {
      return false;
    }

    const size_t size = (size_t) stride * image_height;
    auto *map = static_cast<uint8_t *>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    if (map == MAP_FAILED) {
      return false;
    }

    for (int y = 0; y < image_height; ++y) {
      auto *row = reinterpret_cast<uint32_t *>(map + (size_t) y * stride);
      for (int x = 0; x < image_width; ++x) {
        row[x] = pixel;
      }
    }
    msync(map, size, MS_SYNC);
    munmap(map, size);
    return true;
  }

  /**
   * @brief Reconstructed YCbCr planes of a decoded PyroWave frame.
   */
  struct decoded_frame_t {
    std::vector<uint8_t> y;  ///< Luma plane.
    std::vector<uint8_t> cb;  ///< Cb plane.
    std::vector<uint8_t> cr;  ///< Cr plane.

    /**
     * @brief Sample the luma plane at the image centre.
     *
     * @return Luma value of the centre pixel.
     */
    uint8_t luma_center() const {
      return y[(image_height / 2) * image_width + image_width / 2];
    }

    /**
     * @brief Sample the Cb plane at the image centre.
     *
     * @return Cb value of the centre chroma sample.
     */
    uint8_t cb_center() const {
      return cb[(image_height / 4) * (image_width / 2) + image_width / 4];
    }

    /**
     * @brief Sample the Cr plane at the image centre.
     *
     * @return Cr value of the centre chroma sample.
     */
    uint8_t cr_center() const {
      return cr[(image_height / 4) * (image_width / 2) + image_width / 4];
    }
  };

  /**
   * @brief Decode the framed PyroWave bitstream produced by the encoder.
   *
   * @param bitstream Framed bitstream: [u32 packet count] { [u32 size] [bytes] } * count.
   * @return Decoded planes, or std::nullopt when the frame could not be decoded.
   */
  std::optional<decoded_frame_t> decode_bitstream(const std::vector<uint8_t> &bitstream) {
    pyrowave_device device = nullptr;
    if (pyrowave_create_device_by_compat(0, 0, nullptr, nullptr, nullptr, &device) != PYROWAVE_SUCCESS) {
      return std::nullopt;
    }

    auto cleanup = [&device](pyrowave_decoder decoder) {
      if (decoder) {
        pyrowave_decoder_destroy(decoder);
      }
      pyrowave_device_destroy(device);
    };

    pyrowave_decoder_create_info create_info {};
    create_info.device = device;
    create_info.width = image_width;
    create_info.height = image_height;
    create_info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;
    create_info.fragment_path = false;

    pyrowave_decoder decoder = nullptr;
    if (pyrowave_decoder_create(&create_info, &decoder) != PYROWAVE_SUCCESS) {
      cleanup(nullptr);
      return std::nullopt;
    }

    auto read_u32 = [&bitstream](size_t offset, uint32_t &value) {
      if (offset + sizeof(uint32_t) > bitstream.size()) {
        return false;
      }
      std::memcpy(&value, bitstream.data() + offset, sizeof(uint32_t));
      return true;
    };

    uint32_t packet_count = 0;
    size_t offset = 0;
    bool ok = read_u32(offset, packet_count) && packet_count > 0;
    offset += sizeof(uint32_t);
    for (uint32_t packet = 0; ok && packet < packet_count; ++packet) {
      uint32_t size = 0;
      ok = read_u32(offset, size) && offset + sizeof(uint32_t) + size <= bitstream.size();
      offset += sizeof(uint32_t);
      if (!ok) {
        break;
      }
      ok = pyrowave_decoder_push_packet(decoder, bitstream.data() + offset, size) == PYROWAVE_SUCCESS;
      offset += size;
    }
    if (!ok || offset != bitstream.size()) {
      cleanup(decoder);
      return std::nullopt;
    }

    decoded_frame_t frame;
    frame.y.resize((size_t) image_width * image_height);
    frame.cb.resize((size_t) image_width * image_height / 4);
    frame.cr.resize((size_t) image_width * image_height / 4);

    pyrowave_cpu_buffer buffer {};
    buffer.data[0] = frame.y.data();
    buffer.data[1] = frame.cb.data();
    buffer.data[2] = frame.cr.data();
    buffer.row_stride_in_bytes[0] = image_width;
    buffer.row_stride_in_bytes[1] = image_width / 2;
    buffer.row_stride_in_bytes[2] = image_width / 2;
    buffer.plane_size_in_bytes[0] = frame.y.size();
    buffer.plane_size_in_bytes[1] = frame.cb.size();
    buffer.plane_size_in_bytes[2] = frame.cr.size();
    buffer.width = image_width;
    buffer.height = image_height;
    buffer.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;

    if (!pyrowave_decoder_decode_is_ready(decoder, true) || pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &buffer) != PYROWAVE_SUCCESS) {
      cleanup(decoder);
      return std::nullopt;
    }

    cleanup(decoder);
    return frame;
  }

  /**
   * @brief Encode a solid-colour DMA-BUF and check the colours that come back out.
   *
   * @param fourcc DRM fourcc of the buffer to mint.
   * @param pixel Solid pixel value in the buffer's native representation.
   * @param expected_y Expected decoded luma (BT.601 limited range on sRGB-encoded RGB).
   * @param expected_cb Expected decoded Cb.
   * @param expected_cr Expected decoded Cr.
   * @param ten_bit True to encode into 10-bit (R16_UNORM) plane containers.
   * @param modifier DRM modifier to advertise for the minted buffer; the default takes the import
   * path used when the kernel reports no modifier, while a real modifier takes that of a capture
   * that does.
   */
  void expect_solid_colour_round_trip(uint32_t fourcc, uint32_t pixel, uint8_t expected_y, uint8_t expected_cb, uint8_t expected_cr, bool ten_bit, uint64_t modifier = DRM_FORMAT_MOD_INVALID) {
    if (!platf::pyrowave::validate()) {
      GTEST_SKIP() << "No PyroWave-capable Vulkan device";
    }

    gbm_buffer_t buffer;
    if (!buffer.create(fourcc)) {
      GTEST_SKIP() << "No render node available that can mint a DMA-BUF of the requested format";
    }
    if (!fill_solid(buffer.bo, pixel)) {
      GTEST_SKIP() << "DMA-BUF could not be mapped for writing";
    }

    egl::img_descriptor_t img;
    img.width = image_width;
    img.height = image_height;
    img.data = nullptr;  // DMA-BUF images carry no CPU mapping
    img.pixel_pitch = 4;
    img.row_pitch = (int32_t) gbm_bo_get_stride(buffer.bo);

    std::fill_n(img.sd.fds, 4, -1);
    img.sd.width = image_width;
    img.sd.height = image_height;
    img.sd.fourcc = gbm_bo_get_format(buffer.bo);
    img.sd.modifier = modifier;
    img.sd.pitches[0] = gbm_bo_get_stride(buffer.bo);
    img.sd.offsets[0] = 0;
    img.sd.fds[0] = gbm_bo_get_fd(buffer.bo);  // Ownership moves to img, which closes it.
    ASSERT_GE(img.sd.fds[0], 0);

    auto encoder = platf::pyrowave::encoder_t::create(image_width, image_height, encode_bitrate_kbps, encode_frame_rate, false, ten_bit, false);
    ASSERT_NE(encoder, nullptr) << "PyroWave encoder creation failed";

    std::vector<uint8_t> bitstream;
    size_t head_bytes = 0;
    ASSERT_EQ(encoder->encode(img, bitstream, head_bytes), 0);
    EXPECT_FALSE(bitstream.empty());
    // A flat frame is all "critical" bands, so the encoder is expected to report either no split
    // or a split inside the frame, never an offset past the end.
    EXPECT_LE(head_bytes, bitstream.size());

    auto frame = decode_bitstream(bitstream);
    ASSERT_TRUE(frame.has_value()) << "PyroWave decoder could not decode the encoded frame";

    EXPECT_NEAR(frame->luma_center(), expected_y, colour_tolerance);
    EXPECT_NEAR(frame->cb_center(), expected_cb, colour_tolerance);
    EXPECT_NEAR(frame->cr_center(), expected_cr, colour_tolerance);

    // Red and blue differ in which chroma axis is high, so a swapped R/B source mapping flips these.
    if (expected_cr > expected_cb) {
      EXPECT_GT(frame->cr_center(), frame->cb_center());
    } else {
      EXPECT_LT(frame->cr_center(), frame->cb_center());
    }
  }
}  // namespace

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom8BitDmaBuf) {
  // DRM_FORMAT_ARGB8888 is 0xAARRGGBB in a little-endian word: memory order B, G, R, A.
  expect_solid_colour_round_trip(DRM_FORMAT_ARGB8888, 0xffff0000u, 81, 90, 240, false);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidBlueFrom8BitDmaBuf) {
  expect_solid_colour_round_trip(DRM_FORMAT_ARGB8888, 0xff0000ffu, 41, 240, 110, false);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom8BitXbgrDmaBuf) {
  // DRM_FORMAT_XBGR8888 is 0xXXBBGGRR: memory order R, G, B, X. It maps to a different Vulkan
  // format than ARGB8888, so this catches a swapped red/blue source mapping.
  expect_solid_colour_round_trip(DRM_FORMAT_XBGR8888, 0xff0000ffu, 81, 90, 240, false);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom10BitDmaBuf) {
  // DRM_FORMAT_XRGB2101010 is 0xXXRRGGBB: 10-bit red in bits 20..29. Encoding into R16_UNORM
  // plane containers exercises the 10-bit (HDR profile) storage-image path.
  expect_solid_colour_round_trip(DRM_FORMAT_XRGB2101010, 0x3ff00000u, 81, 90, 240, true);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom10BitXbgrDmaBuf) {
  // DRM_FORMAT_XBGR2101010 is 0xXXBBGGRR: 10-bit red in bits 0..9. This is the format AMD's KMS
  // capture reports for HDR10 buffers, so it is the HDR path that matters in practice.
  expect_solid_colour_round_trip(DRM_FORMAT_XBGR2101010, 0x000003ffu, 81, 90, 240, true);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFromModifierDmaBuf) {
  // The buffers above are imported as DRM_FORMAT_MOD_INVALID, which is how GBM reports them. A
  // capture can instead report the modifier explicitly, which takes the DRM format modifier import
  // path: querying the plane count and passing explicit plane layouts to Vulkan.
  expect_solid_colour_round_trip(DRM_FORMAT_ARGB8888, 0xffff0000u, 81, 90, 240, false, DRM_FORMAT_MOD_LINEAR);
}

#endif  // SUNSHINE_ENABLE_PYROWAVE
