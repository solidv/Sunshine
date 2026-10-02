/**
 * @file tests/unit/test_pyrowave_encode.cpp
 * @brief Tests the PyroWave encoder's DMA-BUF import path (src/platform/linux/pyrowave_encode.*).
 *
 * These tests mint a real DMA-BUF with a known colour, feed it through the encoder, decode the
 * resulting bitstream with PyroWave's own decoder, and check the reconstructed YCbCr values. That
 * is what catches import mistakes a working stream hides: a wrong Vulkan format mapping or source
 * swizzle still streams, but with swapped or shifted colours.
 *
 * The encoder hands the buffer to PyroWave's scaler, so the wire values are the scaler's
 * conventions: full-range BT.709 YCbCr of the sRGB-encoded RGB, with neutral chroma at the
 * container depth's conventional code point (128/255 for R8, 512/1023 for R16).
 *
 * They need a GPU with the Vulkan features PyroWave requires plus a GBM render node, and are
 * skipped otherwise (as in CI containers), matching how the other hardware-dependent tests behave.
 *
 * The fixtures mint their DMA-BUF on the first usable render node, while the encoder pins itself
 * to the GPU Sunshine resolves for capture (the GPU with a connected display, or the configured
 * adapter). On a multi-GPU machine where those differ the imported buffer would not be reachable
 * and these tests would report a failure; that mirrors the real capture path.
 */
#ifdef SUNSHINE_ENABLE_PYROWAVE

  // test includes
  #include "../tests_common.h"

  // standard includes
  #include <algorithm>
  #include <array>
  #include <cstdint>
  #include <cstring>
  #include <optional>
  #include <utility>
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
   * @brief Fill every row of a buffer with a colour chosen per row.
   *
   * @param bo Buffer to fill.
   * @param pixel_at_row Maps a row index to its 32-bit pixel value in the buffer's native
   *                     (native-endian word) representation.
   * @return True on success.
   */
  template<typename PixelAtRow>
  bool fill_rows(gbm_bo *bo, PixelAtRow pixel_at_row) {
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
      std::fill_n(row, image_width, pixel_at_row(y));
    }
    msync(map, size, MS_SYNC);
    munmap(map, size);
    return true;
  }

  /**
   * @brief Fill an entire buffer with one 32-bit pixel value.
   *
   * @param bo Buffer to fill.
   * @param pixel Pixel value in the buffer's native (native-endian word) representation.
   * @return True on success.
   */
  bool fill_solid(gbm_bo *bo, uint32_t pixel) {
    return fill_rows(bo, [pixel](int) {
      return pixel;
    });
  }

  /**
   * @brief Describe a minted GBM buffer as a captured DMA-BUF image.
   *
   * @param buffer Buffer to describe. Its DMA-BUF fd moves into @p img, which closes it.
   * @param img Receives the image descriptor.
   */
  void fill_image_descriptor(gbm_buffer_t &buffer, egl::img_descriptor_t &img) {
    img.width = image_width;
    img.height = image_height;
    img.data = nullptr;  // DMA-BUF images carry no CPU mapping
    img.pixel_pitch = 4;
    img.row_pitch = (int32_t) gbm_bo_get_stride(buffer.bo);

    std::fill_n(img.sd.fds, 4, -1);
    img.sd.width = image_width;
    img.sd.height = image_height;
    img.sd.fourcc = gbm_bo_get_format(buffer.bo);
    img.sd.modifier = DRM_FORMAT_MOD_INVALID;
    img.sd.pitches[0] = gbm_bo_get_stride(buffer.bo);
    img.sd.offsets[0] = 0;
    img.sd.fds[0] = gbm_bo_get_fd(buffer.bo);  // Ownership moves to img, which closes it.
  }

  /**
   * @brief Reconstructed YCbCr planes of a decoded PyroWave frame.
   */
  struct decoded_frame_t {
    int width = 0;  ///< Decoded frame width.
    int height = 0;  ///< Decoded frame height.
    std::vector<uint8_t> y;  ///< Luma plane.
    std::vector<uint8_t> cb;  ///< Cb plane.
    std::vector<uint8_t> cr;  ///< Cr plane.

    /**
     * @brief Sample the luma plane.
     *
     * @param x Sample column.
     * @param row Sample row.
     * @return Luma value at the given position.
     */
    uint8_t luma_at(int x, int row) const {
      return y[(size_t) row * width + x];
    }

    /**
     * @brief Sample the Cb plane at the chroma sample covering a luma position.
     *
     * @param x Luma column.
     * @param row Luma row.
     * @return Cb value for that position.
     */
    uint8_t cb_at(int x, int row) const {
      return cb[(size_t) (row / 2) * (width / 2) + x / 2];
    }

    /**
     * @brief Sample the Cr plane at the chroma sample covering a luma position.
     *
     * @param x Luma column.
     * @param row Luma row.
     * @return Cr value for that position.
     */
    uint8_t cr_at(int x, int row) const {
      return cr[(size_t) (row / 2) * (width / 2) + x / 2];
    }
  };

  /**
   * @brief Decode the framed PyroWave bitstream produced by the encoder.
   *
   * @param bitstream Framed bitstream: [u32 packet count] { [u32 size] [bytes] } * count, with an
   *                  optional [u32 mask words] [words] header after the packet count when
   *                  @p has_active_block_mask is set.
   * @param width Stream width the bitstream was encoded with.
   * @param height Stream height the bitstream was encoded with.
   * @param has_active_block_mask True when the framing includes the active-block mask header.
   * @param active_mask_out Optional receiver for the parsed active-block mask words.
   * @return Decoded planes, or std::nullopt when the frame could not be decoded.
   */
  std::optional<decoded_frame_t> decode_bitstream(const std::vector<uint8_t> &bitstream, int width, int height, bool has_active_block_mask = false, std::vector<uint32_t> *active_mask_out = nullptr) {
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
    create_info.width = width;
    create_info.height = height;
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

    std::vector<uint32_t> active_mask;
    if (ok && has_active_block_mask) {
      uint32_t mask_words = 0;
      ok = read_u32(offset, mask_words) && offset + sizeof(uint32_t) + (size_t) mask_words * sizeof(uint32_t) <= bitstream.size();
      offset += sizeof(uint32_t);
      active_mask.resize(mask_words);
      for (uint32_t word = 0; ok && word < mask_words; ++word) {
        ok = read_u32(offset, active_mask[word]);
        offset += sizeof(uint32_t);
      }
    }
    if (active_mask_out) {
      *active_mask_out = std::move(active_mask);
    }

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
    frame.width = width;
    frame.height = height;
    frame.y.resize((size_t) width * height);
    frame.cb.resize((size_t) width * height / 4);
    frame.cr.resize((size_t) width * height / 4);

    pyrowave_cpu_buffer buffer {};
    buffer.data[0] = frame.y.data();
    buffer.data[1] = frame.cb.data();
    buffer.data[2] = frame.cr.data();
    buffer.row_stride_in_bytes[0] = width;
    buffer.row_stride_in_bytes[1] = width / 2;
    buffer.row_stride_in_bytes[2] = width / 2;
    buffer.plane_size_in_bytes[0] = frame.y.size();
    buffer.plane_size_in_bytes[1] = frame.cb.size();
    buffer.plane_size_in_bytes[2] = frame.cr.size();
    buffer.width = width;
    buffer.height = height;
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
   * @param expected_y Expected decoded luma (full-range BT.709 on sRGB-encoded RGB).
   * @param expected_cb Expected decoded Cb.
   * @param expected_cr Expected decoded Cr.
   * @param ten_bit True to encode into 10-bit (R16_UNORM) plane containers.
   * @param modifier DRM modifier to advertise for the minted buffer; the default takes the
   * modifier-less import path (imported as linear), while a real modifier takes that capture's
   * external image path.
   * @param active_block_sideband True to negotiate the active-block sideband and require the mask
   * header in the framed output.
   */
  void expect_solid_colour_round_trip(uint32_t fourcc, uint32_t pixel, uint8_t expected_y, uint8_t expected_cb, uint8_t expected_cr, bool ten_bit, uint64_t modifier = DRM_FORMAT_MOD_INVALID, bool active_block_sideband = false) {
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
    fill_image_descriptor(buffer, img);
    img.sd.modifier = modifier;
    ASSERT_GE(img.sd.fds[0], 0);

    auto encoder = platf::pyrowave::encoder_t::create(image_width, image_height, encode_bitrate_kbps, encode_frame_rate, false, ten_bit, false, active_block_sideband);
    ASSERT_NE(encoder, nullptr) << "PyroWave encoder creation failed";

    std::vector<uint8_t> bitstream;
    size_t head_bytes = 0;
    ASSERT_EQ(encoder->encode(img, bitstream, head_bytes), 0);
    EXPECT_FALSE(bitstream.empty());
    // A flat frame is all "critical" bands, so the encoder is expected to report either no split
    // or a split inside the frame, never an offset past the end. A negotiated mask must fit in it.
    EXPECT_LE(head_bytes, bitstream.size());
    if (active_block_sideband && head_bytes > 0) {
      EXPECT_GE(head_bytes, 4 + 4);  // Packet count and mask word count are protected.
    }

    std::vector<uint32_t> active_mask;
    auto frame = decode_bitstream(bitstream, image_width, image_height, active_block_sideband, active_block_sideband ? &active_mask : nullptr);
    ASSERT_TRUE(frame.has_value()) << "PyroWave decoder could not decode the encoded frame";

    if (active_block_sideband) {
      // The mask header is always present when negotiated; a real image has active coarse blocks.
      ASSERT_FALSE(active_mask.empty()) << "negotiated sideband must emit a mask";
      EXPECT_TRUE(std::ranges::any_of(active_mask, [](uint32_t word) {
        return word != 0;
      }))
        << "mask must mark the transmitted coarse blocks";
    }

    const int cx = image_width / 2;
    const int cy = image_height / 2;
    EXPECT_NEAR(frame->luma_at(cx, cy), expected_y, colour_tolerance);
    EXPECT_NEAR(frame->cb_at(cx, cy), expected_cb, colour_tolerance);
    EXPECT_NEAR(frame->cr_at(cx, cy), expected_cr, colour_tolerance);

    // Red and blue differ in which chroma axis is high, so a swapped R/B source mapping flips these.
    if (expected_cr > expected_cb) {
      EXPECT_GT(frame->cr_at(cx, cy), frame->cb_at(cx, cy));
    } else {
      EXPECT_LT(frame->cr_at(cx, cy), frame->cb_at(cx, cy));
    }
  }
}  // namespace

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom8BitDmaBuf) {
  // DRM_FORMAT_ARGB8888 is 0xAARRGGBB in a little-endian word: memory order B, G, R, A.
  expect_solid_colour_round_trip(DRM_FORMAT_ARGB8888, 0xffff0000u, 54, 99, 255, false);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidBlueFrom8BitDmaBuf) {
  expect_solid_colour_round_trip(DRM_FORMAT_ARGB8888, 0xff0000ffu, 18, 255, 116, false);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom8BitXbgrDmaBuf) {
  // DRM_FORMAT_XBGR8888 is 0xXXBBGGRR: memory order R, G, B, X. It maps to a different Vulkan
  // format than ARGB8888, so this catches a swapped red/blue source mapping.
  expect_solid_colour_round_trip(DRM_FORMAT_XBGR8888, 0xff0000ffu, 54, 99, 255, false);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom10BitDmaBuf) {
  // DRM_FORMAT_XRGB2101010 is 0xXXRRGGBB: 10-bit red in bits 20..29. Encoding into R16_UNORM
  // plane containers exercises the 10-bit (HDR profile) storage path.
  expect_solid_colour_round_trip(DRM_FORMAT_XRGB2101010, 0x3ff00000u, 54, 99, 255, true);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom10BitXbgrDmaBuf) {
  // DRM_FORMAT_XBGR2101010 is 0xXXBBGGRR: 10-bit red in bits 0..9. This is the format AMD's KMS
  // capture reports for HDR10 buffers, so it is the HDR path that matters in practice.
  expect_solid_colour_round_trip(DRM_FORMAT_XBGR2101010, 0x000003ffu, 54, 99, 255, true);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFromModifierDmaBuf) {
  // The buffers above are imported as DRM_FORMAT_MOD_INVALID, which is how GBM reports them. A
  // capture can instead report the modifier explicitly, which takes PyroWave's external image
  // path: the image is created on, and acquired by, PyroWave's own device.
  expect_solid_colour_round_trip(DRM_FORMAT_ARGB8888, 0xffff0000u, 54, 99, 255, false, DRM_FORMAT_MOD_LINEAR);
}

TEST(PyroWaveEncodeDmaBufTest, EncodesSolidRedFrom10BitModifierDmaBuf) {
  // The external image path with the 10-bit (R16_UNORM) plane containers, i.e. the HDR profile
  // over the fast import route.
  expect_solid_colour_round_trip(DRM_FORMAT_XBGR2101010, 0x000003ffu, 54, 99, 255, true, DRM_FORMAT_MOD_LINEAR);
}

TEST(PyroWaveEncodeDmaBufTest, CarriesActiveBlockMaskWhenNegotiated) {
  // The client requested the sideband, so the framing must carry the mask header in addition to
  // the packets, the mask must mark the transmitted coarse blocks, and the frame must still decode.
  expect_solid_colour_round_trip(DRM_FORMAT_ARGB8888, 0xffff0000u, 54, 99, 255, false, DRM_FORMAT_MOD_INVALID, true);
}

TEST(PyroWaveEncodeDmaBufTest, CenterCropsSourceToStreamAspect) {
  // A 256x256 source encoded into a 256x128 stream. PyroWave's scaler always fills the encoder
  // frame, so the wrapper center-crops the source (rows 64..191) instead of stretching it. Only
  // the green middle band of the source may come back; red (rows 0..63) and blue (rows 192..255)
  // must have been cropped away.
  if (!platf::pyrowave::validate()) {
    GTEST_SKIP() << "No PyroWave-capable Vulkan device";
  }

  gbm_buffer_t buffer;
  if (!buffer.create(DRM_FORMAT_ARGB8888)) {
    GTEST_SKIP() << "No render node available that can mint a DMA-BUF of the requested format";
  }
  constexpr uint32_t red = 0xffff0000u;  // sRGB (1, 0, 0)
  constexpr uint32_t green = 0xff00ff00u;  // sRGB (0, 1, 0)
  constexpr uint32_t blue = 0xff0000ffu;  // sRGB (0, 0, 1)
  if (!fill_rows(buffer.bo, [](int row) {
        return row < image_height / 4 ? red : (row < 3 * image_height / 4 ? green : blue);
      })) {
    GTEST_SKIP() << "DMA-BUF could not be mapped for writing";
  }

  egl::img_descriptor_t img;
  fill_image_descriptor(buffer, img);
  ASSERT_GE(img.sd.fds[0], 0);

  constexpr int stream_width = image_width;
  constexpr int stream_height = image_height / 2;
  auto encoder = platf::pyrowave::encoder_t::create(stream_width, stream_height, encode_bitrate_kbps, encode_frame_rate, false, false, false, false);
  ASSERT_NE(encoder, nullptr) << "PyroWave encoder creation failed";

  std::vector<uint8_t> bitstream;
  size_t head_bytes = 0;
  ASSERT_EQ(encoder->encode(img, bitstream, head_bytes), 0);

  auto frame = decode_bitstream(bitstream, stream_width, stream_height);
  ASSERT_TRUE(frame.has_value()) << "PyroWave decoder could not decode the encoded frame";

  // Full-range BT.709 for sRGB green (0, 1, 0): Y ~= 182, Cb ~= 29, Cr ~= 12. Red and blue are
  // far from these on luma and on one chroma axis, so sampling a few rows well away from the band
  // edges catches a stretch (which would show red at the top and blue at the bottom).
  const int rows[] = {stream_height / 8, stream_height / 2, 7 * stream_height / 8};
  for (int y : rows) {
    EXPECT_NEAR(frame->luma_at(stream_width / 2, y), 182, 20) << "row " << y;
    EXPECT_NEAR(frame->cb_at(stream_width / 2, y), 29, 20) << "row " << y;
    EXPECT_NEAR(frame->cr_at(stream_width / 2, y), 12, 20) << "row " << y;
  }
}

TEST(PyroWaveCropRectTest, MatchesAspectWithoutCrop) {
  EXPECT_FALSE(platf::pyrowave::compute_crop_rect(1920, 1080, 1280, 720).needed);
  EXPECT_FALSE(platf::pyrowave::compute_crop_rect(256, 256, 256, 256).needed);
}

TEST(PyroWaveCropRectTest, TrimsTheSidesWhenTheSourceIsWider) {
  auto crop = platf::pyrowave::compute_crop_rect(2560, 1080, 1920, 1080);
  EXPECT_TRUE(crop.needed);
  EXPECT_EQ(crop.width, 1920);
  EXPECT_EQ(crop.height, 1080);
  EXPECT_EQ(crop.x, 320);
  EXPECT_EQ(crop.y, 0);
}

TEST(PyroWaveCropRectTest, TrimsTheHeightWhenTheSourceIsTaller) {
  auto crop = platf::pyrowave::compute_crop_rect(1920, 1200, 1920, 1080);
  EXPECT_TRUE(crop.needed);
  EXPECT_EQ(crop.width, 1920);
  EXPECT_EQ(crop.height, 1080);
  EXPECT_EQ(crop.x, 0);
  EXPECT_EQ(crop.y, 60);
}

TEST(PyroWaveCropRectTest, AlignsCropAndOffsetToEvenPixels) {
  // 4:2:0 sources need even crop geometry, and odd dimensions must stay inside the source.
  auto crop = platf::pyrowave::compute_crop_rect(1921, 1201, 1920, 1080);
  ASSERT_TRUE(crop.needed);
  EXPECT_EQ(crop.width & 1, 0);
  EXPECT_EQ(crop.height & 1, 0);
  EXPECT_EQ(crop.x & 1, 0);
  EXPECT_EQ(crop.y & 1, 0);
  EXPECT_LE(crop.x + crop.width, 1921);
  EXPECT_LE(crop.y + crop.height, 1201);
}

TEST(PyroWaveCropRectTest, RejectsDegenerateDimensions) {
  EXPECT_FALSE(platf::pyrowave::compute_crop_rect(0, 1080, 1920, 1080).needed);
  EXPECT_FALSE(platf::pyrowave::compute_crop_rect(-1, 1080, 1920, 1080).needed);
  EXPECT_FALSE(platf::pyrowave::compute_crop_rect(1920, 1080, 0, 0).needed);
}

TEST(PyroWavePrivilegedWorkerTest, StartsIdempotentlyAndCreatesEncoder) {
  // Sunshine starts the worker from main() while CAP_SYS_NICE is still held; tests start it here
  // and it must tolerate being started more than once.
  platf::pyrowave::ensure_privileged_device_worker_started();
  platf::pyrowave::ensure_privileged_device_worker_started();

  if (!platf::pyrowave::validate()) {
    GTEST_SKIP() << "No PyroWave-capable Vulkan device";
  }

  // Device creation runs through the privileged worker; without the capability (as in tests and
  // unprivileged builds) Granite's internal fallback still grants medium priority.
  auto encoder = platf::pyrowave::encoder_t::create(image_width, image_height, encode_bitrate_kbps, encode_frame_rate, false, false, false);
  ASSERT_NE(encoder, nullptr) << "PyroWave encoder creation failed";
}

#endif  // SUNSHINE_ENABLE_PYROWAVE
