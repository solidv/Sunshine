/**
 * @file src/pyrowave_protocol.h
 * @brief Wire constants for PyroWave over the GameStream/Sunshine video stream.
 *
 * The values match the PyroWave clients' expectations and are shared with the host policy in
 * `pyrowave_policy.h`. The PyroWave bitstream itself carries no version field, so both ends must
 * use the same encoder revision; the host advertises the revision it vendors in RTSP DESCRIBE.
 */
#pragma once

// standard includes
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace pyrowave::protocol {
  /**
   * @brief `x-nv-vqos[0].bitStreamFormat` value selecting PyroWave (0/1/2 are H.264/HEVC/AV1).
   */
  constexpr int BITSTREAM_FORMAT = 3;

  /**
   * @brief First eight hex digits of the vendored PyroWave commit.
   *
   * Advertised in `a=x-ss-pyrowave.bitstream:` so a client can warn when its decoder was built
   * from a different bitstream revision.
   */
  constexpr std::string_view BITSTREAM_ID = "186f0393";

  /**
   * @brief RTSP DESCRIBE capability marker for PyroWave.
   *
   * No RTP payload type 99 is ever sent; the line only tells PyroWave clients that the host speaks
   * the record-framed protocol.
   */
  constexpr std::string_view DESCRIBE_RTPMAP = "a=rtpmap:99 PYROWAVE/90000";

  /**
   * @brief RTSP DESCRIBE attribute prefix carrying #BITSTREAM_ID.
   */
  constexpr std::string_view DESCRIBE_BITSTREAM_ATTRIBUTE = "a=x-ss-pyrowave.bitstream:";

  /**
   * @brief RTSP ANNOUNCE attribute whose presence selects record framing.
   */
  constexpr std::string_view ANNOUNCE_ADAPTIVE_FEC = "x-ss-video[0].pyrowaveAdaptiveFec";

  /**
   * @brief RTSP ANNOUNCE attribute carrying a bitmask of `FEATURE_*` values.
   */
  constexpr std::string_view ANNOUNCE_FEATURES = "x-ss-video[0].pyrowaveFeatures";

  /// Client parses record framing with in-band padding records.
  constexpr std::uint32_t FEATURE_RECORD_FRAMING = 0x1;

  /// First word of an in-band padding record: `0xFFFFFFFF, N, N zero words`.
  constexpr std::uint32_t PADDING_MAGIC = 0xFFFFFFFFu;

  /// Size of the smallest padding record (magic and word count, no zero words).
  constexpr std::size_t PADDING_RECORD_MIN_BYTES = 8;

  /// Net bytes a negotiated packet size spends on headers: the 32-byte video packet header also
  /// covers the 16-byte RTP header allowance, so `payload = packetSize - SHARD_OVERHEAD_BYTES`.
  constexpr int SHARD_OVERHEAD_BYTES = 16;

  /// Size of the short video frame header that precedes the frame in the first shard.
  constexpr std::size_t FRAME_HEADER_BYTES = 8;

  /// `NV_VIDEO_PACKET::extraFlags` bit marking a shard whose video payload starts with a record.
  constexpr std::uint8_t EXTRA_FLAG_RECORD_START = 0x80;

  /// The video header addresses FEC blocks in two bits.
  constexpr std::size_t MAX_FEC_BLOCKS = 1u << 2;

  /// The video header addresses data shards within an FEC block in ten bits.
  constexpr std::size_t MAX_DATA_SHARDS_PER_BLOCK = (1u << 10) - 1;

  /// PyroWave sequence headers occupy two 32-bit words.
  constexpr std::size_t SEQUENCE_HEADER_BYTES = 2 * sizeof(std::uint32_t);

  /// PyroWave block headers occupy two 32-bit words.
  constexpr std::size_t BLOCK_HEADER_BYTES = 2 * sizeof(std::uint32_t);
}  // namespace pyrowave::protocol
