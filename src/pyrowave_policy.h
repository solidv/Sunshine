/**
 * @file src/pyrowave_policy.h
 * @brief Pure PyroWave host policy: frame framing selection and record layout.
 *
 * Nothing here touches a GPU or Sunshine's runtime, so the unit tests exercise the same code the
 * encoder uses. See `pyrowave_protocol.h` for the wire constants.
 */
#pragma once

// standard includes
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace pyrowave::policy {
  /**
   * @brief Frame framing understood by the PyroWave clients.
   */
  enum class framing_e {
    records,  ///< Concatenated PyroWave records with padding aligned to RTP shards.
    length_prefixed,  ///< `[u32 count] { [u32 size] [packet] }`, for the older compatibility clients.
  };

  /**
   * @brief Choose the frame framing for a PyroWave session.
   *
   * Record framing is selected by the presence of `x-ss-video[0].pyrowaveAdaptiveFec` (any value)
   * or by `FEATURE_RECORD_FRAMING` in `x-ss-video[0].pyrowaveFeatures`. Clients that send neither
   * keep the length-prefixed compatibility framing.
   *
   * @param client_sent_adaptive_fec True when the client sent the adaptive-FEC attribute.
   * @param client_features Value of the client's feature bitmask, if it sent one.
   * @return Framing to use for every frame of the session.
   */
  framing_e select_framing(bool client_sent_adaptive_fec, std::optional<std::uint32_t> client_features);

  /**
   * @brief Video frame bytes carried by one RTP shard for the negotiated packet size.
   *
   * The RTP layer slices a frame into `packetSize - 16` byte payloads, with the 8-byte short frame
   * header in front of the first one. Record alignment additionally requires the payload to be a
   * multiple of four bytes and large enough to hold the frame header, a sequence header and a
   * minimal padding record.
   *
   * @param packetsize Negotiated maximum packet payload in bytes.
   * @return Shard payload in bytes, or 0 when record alignment is impossible.
   */
  std::size_t shard_payload_bytes(int packetsize);

  /**
   * @brief Number of blocks in PyroWave's coarsest wavelet level for a frame size.
   *
   * PyroWave indexes these blocks first. A receiver cannot decode a frame that lost any of them, so
   * the host sends them before the finer blocks and protects them with FEC.
   *
   * @param width Frame width in pixels.
   * @param height Frame height in pixels.
   * @return Number of coarsest-level blocks.
   */
  std::uint32_t coarse_block_count(std::uint32_t width, std::uint32_t height);

  /**
   * @brief Frame payload capacity from the negotiated packet size and FEC block layout.
   *
   * The video header describes at most four FEC blocks of at most 1023 data shards each, and the
   * FEC-enabled path stops at 3000 shards so the critical block has parity headroom.
   *
   * @param packetsize Negotiated maximum packet payload in bytes.
   * @param critical_fec True when the critical block carries parity.
   * @return Maximum frame payload in bytes, or 0 for an unusable packet size.
   */
  std::size_t max_frame_bytes(int packetsize, bool critical_fec = false);

  /**
   * @brief Encoder bitstream budget that leaves room for the transport framing.
   *
   * @param packetsize Negotiated maximum packet payload in bytes.
   * @param length_prefixed True when each PyroWave packet carries a length prefix.
   * @param critical_fec True when the critical block carries parity.
   * @return Maximum bitstream size in bytes, aligned down to a 32-bit word, or 0 when unusable.
   */
  std::size_t max_bitstream_bytes(int packetsize, bool length_prefixed, bool critical_fec = false);

  /**
   * @brief Maximum per-packet bytes a PyroWave datagram spends outside the negotiated packet size.
   *
   * The negotiated packet size covers the RTP header allowance and the video packet header, but
   * not the Ethernet preamble/gap/FCS, IP and UDP headers, or the AES-GCM prefix of an encrypted
   * video packet. Summing these gives the bytes each packet occupies on the wire in addition to
   * #packetsize.
   */
  constexpr std::size_t MAX_WIRE_PACKET_OVERHEAD_BYTES = 16 + 38 + 40 + 8 + 32;

  /**
   * @brief Per-packet bytes a PyroWave datagram spends outside the negotiated packet size.
   *
   * @param ipv6 True for an IPv6 peer (40-byte header) instead of IPv4 (20-byte header).
   * @param video_encryption True when video packets carry the AES-GCM prefix.
   * @return Bytes per packet, including Ethernet framing, IP, UDP, the RTP allowance and the
   *         optional encryption prefix.
   */
  std::size_t wire_packet_overhead_bytes(bool ipv6, bool video_encryption);

  /**
   * @brief Wire cost model for a record-framed PyroWave session.
   */
  struct wire_budget_t {
    int packetsize = 0;  ///< Negotiated maximum packet payload in bytes.
    int critical_fec_percentage = 0;  ///< Parity percentage for the loss-critical head block.
    std::size_t min_parity_shards = 2;  ///< Minimum parity shards the head block carries.
    std::size_t packet_overhead_bytes = MAX_WIRE_PACKET_OVERHEAD_BYTES;  ///< Per-packet bytes beyond #packetsize.
  };

  /**
   * @brief Largest critical-block parity the negotiated FEC layout can describe.
   *
   * A critical block may occupy at most 255 Reed-Solomon shards. Its parity is the configured
   * percentage of its data shards, but at least #wire_budget_t::min_parity_shards. The image
   * budget must reserve the worst case because the head size is known only after encoding.
   *
   * @param transport Wire budget.
   * @param packets Number of data packets in the frame.
   * @return Parity shards to reserve, or 0 when head protection is disabled.
   */
  std::size_t max_critical_parity_shards(const wire_budget_t &transport, std::size_t packets);

  /**
   * @brief Video traffic left after audio and control data.
   *
   * The client's configured bitrate covers all stream traffic. Audio carries 50% Reed-Solomon
   * parity (four data and two parity shards), so its wire cost is 1.5 times the configured audio
   * bitrate. A fixed allowance covers control traffic and other packet overhead.
   *
   * @param configured_kbps Total wire budget requested by the client, in kbps.
   * @param audio_channels Number of audio channels.
   * @param high_quality True for 256 kbps per channel, false for 96 kbps.
   * @return Video wire budget in kbps, or 0 when audio and control consume the whole budget.
   */
  std::int64_t video_wire_budget_kbps(std::int64_t configured_kbps, int audio_channels, bool high_quality);

  /**
   * @brief Image bytes that fit a per-frame wire allowance.
   *
   * Deducts the per-packet wire overhead and the reserved critical-block parity, then the short
   * frame header, and aligns the result down to a 32-bit word.
   *
   * @param wire_bytes Wire allowance for the frame's video packets.
   * @param transport Wire budget.
   * @return Encoder bitstream budget in bytes, or 0 when the allowance cannot carry a frame.
   */
  std::size_t frame_payload_bytes(std::size_t wire_bytes, const wire_budget_t &transport);

  /**
   * @brief Wire bytes a frame occupies for a given encoder bitstream budget.
   *
   * Inverse of frame_payload_bytes() for diagnostics and tests. The result is rounded up to whole
   * packets and includes the reserved critical-block parity and per-packet overhead.
   *
   * @param frame_bytes Encoder bitstream budget in bytes.
   * @param transport Wire budget.
   * @return Wire bytes, or 0 when the packet size is unusable or @p frame_bytes is 0.
   */
  std::size_t wire_bytes_for_frame(std::size_t frame_bytes, const wire_budget_t &transport);

  /**
   * @brief Statistics from laying out one record-framed frame.
   */
  struct record_frame_stats_t {
    std::size_t block_records = 0;  ///< Block records written, excluding the sequence header.
    std::size_t padding_records = 0;  ///< Padding records written.
    std::size_t padding_bytes = 0;  ///< Total bytes spent on padding records.
    std::size_t oversized_records = 0;  ///< Records too large to share a shard with padding.
    std::size_t reordered_records = 0;  ///< Records placed ahead of an earlier record to fill a shard.

    /// Frame bytes through the last record of the coarsest wavelet level (at least the sequence
    /// header). The shards holding them are the frame's critical shards.
    std::size_t critical_bytes = 0;
  };

  /**
   * @brief Append a record-framed frame to @p out.
   *
   * @p bitstream is one PyroWave packet holding the whole frame: the 8-byte sequence header
   * followed by every block record (the encoder's packetizer called with an unlimited boundary).
   * The layout lets a receiver decode a frame that lost packets:
   *
   * 1. The sequence header.
   * 2. The coarsest wavelet level, which a receiver cannot do without: its shards
   *    (`critical_bytes`) are the ones the RTP layer protects with parity.
   * 3. Every other record.
   *
   * Within groups 2 and 3, oversized records (too large to share a shard with a padding record)
   * come first, in frame order, and may span shards. The group's other records follow, packed
   * first-fit without crossing a shard boundary; a padding record fills a shard remainder only
   * when no record of the group fits. None of them leaves a 4-byte remainder (too small for a
   * padding record), so every shard after the oversized records starts with a record.
   *
   * @param bitstream Encoded PyroWave frame.
   * @param shard_payload Result of shard_payload_bytes(); 0 copies the bitstream unchanged.
   * @param out Receives the frame bytes, appended after any existing content.
   * @param frame_limit If nonzero, padding is omitted when the result would exceed this capacity.
   * @return Statistics, or std::nullopt when @p bitstream is not a valid PyroWave frame or already
   *         exceeds @p frame_limit (nothing is appended then).
   */
  std::optional<record_frame_stats_t> write_record_frame(
    std::span<const std::uint8_t> bitstream,
    std::size_t shard_payload,
    std::vector<std::uint8_t> &out,
    std::size_t frame_limit = 0
  );

  /**
   * @brief Result of parsing a record-framed frame the way a receiver does.
   */
  struct record_frame_info_t {
    bool valid = false;  ///< True when the whole frame parsed.
    std::string error;  ///< Parse failure description.
    bool has_sequence_header = false;  ///< True once a sequence header was seen.
    std::uint32_t width = 0;  ///< Frame width from the sequence header.
    std::uint32_t height = 0;  ///< Frame height from the sequence header.
    bool chroma444 = false;  ///< True for 4:4:4 chroma.
    std::uint32_t total_blocks = 0;  ///< Block count announced by the sequence header.
    std::size_t block_records = 0;  ///< Block records parsed.
    std::size_t padding_records = 0;  ///< Padding records parsed.
    std::size_t padding_bytes = 0;  ///< Bytes spent on padding records.
    std::size_t misaligned_records = 0;  ///< Ordinary records that cross a shard boundary.
    std::size_t out_of_order_records = 0;  ///< Block records with a lower index than the previous one.
    std::size_t oversized_records = 0;  ///< Records too large to share a shard with padding.
    std::size_t late_oversized_records = 0;  ///< Oversized records after an ordinary record of their level.
    std::size_t late_coarse_records = 0;  ///< Coarsest-level records after a finer one.
    std::size_t critical_bytes = 0;  ///< Frame bytes through the last coarsest-level record.
    std::vector<std::uint8_t> stripped;  ///< The frame with padding records removed.
  };

  /**
   * @brief Parse a record-framed frame the way a receiver does.
   *
   * @param frame Frame bytes (without the short video frame header).
   * @param shard_payload Shard size used for alignment statistics; 0 skips them.
   * @return Parse result; `valid` is false with `error` set when the frame must be dropped.
   */
  record_frame_info_t inspect_record_frame(std::span<const std::uint8_t> frame, std::size_t shard_payload);

  /**
   * @brief Which shards of a record-framed frame start with a record.
   *
   * Shard 0 holds the frame from its first byte (after the short video frame header) and so starts
   * with the sequence header. Shard `s > 0` starts at frame byte `s * shard_payload -
   * FRAME_HEADER_BYTES`. A record here is a sequence header, a block record or a padding record.
   *
   * @param frame Frame bytes (without the short video frame header).
   * @param shard_payload Shard size in bytes.
   * @return One entry per shard, or an empty vector when the frame cannot be walked.
   */
  std::vector<bool> record_start_shards(std::span<const std::uint8_t> frame, std::size_t shard_payload);
}  // namespace pyrowave::policy
