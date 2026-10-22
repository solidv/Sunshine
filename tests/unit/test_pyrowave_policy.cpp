/**
 * @file tests/unit/test_pyrowave_policy.cpp
 * @brief Test src/pyrowave_policy.*
 */

// test includes
#include "../tests_common.h"

// standard includes
#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// local includes
#include <src/pyrowave_policy.h>
#include <src/pyrowave_protocol.h>

namespace {
  /**
   * @brief Append a little-endian 32-bit word.
   *
   * @param out Destination buffer.
   * @param value Value to append.
   */
  void append_u32(std::vector<std::uint8_t> &out, std::uint32_t value) {
    out.push_back(std::uint8_t(value & 0xff));
    out.push_back(std::uint8_t((value >> 8) & 0xff));
    out.push_back(std::uint8_t((value >> 16) & 0xff));
    out.push_back(std::uint8_t((value >> 24) & 0xff));
  }

  /**
   * @brief Append a PyroWave sequence header.
   *
   * @param out Destination buffer.
   * @param width Frame width in pixels.
   * @param height Frame height in pixels.
   * @param total_blocks Block count announced by the header.
   */
  void append_sequence_header(std::vector<std::uint8_t> &out, std::uint32_t width, std::uint32_t height, std::uint32_t total_blocks) {
    append_u32(out, ((height - 1) << 14) | (width - 1) | 0x80000000u);
    append_u32(out, total_blocks & 0xffffffu);
  }

  /**
   * @brief Append a PyroWave block record filled with a block-dependent pattern.
   *
   * @param out Destination buffer.
   * @param block_index PyroWave block index.
   * @param bytes Record size in bytes; at least 8 and a multiple of 4.
   */
  void append_block_record(std::vector<std::uint8_t> &out, std::uint32_t block_index, std::size_t bytes) {
    append_u32(out, std::uint32_t(bytes / 4) << 16);
    append_u32(out, block_index << 8);
    for (std::size_t i = 8; i < bytes; ++i) {
      out.push_back(std::uint8_t((block_index + i) & 0xff));
    }
  }

  /**
   * @brief One block record to place in a synthetic frame.
   */
  struct block_spec_t {
    std::uint32_t index;  ///< PyroWave block index.
    std::size_t bytes;  ///< Record size in bytes.
  };

  /**
   * @brief Build a synthetic PyroWave frame.
   *
   * @param width Frame width in pixels.
   * @param height Frame height in pixels.
   * @param blocks Block records in encoder order.
   * @return Encoded frame bytes.
   */
  std::vector<std::uint8_t> make_frame(std::uint32_t width, std::uint32_t height, const std::vector<block_spec_t> &blocks) {
    std::vector<std::uint8_t> frame;
    append_sequence_header(frame, width, height, std::uint32_t(blocks.size()));
    for (const auto &block : blocks) {
      append_block_record(frame, block.index, block.bytes);
    }
    return frame;
  }

  /**
   * @brief Build a frame with coarse (0..11) and fine (12+) records.
   *
   * 128x128 has 12 coarsest-level blocks, so indices below 12 are the critical level.
   *
   * @param coarse_bytes Size of each coarse record.
   * @param coarse_count Number of coarse records.
   * @param fine_sizes Sizes of the fine records, starting at block index 12.
   * @return Encoded frame bytes.
   */
  std::vector<std::uint8_t> make_two_level_frame(std::size_t coarse_bytes, std::size_t coarse_count, const std::vector<std::size_t> &fine_sizes) {
    std::vector<block_spec_t> blocks;
    for (std::size_t i = 0; i < coarse_count; ++i) {
      blocks.push_back({std::uint32_t(i), coarse_bytes});
    }
    for (std::size_t i = 0; i < fine_sizes.size(); ++i) {
      blocks.push_back({std::uint32_t(coarse_count + i), fine_sizes[i]});
    }
    return make_frame(128, 128, blocks);
  }

  /**
   * @brief One record start found by walking a record-framed payload.
   */
  struct record_start_t {
    std::size_t offset;  ///< Frame byte offset.
    std::size_t bytes;  ///< Record size.
    bool padding;  ///< True for a padding record.
  };

  /**
   * @brief Walk a record-framed payload and collect every record.
   *
   * @param frame Record-framed bytes.
   * @return Records in payload order, or an empty vector when the payload cannot be walked.
   */
  std::vector<record_start_t> walk_records(std::span<const std::uint8_t> frame) {
    std::vector<record_start_t> records;
    for (std::size_t position = 0; position < frame.size();) {
      if (frame.size() - position < 8) {
        return {};
      }
      const auto read = [&](std::size_t at) {
        return std::uint32_t(frame[at]) | (std::uint32_t(frame[at + 1]) << 8) |
               (std::uint32_t(frame[at + 2]) << 16) | (std::uint32_t(frame[at + 3]) << 24);
      };
      const auto word0 = read(position);
      const auto word1 = read(position + 4);
      std::size_t bytes;
      const bool padding = word0 == pyrowave::protocol::PADDING_MAGIC;
      if (padding) {
        bytes = pyrowave::protocol::PADDING_RECORD_MIN_BYTES + std::size_t(word1) * 4;
      } else if ((word0 >> 31) != 0) {
        bytes = pyrowave::protocol::SEQUENCE_HEADER_BYTES;
      } else {
        bytes = std::size_t((word0 >> 16) & 0xfff) * 4;
      }
      if (bytes < 8 || bytes > frame.size() - position) {
        return {};
      }
      records.push_back({position, bytes, padding});
      position += bytes;
    }
    return records;
  }
}  // namespace

TEST(PyroWaveCoarseBlockCountTests, RoundsExtentsUpLikePyroWave) {
  using pyrowave::policy::coarse_block_count;

  EXPECT_EQ(coarse_block_count(128, 128), 12u);
  EXPECT_EQ(coarse_block_count(1, 1), 12u);
  EXPECT_EQ(coarse_block_count(1920, 1080), 48u);
  EXPECT_EQ(coarse_block_count(2560, 1440), 72u);
  EXPECT_EQ(coarse_block_count(3840, 2160), 144u);
}

TEST(PyroWaveShardPayloadTests, DerivesPayloadFromPacketSize) {
  using pyrowave::policy::shard_payload_bytes;

  EXPECT_EQ(shard_payload_bytes(1392), 1376u);
  EXPECT_EQ(shard_payload_bytes(1024), 1008u);
  EXPECT_EQ(shard_payload_bytes(40), 24u);
}

TEST(PyroWaveShardPayloadTests, RejectsSizesThatCannotAlign) {
  using pyrowave::policy::shard_payload_bytes;

  EXPECT_EQ(shard_payload_bytes(0), 0u);
  EXPECT_EQ(shard_payload_bytes(-1), 0u);
  EXPECT_EQ(shard_payload_bytes(16), 0u);
  EXPECT_EQ(shard_payload_bytes(32), 0u);  // Payload below frame header + sequence header + padding.
  EXPECT_EQ(shard_payload_bytes(1390), 0u);  // Not a multiple of four.
}

TEST(PyroWaveSelectFramingTests, AdaptiveFecPresenceSelectsRecords) {
  using pyrowave::policy::framing_e;
  using pyrowave::policy::select_framing;

  EXPECT_EQ(select_framing(true, std::nullopt), framing_e::records);
  EXPECT_EQ(select_framing(true, 0u), framing_e::records);
}

TEST(PyroWaveSelectFramingTests, FeatureBitSelectsRecords) {
  using pyrowave::policy::framing_e;
  using pyrowave::policy::select_framing;

  EXPECT_EQ(select_framing(false, pyrowave::protocol::FEATURE_RECORD_FRAMING), framing_e::records);
  EXPECT_EQ(select_framing(false, pyrowave::protocol::FEATURE_RECORD_FRAMING | 0x2u), framing_e::records);
}

TEST(PyroWaveSelectFramingTests, ClientsWithoutTheFeatureKeepLengthPrefixes) {
  using pyrowave::policy::framing_e;
  using pyrowave::policy::select_framing;

  EXPECT_EQ(select_framing(false, std::nullopt), framing_e::length_prefixed);
  EXPECT_EQ(select_framing(false, 0u), framing_e::length_prefixed);
  EXPECT_EQ(select_framing(false, 0x2u), framing_e::length_prefixed);
  EXPECT_EQ(select_framing(false, 0x4u), framing_e::length_prefixed);
}

TEST(PyroWaveFrameCapacityTests, CapsFramesAtTheProtocolBlockLimit) {
  using pyrowave::policy::max_frame_bytes;

  EXPECT_EQ(max_frame_bytes(1392, true), 3000u * 1376u - 8u);
  EXPECT_EQ(max_frame_bytes(1392, false), 4000u * 1376u - 8u);
  EXPECT_EQ(max_frame_bytes(16, true), 0u);
  EXPECT_EQ(max_frame_bytes(0, true), 0u);
}

TEST(PyroWaveFrameCapacityTests, ReservesLengthPrefixesForCompatibilityFraming) {
  using pyrowave::policy::max_bitstream_bytes;

  const auto records = max_bitstream_bytes(1392, false, true);
  EXPECT_EQ(records, (3000u * 1376u - 8u) & ~std::size_t(3));

  const auto prefixed = max_bitstream_bytes(1392, true, true);
  EXPECT_LT(prefixed, records);
  EXPECT_EQ(prefixed, (((records - 4u) / 12u) * 8u) & ~std::size_t(3));
}

TEST(PyroWaveRecordFrameTests, WritesSequenceHeaderFirst) {
  const auto frame = make_two_level_frame(16, 12, {16, 16});
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 1376, out);
  ASSERT_TRUE(stats.has_value());
  ASSERT_FALSE(out.empty());

  // The sequence header keeps its extended bit, which no packet count can have.
  EXPECT_NE(out[3] & 0x80, 0);
  EXPECT_EQ(stats->block_records, 14u);
  EXPECT_EQ(stats->critical_bytes, 8u + 12u * 16u);  // Sequence header plus the coarsest level.
}

TEST(PyroWaveRecordFrameTests, KeepsAllBlocksAndReportsCriticalBytes) {
  // 20-byte records force padding in 32-byte shards.
  const auto frame = make_two_level_frame(20, 12, {12, 20});
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 32, out);
  ASSERT_TRUE(stats.has_value());
  ASSERT_FALSE(out.empty());

  const auto info = pyrowave::policy::inspect_record_frame(out, 32);
  ASSERT_TRUE(info.valid) << info.error;
  EXPECT_TRUE(info.has_sequence_header);
  EXPECT_EQ(info.width, 128u);
  EXPECT_EQ(info.height, 128u);
  EXPECT_EQ(info.block_records, 14u);
  EXPECT_EQ(info.total_blocks, 14u);
  EXPECT_EQ(info.late_coarse_records, 0u);
  EXPECT_EQ(info.misaligned_records, 0u);
  EXPECT_EQ(info.stripped, frame);

  ASSERT_GT(stats->padding_records, 0u);
  EXPECT_EQ(stats->padding_records, info.padding_records);
  EXPECT_EQ(stats->padding_bytes, info.padding_bytes);
  EXPECT_EQ(stats->critical_bytes, info.critical_bytes);
  EXPECT_GE(stats->critical_bytes, pyrowave::protocol::SEQUENCE_HEADER_BYTES);
  EXPECT_LT(stats->critical_bytes, out.size());
}

TEST(PyroWaveRecordFrameTests, ReordersRecordsToFillShards) {
  // No coarse records (indices start at 12), so the fine group starts 24 bytes into a 40-byte
  // shard. The 20-byte record would leave 4 bytes, which is too small for padding, so the packer
  // takes the later 12-byte record first.
  const std::vector<block_spec_t> blocks = {{12, 20}, {13, 12}};
  const auto frame = make_frame(128, 128, blocks);
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 40, out);
  ASSERT_TRUE(stats.has_value());

  EXPECT_EQ(stats->reordered_records, 1u);
  EXPECT_GT(stats->padding_records, 0u);

  const auto info = pyrowave::policy::inspect_record_frame(out, 40);
  ASSERT_TRUE(info.valid) << info.error;
  EXPECT_EQ(info.misaligned_records, 0u);
  EXPECT_EQ(info.block_records, 2u);
  EXPECT_EQ(info.out_of_order_records, 1u);  // The packer wrote block 13 before block 12.
}

TEST(PyroWaveRecordFrameTests, MarksEveryShardThatStartsWithARecord) {
  const auto frame = make_two_level_frame(20, 12, {12, 20});
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 32, out);
  ASSERT_TRUE(stats.has_value());

  const auto records = walk_records(out);
  ASSERT_FALSE(records.empty());

  const auto starts = pyrowave::policy::record_start_shards(out, 32);
  const std::size_t expected_shards = (out.size() + pyrowave::protocol::FRAME_HEADER_BYTES + 31) / 32;
  ASSERT_EQ(starts.size(), expected_shards);

  // No record is oversized here, so every shard begins with a record.
  EXPECT_TRUE(std::all_of(starts.begin(), starts.end(), [](bool start) {
    return start;
  }));

  for (const auto &record : records) {
    const auto shard = (record.offset + pyrowave::protocol::FRAME_HEADER_BYTES) / 32;
    ASSERT_LT(shard, starts.size());
    EXPECT_TRUE(starts[shard]);
  }
}

TEST(PyroWaveRecordFrameTests, LetsOversizedRecordsSpanShards) {
  const std::vector<block_spec_t> blocks = {{0, 60}, {1, 8}};
  const auto frame = make_frame(128, 128, blocks);
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 32, out);
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->oversized_records, 1u);

  const auto info = pyrowave::policy::inspect_record_frame(out, 32);
  ASSERT_TRUE(info.valid) << info.error;
  EXPECT_EQ(info.oversized_records, 1u);
  EXPECT_EQ(info.stripped, frame);
}

TEST(PyroWaveRecordFrameTests, FallsBackToStrictOrderWhenPaddingDoesNotFit) {
  const auto frame = make_two_level_frame(20, 12, {12, 20});
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 32, out, frame.size());
  ASSERT_TRUE(stats.has_value());

  // Padding would exceed the transport budget, so the frame keeps encoder order and costs no
  // extra bytes. The critical prefix still ends after the coarsest level.
  EXPECT_EQ(stats->padding_records, 0u);
  EXPECT_EQ(stats->reordered_records, 0u);
  EXPECT_EQ(out, frame);
  EXPECT_EQ(stats->critical_bytes, 8u + 12u * 20u);
  EXPECT_EQ(stats->block_records, 14u);
}

TEST(PyroWaveRecordFrameTests, CopiesTheBitstreamWithoutAlignment) {
  const auto frame = make_two_level_frame(20, 12, {12, 20});
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 0, out);
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(out, frame);
  EXPECT_EQ(stats->critical_bytes, 0u);
  EXPECT_EQ(stats->block_records, 14u);
}

TEST(PyroWaveRecordFrameTests, AcceptsAFrameWithOnlyASequenceHeader) {
  const auto frame = make_frame(128, 128, {});
  std::vector<std::uint8_t> out;

  const auto stats = pyrowave::policy::write_record_frame(frame, 32, out);
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(out, frame);
  EXPECT_EQ(stats->block_records, 0u);
  EXPECT_EQ(stats->critical_bytes, 8u);
}

TEST(PyroWaveRecordFrameTests, RejectsInvalidBitstreams) {
  std::vector<std::uint8_t> out;

  EXPECT_FALSE(pyrowave::policy::write_record_frame({}, 32, out).has_value());

  // Not a sequence header (extended bit clear).
  const std::vector<std::uint8_t> no_header(8, 0);
  EXPECT_FALSE(pyrowave::policy::write_record_frame(no_header, 32, out).has_value());

  // Truncated block header.
  auto frame = make_frame(128, 128, {});
  frame.resize(12);
  EXPECT_FALSE(pyrowave::policy::write_record_frame(frame, 32, out).has_value());

  // Block record with payload_words below the header size.
  frame = make_frame(128, 128, {});
  append_u32(frame, 1u << 16);
  append_u32(frame, 0);
  EXPECT_FALSE(pyrowave::policy::write_record_frame(frame, 32, out).has_value());

  // Block record larger than the frame.
  frame = make_frame(128, 128, {});
  append_u32(frame, 100u << 16);
  append_u32(frame, 0);
  EXPECT_FALSE(pyrowave::policy::write_record_frame(frame, 32, out).has_value());

  // Frame already over the transport budget.
  const auto valid = make_two_level_frame(16, 2, {16});
  EXPECT_FALSE(pyrowave::policy::write_record_frame(valid, 32, out, valid.size() - 1).has_value());
}

TEST(PyroWaveInspectRecordFrameTests, RejectsPaddingPastTheEnd) {
  std::vector<std::uint8_t> frame;
  append_u32(frame, pyrowave::protocol::PADDING_MAGIC);
  append_u32(frame, 100);

  const auto info = pyrowave::policy::inspect_record_frame(frame, 32);
  EXPECT_FALSE(info.valid);
  EXPECT_NE(info.error.find("padding"), std::string::npos);
}

TEST(PyroWaveInspectRecordFrameTests, RejectsASecondSequenceHeader) {
  std::vector<std::uint8_t> frame;
  append_sequence_header(frame, 128, 128, 0);
  append_sequence_header(frame, 128, 128, 0);

  const auto info = pyrowave::policy::inspect_record_frame(frame, 32);
  EXPECT_FALSE(info.valid);
  EXPECT_NE(info.error.find("second sequence header"), std::string::npos);
}

TEST(PyroWaveInspectRecordFrameTests, RejectsBlocksBeforeTheSequenceHeader) {
  std::vector<std::uint8_t> frame;
  append_block_record(frame, 0, 8);

  const auto info = pyrowave::policy::inspect_record_frame(frame, 32);
  EXPECT_FALSE(info.valid);
  EXPECT_NE(info.error.find("before the sequence header"), std::string::npos);
}

TEST(PyroWaveRecordStartShardsTests, RejectsUnwalkableFrames) {
  const auto frame = make_two_level_frame(16, 12, {16});

  EXPECT_TRUE(pyrowave::policy::record_start_shards(frame, 0).empty());
  EXPECT_TRUE(pyrowave::policy::record_start_shards({}, 32).empty());

  std::vector<std::uint8_t> truncated(frame.begin(), frame.begin() + 10);
  EXPECT_TRUE(pyrowave::policy::record_start_shards(truncated, 32).empty());

  std::vector<std::uint8_t> bad_block;
  append_sequence_header(bad_block, 128, 128, 1);
  append_u32(bad_block, 1u << 16);
  append_u32(bad_block, 0);
  EXPECT_TRUE(pyrowave::policy::record_start_shards(bad_block, 32).empty());
}

TEST(PyroWaveRecordStartShardsTests, FlagsTheFirstShard) {
  const auto frame = make_two_level_frame(16, 12, {16});
  const auto starts = pyrowave::policy::record_start_shards(frame, pyrowave::policy::shard_payload_bytes(1392));

  ASSERT_FALSE(starts.empty());
  EXPECT_TRUE(starts.front());
}

TEST(PyroWaveWireBudgetTests, CountsHeadersOutsideTheNegotiatedPacketSize) {
  using pyrowave::policy::MAX_WIRE_PACKET_OVERHEAD_BYTES;
  using pyrowave::policy::wire_packet_overhead_bytes;

  EXPECT_EQ(wire_packet_overhead_bytes(false, false), 82u);
  EXPECT_EQ(wire_packet_overhead_bytes(true, false), 102u);
  EXPECT_EQ(wire_packet_overhead_bytes(false, true), 114u);
  EXPECT_EQ(wire_packet_overhead_bytes(true, true), 134u);
  EXPECT_EQ(MAX_WIRE_PACKET_OVERHEAD_BYTES, 134u);
}

TEST(PyroWaveWireBudgetTests, DeductsAudioFecAndControlFromTheConfiguredBitrate) {
  using pyrowave::policy::video_wire_budget_kbps;

  EXPECT_EQ(video_wire_budget_kbps(0, 2, true), 0);
  EXPECT_EQ(video_wire_budget_kbps(-1, 2, true), 0);
  EXPECT_EQ(video_wire_budget_kbps(20000, 2, true), 18732);
  EXPECT_EQ(video_wire_budget_kbps(20000, 8, true), 16428);
  EXPECT_EQ(video_wire_budget_kbps(20000, 8, false), 18348);
  EXPECT_EQ(video_wire_budget_kbps(5000, 8, true), 1428);
  EXPECT_EQ(video_wire_budget_kbps(1000, 8, true), 0);
  EXPECT_EQ(video_wire_budget_kbps(20000, -1, true), 19500);
}

TEST(PyroWaveWireBudgetTests, ReservesTheLargestCriticalBlockParity) {
  using pyrowave::policy::max_critical_parity_shards;
  using pyrowave::policy::wire_budget_t;

  wire_budget_t transport {1392, 0, 2, 134};
  EXPECT_EQ(max_critical_parity_shards(transport, 100), 0u);

  transport.critical_fec_percentage = 20;
  EXPECT_EQ(max_critical_parity_shards(transport, 0), 0u);
  EXPECT_EQ(max_critical_parity_shards(transport, 10), 2u);
  EXPECT_EQ(max_critical_parity_shards(transport, 100), 20u);
  EXPECT_EQ(max_critical_parity_shards(transport, 400), 43u);

  transport.min_parity_shards = 5;
  EXPECT_EQ(max_critical_parity_shards(transport, 10), 5u);

  transport.critical_fec_percentage = 100;
  EXPECT_EQ(max_critical_parity_shards(transport, 255), 127u);

  transport.min_parity_shards = 256;
  EXPECT_EQ(max_critical_parity_shards(transport, 255), 0u);
}

TEST(PyroWaveWireBudgetTests, ReservesWireOverheadAndParityForTheImageBudget) {
  using pyrowave::policy::frame_payload_bytes;
  using pyrowave::policy::wire_budget_t;

  const wire_budget_t transport {1392, 20, 2, 102};
  constexpr std::size_t wire_packet = 1392 + 102;

  // 100 packets reserve 20 parity shards, leaving 80 data shards of 1376 bytes.
  EXPECT_EQ(frame_payload_bytes(100 * wire_packet, transport), 80u * 1376u - 8u);
  EXPECT_EQ(frame_payload_bytes(0, transport), 0u);
  EXPECT_EQ(frame_payload_bytes(wire_packet - 1, transport), 0u);

  auto unusable = transport;
  unusable.packetsize = 16;
  EXPECT_EQ(frame_payload_bytes(100 * wire_packet, unusable), 0u);

  auto fully_protected = transport;
  fully_protected.critical_fec_percentage = 100;
  EXPECT_EQ(frame_payload_bytes(2 * wire_packet, fully_protected), 0u);
}

TEST(PyroWaveWireBudgetTests, RoundTripsWithinTheWireAllowance) {
  using pyrowave::policy::frame_payload_bytes;
  using pyrowave::policy::wire_budget_t;
  using pyrowave::policy::wire_bytes_for_frame;

  const wire_budget_t transport {1392, 20, 2, 134};
  for (const auto wire : {1u, 1000u, 100000u, 1000000u, 2000000u}) {
    EXPECT_LE(wire_bytes_for_frame(frame_payload_bytes(wire, transport), transport), wire);
  }

  // 13752 image bytes need 10 data packets; the critical head reserves 2 parity shards.
  EXPECT_EQ(wire_bytes_for_frame(1376u * 10u - 8u, transport), 12u * (1392u + 134u));
  EXPECT_EQ(wire_bytes_for_frame(0, transport), 0u);

  auto unusable = transport;
  unusable.packetsize = 16;
  EXPECT_EQ(wire_bytes_for_frame(1376, unusable), 0u);
}
