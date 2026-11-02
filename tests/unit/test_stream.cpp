/**
 * @file tests/unit/test_stream.cpp
 * @brief Test src/stream.*
 */

// test includes
#include "../tests_common.h"

// standard includes
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// local includes
#include <src/network.h>

namespace stream {
  std::vector<uint8_t> concat_and_insert(uint64_t insert_size, uint64_t slice_size, const std::string_view &data1, const std::string_view &data2);
  std::optional<std::pair<std::uint16_t, std::string_view>> parse_control_packet(const ENetPacket &packet);
  enum class fec_split_fallback_e {
    none,  ///< The split was applied.
    no_tail,  ///< There is no head, or the head covers the whole frame.
    head_shard_limit,  ///< The protected head would exceed the Reed-Solomon shard limit.
    tail_block_limit,  ///< The bare tail would need more FEC blocks than the protocol can describe.
  };
  int split_fec_blocks_head_tail(const std::string_view &payload, size_t blocksize, size_t payload_blocksize, size_t head_stream_bytes, int head_fec_percentage, std::array<std::string_view, 4> &blocks, std::array<int, 4> &percentages, fec_split_fallback_e &fallback);
  size_t ratecontrol_packets_per_ms(const int percentage, const size_t blocksize);
  int pacing_percent_for_codec(bool is_pyrowave, int pacing_percent, int pyrowave_pacing_percent);
  int fec_percent_for_codec(bool is_pyrowave, int fec_percent, int pyrowave_critical_fec_percentage);
  std::uint16_t pyrowave_critical_packet_count(size_t head_stream_bytes, size_t payload_blocksize, size_t total_shards);
  size_t pyrowave_shard_count(size_t payload_bytes, size_t blocksize);
  bool test_receive_ping_endpoints(bool session_id_pings);
}  // namespace stream

TEST(StreamPingTests, KeepsLegacyAudioAndVideoSenderEndpointsSeparate) {
  for (int attempt = 0; attempt < 20; ++attempt) {
    ASSERT_TRUE(stream::test_receive_ping_endpoints(false)) << "Attempt " << attempt;
  }
}

TEST(StreamPingTests, KeepsSessionIdAudioAndVideoSenderEndpointsSeparate) {
  for (int attempt = 0; attempt < 20; ++attempt) {
    ASSERT_TRUE(stream::test_receive_ping_endpoints(true)) << "Attempt " << attempt;
  }
}

TEST(ConcatAndInsertTests, ConcatNoInsertionTest) {
  std::array<char, 2> b1 = {'a', 'b'};
  std::array<char, 3> b2 = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(0, 2, std::string_view {b1.data(), b1.size()}, std::string_view {b2.data(), b2.size()});
  auto expected = std::vector<uint8_t> {'a', 'b', 'c', 'd', 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, ConcatLargeStrideTest) {
  std::array<char, 2> b1 = {'a', 'b'};
  std::array<char, 3> b2 = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(1, b1.size() + b2.size() + 1, std::string_view {b1.data(), b1.size()}, std::string_view {b2.data(), b2.size()});
  auto expected = std::vector<uint8_t> {0, 'a', 'b', 'c', 'd', 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, ConcatSmallStrideTest) {
  std::array<char, 2> b1 = {'a', 'b'};
  std::array<char, 3> b2 = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(1, 1, std::string_view {b1.data(), b1.size()}, std::string_view {b2.data(), b2.size()});
  auto expected = std::vector<uint8_t> {0, 'a', 0, 'b', 0, 'c', 0, 'd', 0, 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ControlPacketTests, RejectsZeroLengthPacket) {
  net::packet_t packet {enet_packet_create(nullptr, 0, 0)};

  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(packet->data, nullptr);
  EXPECT_EQ(stream::parse_control_packet(*packet), std::nullopt);
}

TEST(ControlPacketTests, RejectsOneBytePacket) {
  const std::uint8_t data {0x06};
  net::packet_t packet {enet_packet_create(&data, sizeof(data), 0)};

  ASSERT_NE(packet, nullptr);
  EXPECT_EQ(stream::parse_control_packet(*packet), std::nullopt);
}

TEST(ControlPacketTests, AcceptsTypeWithoutPayload) {
  const std::array<std::uint8_t, 2> data {0x06, 0x02};
  net::packet_t packet {enet_packet_create(data.data(), data.size(), 0)};

  ASSERT_NE(packet, nullptr);
  auto message = stream::parse_control_packet(*packet);
  ASSERT_TRUE(message);
  EXPECT_EQ(message->first, 0x0206);
  EXPECT_TRUE(message->second.empty());
}

TEST(ControlPacketTests, AcceptsTypeAndPayload) {
  const std::array<std::uint8_t, 5> data {0x06, 0x02, 'a', 'b', 'c'};
  net::packet_t packet {enet_packet_create(data.data(), data.size(), 0)};

  ASSERT_NE(packet, nullptr);
  auto message = stream::parse_control_packet(*packet);
  ASSERT_TRUE(message);
  EXPECT_EQ(message->first, 0x0206);
  EXPECT_EQ(message->second, "abc");
}

namespace {
  /// Video payload bytes per packet, matching the RTP/video header overhead of a real stream.
  constexpr size_t test_blocksize = 64;
  constexpr size_t test_payload_blocksize = 56;
  /// Head parity matching the default PyroWave critical FEC percentage.
  constexpr int test_head_fec_percentage = 20;

  /**
   * @brief Assert that the split blocks tile the payload exactly, in order.
   *
   * @param payload Original payload the blocks were cut from.
   * @param blocks Blocks produced by split_fec_blocks_head_tail().
   * @param count Number of valid entries in @p blocks.
   */
  void expect_blocks_tile_payload(const std::string &payload, const std::array<std::string_view, 4> &blocks, int count) {
    size_t offset = 0;
    for (int x = 0; x < count; ++x) {
      EXPECT_EQ(static_cast<const void *>(blocks[x].data()), static_cast<const void *>(payload.data() + offset))
        << "block " << x << " is not contiguous";
      offset += blocks[x].size();
    }
    EXPECT_EQ(offset, payload.size());
  }
}  // namespace

TEST(SplitFecBlocksHeadTailTests, SplitsProtectedHeadFromBareTail) {
  const std::string payload(test_blocksize * 400, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::no_tail;

  // 1000 protected stream bytes cover ceil(1000 / 56) = 18 packets, so the 400-packet frame
  // leaves a 382-packet tail that needs two bare blocks (255 shards maximum each).
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, 1000, test_head_fec_percentage, blocks, percentages, fallback);

  ASSERT_EQ(count, 3);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::none);
  EXPECT_EQ(percentages[0], test_head_fec_percentage);
  EXPECT_EQ(percentages[1], 0);
  EXPECT_EQ(percentages[2], 0);
  EXPECT_EQ(blocks[0].size(), 18 * test_blocksize);
  expect_blocks_tile_payload(payload, blocks, count);
}

TEST(SplitFecBlocksHeadTailTests, UsesAllFourBlocksForALongTail) {
  const std::string payload(test_blocksize * 700, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::no_tail;

  // A 695-packet tail fills three bare blocks, which is the most the protocol allows.
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 5, test_head_fec_percentage, blocks, percentages, fallback);

  ASSERT_EQ(count, 4);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::none);
  EXPECT_EQ(percentages[0], test_head_fec_percentage);
  EXPECT_EQ(percentages[3], 0);
  expect_blocks_tile_payload(payload, blocks, count);
}

TEST(SplitFecBlocksHeadTailTests, AcceptsHeadExactlyAtShardLimit) {
  const std::string payload(test_blocksize * 425, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::no_tail;

  // 212 protected packets fill the 255-shard Reed-Solomon limit exactly at 20% parity.
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 212, test_head_fec_percentage, blocks, percentages, fallback);

  ASSERT_EQ(count, 2);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::none);
  EXPECT_EQ(percentages[0], test_head_fec_percentage);
  EXPECT_EQ(percentages[1], 0);
  EXPECT_EQ(blocks[0].size(), 212 * test_blocksize);
  expect_blocks_tile_payload(payload, blocks, count);
}

TEST(SplitFecBlocksHeadTailTests, AppliesConfiguredHeadPercentage) {
  const std::string payload(test_blocksize * 400, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::no_tail;

  // 170 protected packets fill the 255-shard Reed-Solomon limit exactly at 50% parity.
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 170, 50, blocks, percentages, fallback);

  ASSERT_EQ(count, 2);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::none);
  EXPECT_EQ(percentages[0], 50);
  EXPECT_EQ(percentages[1], 0);
}

TEST(SplitFecBlocksHeadTailTests, KeepsTheBlockSplitWithZeroHeadPercentage) {
  const std::string payload(test_blocksize * 400, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::no_tail;

  // 0 disables parity on every block but keeps the head/tail block layout and critical count.
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, 1000, 0, blocks, percentages, fallback);

  ASSERT_EQ(count, 3);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::none);
  EXPECT_EQ(percentages[0], 0);
  EXPECT_EQ(percentages[1], 0);
  EXPECT_EQ(percentages[2], 0);
  expect_blocks_tile_payload(payload, blocks, count);
}

TEST(SplitFecBlocksHeadTailTests, AcceptsTailExactlyAtBlockLimit) {
  const std::string payload(test_blocksize * 770, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::no_tail;

  // A 765-packet tail fills exactly three bare blocks, the most the protocol can describe.
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 5, test_head_fec_percentage, blocks, percentages, fallback);

  ASSERT_EQ(count, 4);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::none);
  EXPECT_EQ(percentages[0], test_head_fec_percentage);
  EXPECT_EQ(percentages[1], 0);
  EXPECT_EQ(percentages[2], 0);
  EXPECT_EQ(percentages[3], 0);
  expect_blocks_tile_payload(payload, blocks, count);
}

TEST(SplitFecBlocksHeadTailTests, RejectsEmptyHead) {
  const std::string payload(test_blocksize * 400, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::none;

  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, 0, test_head_fec_percentage, blocks, percentages, fallback), 0);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::no_tail);
}

TEST(SplitFecBlocksHeadTailTests, RejectsFrameWithoutTail) {
  const std::string payload(test_blocksize * 10, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::none;

  // The head covers the whole frame, so there is nothing to leave bare.
  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, payload.size(), test_head_fec_percentage, blocks, percentages, fallback), 0);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::no_tail);
}

TEST(SplitFecBlocksHeadTailTests, RejectsProtectedHeadBeyondShardLimit) {
  const std::string payload(test_blocksize * 400, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::none;

  // 213 packets need more than 20% parity to stay within the 255-shard Reed-Solomon limit.
  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 213, test_head_fec_percentage, blocks, percentages, fallback), 0);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::head_shard_limit);
}

TEST(SplitFecBlocksHeadTailTests, RejectsTailBeyondBlockLimit) {
  const std::string payload(test_blocksize * 771, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};
  stream::fec_split_fallback_e fallback = stream::fec_split_fallback_e::none;

  // A 766-packet tail would need four bare blocks, one more than the protocol can describe.
  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 5, test_head_fec_percentage, blocks, percentages, fallback), 0);
  EXPECT_EQ(fallback, stream::fec_split_fallback_e::tail_block_limit);
}

TEST(RatecontrolPacketsPerMsTests, DisabledTest) {
  // 0 disables intra-frame pacing entirely.
  EXPECT_EQ(stream::ratecontrol_packets_per_ms(0, 1408), 0);
}

TEST(RatecontrolPacketsPerMsTests, DefaultPercentageTest) {
  // 80% of 1 Gbps at 1408-byte packets matches the historical hardcoded rate.
  EXPECT_EQ(stream::ratecontrol_packets_per_ms(80, 1408), 71);
}

TEST(RatecontrolPacketsPerMsTests, HigherPercentagesTest) {
  EXPECT_EQ(stream::ratecontrol_packets_per_ms(95, 1408), 84);
  EXPECT_EQ(stream::ratecontrol_packets_per_ms(100, 1408), 88);
}

TEST(PacingPercentForCodecTests, PyroWaveUsesItsOwnSetting) {
  // PyroWave is unpaced by default even though regular sessions keep the generic percentage.
  EXPECT_EQ(stream::pacing_percent_for_codec(true, 80, 0), 0);
  EXPECT_EQ(stream::pacing_percent_for_codec(true, 80, 40), 40);
}

TEST(PacingPercentForCodecTests, OtherCodecsUseTheGenericSetting) {
  EXPECT_EQ(stream::pacing_percent_for_codec(false, 80, 0), 80);
  EXPECT_EQ(stream::pacing_percent_for_codec(false, 0, 40), 0);
}

TEST(FecPercentForCodecTests, PyroWaveScalesWithTheConfiguredHeadParity) {
  // The ~20% head carries the configured parity; the rest of the frame is bare, so the effective
  // overhead is a fifth of the head percentage, regardless of the generic FEC setting.
  EXPECT_EQ(stream::fec_percent_for_codec(true, 20, 20), 4);
  EXPECT_EQ(stream::fec_percent_for_codec(true, 80, 20), 4);
  EXPECT_EQ(stream::fec_percent_for_codec(true, 20, 50), 10);
  EXPECT_EQ(stream::fec_percent_for_codec(true, 20, 0), 0);
  EXPECT_EQ(stream::fec_percent_for_codec(true, 20, 100), 20);
}

TEST(FecPercentForCodecTests, OtherCodecsUseTheGenericSetting) {
  EXPECT_EQ(stream::fec_percent_for_codec(false, 20, 20), 20);
  EXPECT_EQ(stream::fec_percent_for_codec(false, 5, 0), 5);
}

TEST(RatecontrolPacketsPerMsTests, NeverReturnsZeroWhileEnabledTest) {
  // Very low percentages with a large packet size round down, but callers divide by this value.
  EXPECT_EQ(stream::ratecontrol_packets_per_ms(1, 1408), 1);
}

TEST(RatecontrolPacketsPerMsTests, ZeroBlocksizeTest) {
  EXPECT_EQ(stream::ratecontrol_packets_per_ms(80, 0), 0);
}

TEST(PyrowaveCriticalPacketCountTests, ReportsZeroWhenUnknown) {
  EXPECT_EQ(stream::pyrowave_critical_packet_count(0, 1376, 10), 0);
  EXPECT_EQ(stream::pyrowave_critical_packet_count(8, 0, 10), 0);
  EXPECT_EQ(stream::pyrowave_critical_packet_count(8, 1376, 0), 0);
}

TEST(PyrowaveCriticalPacketCountTests, RoundsTheShardSpanUp) {
  // The short frame header rides in the first shard, so head_stream_bytes already includes it.
  EXPECT_EQ(stream::pyrowave_critical_packet_count(8, 1376, 10), 1);
  EXPECT_EQ(stream::pyrowave_critical_packet_count(1376, 1376, 10), 1);
  EXPECT_EQ(stream::pyrowave_critical_packet_count(1377, 1376, 10), 2);
  EXPECT_EQ(stream::pyrowave_critical_packet_count(8 + 1376 * 3, 1376, 10), 4);
}

TEST(PyrowaveCriticalPacketCountTests, ClampsToTheFrameAndHeaderField) {
  EXPECT_EQ(stream::pyrowave_critical_packet_count(100000, 1376, 2), 2);
  EXPECT_EQ(stream::pyrowave_critical_packet_count(1000000, 4, 1000000), std::numeric_limits<uint16_t>::max());
}

TEST(PyrowaveShardCountTests, RoundsTheFinalPartialShardUp) {
  EXPECT_EQ(stream::pyrowave_shard_count(0, 1408), 0);
  EXPECT_EQ(stream::pyrowave_shard_count(1, 1408), 1);
  EXPECT_EQ(stream::pyrowave_shard_count(1408, 1408), 1);
  EXPECT_EQ(stream::pyrowave_shard_count(1409, 1408), 2);
  EXPECT_EQ(stream::pyrowave_shard_count(2816, 1408), 2);
  EXPECT_EQ(stream::pyrowave_shard_count(2817, 1408), 3);
  EXPECT_EQ(stream::pyrowave_shard_count(2817, 0), 0);
}
