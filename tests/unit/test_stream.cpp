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
  int split_fec_blocks_head_tail(const std::string_view &payload, size_t blocksize, size_t payload_blocksize, size_t head_stream_bytes, std::array<std::string_view, 4> &blocks, std::array<int, 4> &percentages);
}  // namespace stream

TEST(ConcatAndInsertTests, ConcatNoInsertionTest) {
  char b1[] = {'a', 'b'};
  char b2[] = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(0, 2, std::string_view {b1, sizeof(b1)}, std::string_view {b2, sizeof(b2)});
  auto expected = std::vector<uint8_t> {'a', 'b', 'c', 'd', 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, ConcatLargeStrideTest) {
  char b1[] = {'a', 'b'};
  char b2[] = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(1, sizeof(b1) + sizeof(b2) + 1, std::string_view {b1, sizeof(b1)}, std::string_view {b2, sizeof(b2)});
  auto expected = std::vector<uint8_t> {0, 'a', 'b', 'c', 'd', 'e'};
  ASSERT_EQ(res, expected);
}

TEST(ConcatAndInsertTests, ConcatSmallStrideTest) {
  char b1[] = {'a', 'b'};
  char b2[] = {'c', 'd', 'e'};
  auto res = stream::concat_and_insert(1, 1, std::string_view {b1, sizeof(b1)}, std::string_view {b2, sizeof(b2)});
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

  // 1000 protected stream bytes cover ceil(1000 / 56) = 18 packets, so the 400-packet frame
  // leaves a 382-packet tail that needs two bare blocks (255 shards maximum each).
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, 1000, blocks, percentages);

  ASSERT_EQ(count, 3);
  EXPECT_EQ(percentages[0], 50);
  EXPECT_EQ(percentages[1], 0);
  EXPECT_EQ(percentages[2], 0);
  EXPECT_EQ(blocks[0].size(), 18 * test_blocksize);
  expect_blocks_tile_payload(payload, blocks, count);
}

TEST(SplitFecBlocksHeadTailTests, UsesAllFourBlocksForALongTail) {
  const std::string payload(test_blocksize * 700, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};

  // A 695-packet tail fills three bare blocks, which is the most the protocol allows.
  const auto count = stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 5, blocks, percentages);

  ASSERT_EQ(count, 4);
  EXPECT_EQ(percentages[0], 50);
  EXPECT_EQ(percentages[3], 0);
  expect_blocks_tile_payload(payload, blocks, count);
}

TEST(SplitFecBlocksHeadTailTests, RejectsEmptyHead) {
  const std::string payload(test_blocksize * 400, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};

  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, 0, blocks, percentages), 0);
}

TEST(SplitFecBlocksHeadTailTests, RejectsFrameWithoutTail) {
  const std::string payload(test_blocksize * 10, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};

  // The head covers the whole frame, so there is nothing to leave bare.
  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, payload.size(), blocks, percentages), 0);
}

TEST(SplitFecBlocksHeadTailTests, RejectsProtectedHeadBeyondShardLimit) {
  const std::string payload(test_blocksize * 400, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};

  // 171 packets need more than 50% parity to stay within the 255-shard Reed-Solomon limit.
  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 171, blocks, percentages), 0);
}

TEST(SplitFecBlocksHeadTailTests, RejectsTailBeyondShardLimit) {
  const std::string payload(test_blocksize * 900, 'x');
  std::array<std::string_view, 4> blocks {};
  std::array<int, 4> percentages {};

  // An 895-packet tail would need five FEC blocks, more than the protocol can describe.
  EXPECT_EQ(stream::split_fec_blocks_head_tail(payload, test_blocksize, test_payload_blocksize, test_payload_blocksize * 5, blocks, percentages), 0);
}
