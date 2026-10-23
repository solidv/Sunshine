/**
 * @file src/pyrowave_policy.cpp
 * @brief Pure PyroWave host policy implementation.
 */
#include "pyrowave_policy.h"

// local includes
#include "pyrowave_protocol.h"

// standard includes
#include <algorithm>
#include <limits>

namespace pyrowave::policy {
  namespace {
    using namespace pyrowave::protocol;

    /**
     * @brief Read a little-endian 32-bit word.
     *
     * @param p Source bytes; at least four bytes must be readable.
     * @return Decoded value.
     */
    std::uint32_t read_u32(const std::uint8_t *p) {
      return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    }

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
     * @brief Append a padding record of @p bytes bytes.
     *
     * @param out Destination buffer.
     * @param bytes Total record size; a multiple of four and at least PADDING_RECORD_MIN_BYTES.
     */
    void append_padding(std::vector<std::uint8_t> &out, std::size_t bytes) {
      append_u32(out, PADDING_MAGIC);
      append_u32(out, std::uint32_t(bytes / 4 - 2));
      out.resize(out.size() + (bytes - PADDING_RECORD_MIN_BYTES), 0);
    }

    /**
     * @brief One encoded PyroWave record inside a frame.
     */
    struct record_t {
      std::size_t offset;  ///< Byte offset in the encoded bitstream.
      std::size_t size;  ///< Record size in bytes.
      std::uint32_t block_index;  ///< PyroWave block index, or 0 for the sequence header.
    };

    /**
     * @brief Whether a record is too large to share a shard with a padding record.
     *
     * A record that leaves exactly four bytes of a shard would always strand that remainder, so it
     * counts as oversized too.
     *
     * @param record_bytes Record size in bytes.
     * @param shard_payload Shard payload in bytes.
     * @return True when the record cannot be followed by padding in one shard.
     */
    bool oversized(std::size_t record_bytes, std::size_t shard_payload) {
      return record_bytes + PADDING_RECORD_MIN_BYTES > shard_payload;
    }

    /**
     * @brief Bytes left in the shard holding frame byte @p position.
     *
     * @param position Frame byte offset.
     * @param shard_payload Shard payload in bytes.
     * @return Remaining bytes; never 0.
     */
    std::size_t shard_remaining(std::size_t position, std::size_t shard_payload) {
      return shard_payload - (position + FRAME_HEADER_BYTES) % shard_payload;
    }

    /**
     * @brief Record-size index used to find the earliest record that fits a shard remainder.
     *
     * Records are bucketed by size in words, each bucket holding its record indices in frame order.
     * A min segment tree over the buckets keeps each bucket's earliest remaining index, so "earliest
     * record of at most N words" is a prefix-minimum query. Records are only ever removed from the
     * front of their bucket.
     *
     * Only the range of sizes the frame uses is bucketed. `payload_words` is a 12-bit field, so the
     * absolute range is 0-4095, but a frame's records normally span a much narrower band and the
     * counting sort below runs once per bucket.
     */
    class fit_index_t {
    public:
      static constexpr std::size_t NONE = std::numeric_limits<std::size_t>::max();

      /**
       * @brief Build the index from record sizes in words.
       *
       * @param words One entry per record, in frame order.
       */
      explicit fit_index_t(const std::vector<std::uint32_t> &words) {
        if (words.empty()) {
          base_ = 0;
          leaves_ = 1;
        } else {
          const auto range = std::minmax_element(words.begin(), words.end());
          base_ = *range.first;
          const auto span = std::size_t(*range.second) - base_ + 1;
          leaves_ = 1;
          while (leaves_ < span) {
            leaves_ <<= 1;
          }
        }

        heads.assign(leaves_ + 1, 0);
        ends.assign(leaves_, 0);
        order.resize(words.size());
        tree.assign(2 * leaves_, NONE);

        // Counting sort of the record indices by size, stable in frame order.
        for (const auto w : words) {
          ++heads[index(w) + 1];
        }
        for (std::size_t b = 0; b < leaves_; ++b) {
          heads[b + 1] += heads[b];
          ends[b] = heads[b + 1];
        }
        std::vector<std::size_t> fill(heads.begin(), heads.end() - 1);
        for (std::size_t i = 0; i < words.size(); ++i) {
          order[fill[index(words[i])]++] = i;
        }
        for (std::size_t b = 0; b < leaves_; ++b) {
          tree[leaves_ + b] = heads[b] < ends[b] ? order[heads[b]] : NONE;
        }
        for (std::size_t n = leaves_ - 1; n > 0; --n) {
          tree[n] = std::min(tree[2 * n], tree[2 * n + 1]);
        }
      }

      /**
       * @brief Earliest remaining record of at most @p max_words words.
       *
       * @param max_words Record-size limit in words.
       * @return Record index, or #NONE.
       */
      [[nodiscard]] std::size_t earliest_fitting(std::size_t max_words) const {
        if (max_words < base_) {
          return NONE;
        }
        std::size_t best = NONE;
        std::size_t lo = leaves_;
        std::size_t hi = leaves_ + std::min(max_words - base_, leaves_ - 1) + 1;
        while (lo < hi) {
          if (lo & 1) {
            best = std::min(best, tree[lo++]);
          }
          if (hi & 1) {
            best = std::min(best, tree[--hi]);
          }
          lo >>= 1;
          hi >>= 1;
        }
        return best;
      }

      /**
       * @brief Earliest remaining record of exactly @p words words.
       *
       * @param words Record size in words.
       * @return Record index, or #NONE.
       */
      [[nodiscard]] std::size_t earliest_exact(std::uint32_t words) const {
        if (words < base_) {
          return NONE;
        }
        const auto bucket = std::size_t(words) - base_;
        return bucket < leaves_ ? tree[leaves_ + bucket] : NONE;
      }

      /**
       * @brief Remove the earliest remaining record of @p words words.
       *
       * @param words Record size in words.
       */
      void remove(std::uint32_t words) {
        const auto bucket = index(words);
        const auto head = ++heads[bucket];
        std::size_t n = leaves_ + bucket;
        tree[n] = head < ends[bucket] ? order[head] : NONE;
        for (n >>= 1; n > 0; n >>= 1) {
          tree[n] = std::min(tree[2 * n], tree[2 * n + 1]);
        }
      }

    private:
      /**
       * @brief Bucket index of a record size in words.
       *
       * @param words Record size in words, at least #base_.
       * @return Index into the bucket arrays.
       */
      [[nodiscard]] std::size_t index(std::uint32_t words) const {
        return std::size_t(words) - base_;
      }

      std::uint32_t base_ = 0;  ///< Size in words of bucket 0.
      std::size_t leaves_ = 1;  ///< Power-of-two bucket count covering the frame's size range.

      std::vector<std::size_t> heads;  ///< Per bucket: position of the earliest remaining record in `order`.
      std::vector<std::size_t> ends;  ///< Per bucket: end of its run in `order`.
      std::vector<std::size_t> order;  ///< Record indices grouped by size, in frame order.
      std::vector<std::size_t> tree;  ///< Min segment tree over the earliest remaining indices.
    };
  }  // namespace

  framing_e select_framing(bool client_sent_adaptive_fec, std::optional<std::uint32_t> client_features) {
    if (client_sent_adaptive_fec) {
      return framing_e::records;
    }
    if (client_features && (*client_features & FEATURE_RECORD_FRAMING)) {
      return framing_e::records;
    }
    return framing_e::length_prefixed;
  }

  std::size_t shard_payload_bytes(int packetsize) {
    if (packetsize <= SHARD_OVERHEAD_BYTES) {
      return 0;
    }
    const auto shard = std::size_t(packetsize - SHARD_OVERHEAD_BYTES);
    if (shard < FRAME_HEADER_BYTES + SEQUENCE_HEADER_BYTES + PADDING_RECORD_MIN_BYTES || shard % 4 != 0) {
      return 0;
    }
    return shard;
  }

  std::uint32_t coarse_block_count(std::uint32_t width, std::uint32_t height) {
    // PyroWave pads each dimension to 32 pixels (at least 128), then five wavelet levels halve
    // it; the coarsest level holds four bands of three components.
    const auto blocks = [](std::uint32_t extent) {
      const std::uint32_t aligned = std::max<std::uint32_t>((extent + 31) / 32 * 32, 128);
      return ((aligned >> 5) + 31) / 32;
    };
    return blocks(width) * blocks(height) * 4 * 3;
  }

  std::size_t max_frame_bytes(int packetsize, bool critical_fec) {
    if (packetsize <= SHARD_OVERHEAD_BYTES) {
      return 0;
    }
    const auto max_shards = critical_fec ? 3000u : 4000u;
    return std::min<std::size_t>(max_shards, MAX_FEC_BLOCKS * MAX_DATA_SHARDS_PER_BLOCK) *
             std::size_t(packetsize - SHARD_OVERHEAD_BYTES) -
           FRAME_HEADER_BYTES;
  }

  std::size_t max_bitstream_bytes(int packetsize, bool length_prefixed, bool critical_fec) {
    auto capacity = max_frame_bytes(packetsize, critical_fec);
    if (length_prefixed) {
      // At worst each block and the sequence header occupy their own packet. Each packet needs a
      // length word, plus the frame's packet-count word.
      if (capacity < sizeof(std::uint32_t)) {
        return 0;
      }
      capacity = (capacity - sizeof(std::uint32_t)) / (BLOCK_HEADER_BYTES + sizeof(std::uint32_t)) * BLOCK_HEADER_BYTES;
    }
    return capacity & ~std::size_t(3);
  }

  std::size_t wire_packet_overhead_bytes(bool ipv6, bool video_encryption) {
    // RTP header allowance beyond the negotiated packet size, Ethernet preamble/SFD/header/FCS/IFG,
    // the UDP header, and the IP header.
    constexpr std::size_t RTP_OVERHEAD = 16;
    constexpr std::size_t ETHERNET_OVERHEAD = 38;
    constexpr std::size_t UDP_OVERHEAD = 8;
    constexpr std::size_t IPV4_OVERHEAD = 20;
    constexpr std::size_t IPV6_OVERHEAD = 40;
    constexpr std::size_t ENCRYPTION_OVERHEAD = 32;

    return RTP_OVERHEAD + ETHERNET_OVERHEAD + UDP_OVERHEAD + (ipv6 ? IPV6_OVERHEAD : IPV4_OVERHEAD) + (video_encryption ? ENCRYPTION_OVERHEAD : 0);
  }

  std::size_t max_critical_parity_shards(const wire_budget_t &transport, std::size_t packets) {
    if (transport.critical_fec_percentage <= 0 || packets == 0) {
      return 0;
    }

    // A Reed-Solomon block can describe at most 255 shards in total.
    constexpr std::size_t MAX_SHARDS = 255;
    const auto percentage = std::size_t(transport.critical_fec_percentage);
    std::size_t parity = 0;
    for (std::size_t critical = 1; critical <= std::min(packets, MAX_SHARDS); ++critical) {
      const auto candidate = std::max((critical * percentage + 99) / 100, transport.min_parity_shards);
      if (critical + candidate <= MAX_SHARDS) {
        parity = std::max(parity, candidate);
      }
    }
    return parity;
  }

  std::int64_t video_wire_budget_kbps(std::int64_t configured_kbps, int audio_channels, bool high_quality) {
    if (configured_kbps <= 0) {
      return 0;
    }

    constexpr std::int64_t CONTROL_KBPS = 500;
    const auto audio_kbps = std::int64_t(high_quality ? 256 : 96) * std::max(audio_channels, 0);
    // Audio parity is four data shards plus two parity shards.
    const auto audio_wire_kbps = (audio_kbps * 3 + 1) / 2;
    return std::max<std::int64_t>(configured_kbps - audio_wire_kbps - CONTROL_KBPS, 0);
  }

  std::size_t frame_payload_bytes(std::size_t wire_bytes, const wire_budget_t &transport) {
    if (transport.packetsize <= SHARD_OVERHEAD_BYTES || wire_bytes == 0) {
      return 0;
    }

    const auto wire_packet_bytes = std::size_t(transport.packetsize) + transport.packet_overhead_bytes;
    const auto packets = wire_bytes / wire_packet_bytes;
    const auto parity = max_critical_parity_shards(transport, packets);
    if (packets <= parity) {
      return 0;
    }

    const auto data = (packets - parity) * std::size_t(transport.packetsize - SHARD_OVERHEAD_BYTES);
    return data > FRAME_HEADER_BYTES ? (data - FRAME_HEADER_BYTES) & ~std::size_t(3) : 0;
  }

  std::size_t wire_bytes_for_frame(std::size_t frame_bytes, const wire_budget_t &transport) {
    if (transport.packetsize <= SHARD_OVERHEAD_BYTES || frame_bytes == 0) {
      return 0;
    }

    const auto payload = std::size_t(transport.packetsize - SHARD_OVERHEAD_BYTES);
    const auto packets = (frame_bytes + FRAME_HEADER_BYTES + payload - 1) / payload;
    const auto parity = max_critical_parity_shards(transport, packets);
    return (packets + parity) * (std::size_t(transport.packetsize) + transport.packet_overhead_bytes);
  }

  std::optional<record_frame_stats_t> write_record_frame(
    std::span<const std::uint8_t> bitstream,
    std::size_t shard_payload,
    std::vector<std::uint8_t> &out,
    std::size_t frame_limit
  ) {
    // Split the frame into its sequence header and block records.
    if (bitstream.size() < SEQUENCE_HEADER_BYTES || (read_u32(bitstream.data()) >> 31) == 0 || (frame_limit && bitstream.size() > frame_limit)) {
      return std::nullopt;
    }

    std::vector<record_t> records;
    for (std::size_t position = SEQUENCE_HEADER_BYTES; position < bitstream.size();) {
      if (bitstream.size() - position < BLOCK_HEADER_BYTES) {
        return std::nullopt;
      }
      const std::uint32_t word0 = read_u32(bitstream.data() + position);
      const std::uint32_t payload_words = (word0 >> 16) & 0xfff;
      if ((word0 >> 31) != 0 || payload_words < 2 || std::size_t(payload_words) * 4 > bitstream.size() - position) {
        return std::nullopt;
      }
      records.push_back({position, std::size_t(payload_words) * 4, read_u32(bitstream.data() + position + 4) >> 8});
      position += records.back().size;
    }

    record_frame_stats_t stats;
    stats.block_records = records.size();
    const std::size_t frame_start = out.size();
    out.reserve(frame_start + bitstream.size());
    out.insert(out.end(), bitstream.begin(), bitstream.begin() + SEQUENCE_HEADER_BYTES);

    if (!shard_payload) {
      out.insert(out.end(), bitstream.begin() + SEQUENCE_HEADER_BYTES, bitstream.end());
      return stats;
    }

    const std::uint32_t sequence_word = read_u32(bitstream.data());
    const std::uint32_t coarse_blocks = coarse_block_count((sequence_word & 0x3fff) + 1, ((sequence_word >> 14) & 0x3fff) + 1);

    // Frame byte p sits at offset p + FRAME_HEADER_BYTES of the shard stream, because the short
    // frame header precedes the frame in the first shard.
    const auto remaining = [&]() {
      return shard_remaining(out.size() - frame_start, shard_payload);
    };
    const auto place = [&](const record_t &record) {
      out.insert(out.end(), bitstream.begin() + record.offset, bitstream.begin() + record.offset + record.size);
    };
    const auto pad = [&](std::size_t bytes) {
      append_padding(out, bytes);
      ++stats.padding_records;
      stats.padding_bytes += bytes;
    };

    // The coarsest wavelet level goes first: a receiver cannot decode a frame that lost any of it,
    // so those shards (`critical_bytes`) are the ones worth parity. Then everything else. Within
    // each group, oversized records come first and span shards wherever they start; each is shifted
    // by a minimal padding record when it would end 4 bytes before a boundary, since nothing could
    // fill that remainder. The group's other records follow, packed first-fit without crossing a
    // shard boundary, so every shard after the finer level's oversized records starts with a record.
    std::vector<std::size_t> coarse;
    std::vector<std::size_t> fine;
    for (std::size_t i = 0; i < records.size(); ++i) {
      (records[i].block_index < coarse_blocks ? coarse : fine).push_back(i);
    }

    for (const auto *group : {&coarse, &fine}) {
      std::vector<std::size_t> ordinary;
      for (const auto i : *group) {
        if (oversized(records[i].size, shard_payload)) {
          if (records[i].size % shard_payload == remaining() - 4) {
            pad(PADDING_RECORD_MIN_BYTES);
          }
          place(records[i]);
          ++stats.oversized_records;
        } else {
          ordinary.push_back(i);
        }
      }

      std::vector<std::uint32_t> words;
      words.reserve(ordinary.size());
      for (const auto i : ordinary) {
        words.push_back(std::uint32_t(records[i].size / 4));
      }

      fit_index_t fit(words);
      std::vector<bool> placed(ordinary.size(), false);
      std::size_t earliest = 0;
      for (std::size_t left = ordinary.size(); left != 0;) {
        while (placed[earliest]) {
          ++earliest;
        }
        const std::size_t room = remaining();

        std::size_t chosen = fit.earliest_fitting(room / 4);
        if (chosen != fit_index_t::NONE && room - std::size_t(words[chosen]) * 4 == 4) {
          // That would strand 4 bytes, too few for a padding record. Take a record that fills the
          // shard exactly or leaves room for a padding record.
          const std::size_t exact = fit.earliest_exact(std::uint32_t(room / 4));
          const std::size_t roomy = room >= 16 ? fit.earliest_fitting((room - 8) / 4) : fit_index_t::NONE;
          chosen = std::min(exact, roomy);
        }
        if (chosen == fit_index_t::NONE) {
          // Never 4: a shard starts with at least 16 frame bytes and no placement leaves 4.
          pad(room);
          continue;
        }

        if (chosen != earliest) {
          ++stats.reordered_records;
        }
        placed[chosen] = true;
        fit.remove(words[chosen]);
        --left;
        place(records[ordinary[chosen]]);
      }

      if (group == &coarse) {
        stats.critical_bytes = out.size() - frame_start;
      }
    }

    if (frame_limit && out.size() - frame_start > frame_limit) {
      // Padding and reordering did not fit the transport budget. Keep the importance order, which
      // costs no extra bytes, and let the client parse the frame sequentially.
      out.resize(frame_start);
      out.insert(out.end(), bitstream.begin(), bitstream.begin() + SEQUENCE_HEADER_BYTES);
      stats = {};
      stats.block_records = records.size();
      for (const auto *group : {&coarse, &fine}) {
        for (const auto i : *group) {
          place(records[i]);
        }
        if (group == &coarse) {
          stats.critical_bytes = out.size() - frame_start;
        }
      }
    }
    return stats;
  }

  record_frame_info_t inspect_record_frame(std::span<const std::uint8_t> frame, std::size_t shard_payload) {
    record_frame_info_t info;
    std::size_t position = 0;
    std::uint32_t last_block_index = 0;

    const auto fail = [&](std::string error) {
      info.valid = false;
      info.error = std::move(error);
      return info;
    };

    std::uint32_t coarse_blocks = 0;
    bool seen_ordinary_coarse = false;
    bool seen_ordinary_fine = false;
    bool seen_fine = false;

    // Returns whether the record is oversized; ordinary records and padding must stay inside one
    // shard so that every shard after the oversized records starts with a record.
    const auto note_alignment = [&](std::size_t start, std::size_t size) {
      if (!shard_payload) {
        return false;
      }
      const std::size_t wire_start = start + FRAME_HEADER_BYTES;
      const std::size_t first_shard = wire_start / shard_payload;
      const std::size_t last_shard = (wire_start + size - 1) / shard_payload;
      if (oversized(size, shard_payload)) {
        ++info.oversized_records;
        return true;
      }
      if (first_shard != last_shard) {
        ++info.misaligned_records;
      }
      return false;
    };

    info.stripped.reserve(frame.size());
    while (position < frame.size()) {
      if (frame.size() - position < BLOCK_HEADER_BYTES) {
        return fail("truncated record header at byte " + std::to_string(position));
      }
      const auto *record = frame.data() + position;
      const std::uint32_t word0 = read_u32(record);
      const std::uint32_t word1 = read_u32(record + sizeof(std::uint32_t));

      if (word0 == PADDING_MAGIC) {
        const std::size_t bytes = PADDING_RECORD_MIN_BYTES + std::size_t(word1) * 4;
        if (bytes > frame.size() - position) {
          return fail("padding record runs past the end of the frame at byte " + std::to_string(position));
        }
        ++info.padding_records;
        info.padding_bytes += bytes;
        note_alignment(position, bytes);
        position += bytes;
        continue;
      }

      const bool extended = (word0 >> 31) != 0;
      if (extended) {
        if (info.has_sequence_header) {
          return fail("second sequence header at byte " + std::to_string(position));
        }
        if (info.block_records != 0) {
          return fail("sequence header after block records");
        }
        info.has_sequence_header = true;
        info.width = (word0 & 0x3fff) + 1;
        info.height = ((word0 >> 14) & 0x3fff) + 1;
        info.total_blocks = word1 & 0xffffff;
        const std::uint32_t code = (word1 >> 24) & 0x3;
        if (code != 0) {
          return fail("sequence header code " + std::to_string(code) + " is not START_OF_FRAME");
        }
        info.chroma444 = ((word1 >> 26) & 0x1) != 0;
        coarse_blocks = coarse_block_count(info.width, info.height);
        info.critical_bytes = position + SEQUENCE_HEADER_BYTES;
        note_alignment(position, SEQUENCE_HEADER_BYTES);
        info.stripped.insert(info.stripped.end(), record, record + SEQUENCE_HEADER_BYTES);
        position += SEQUENCE_HEADER_BYTES;
        continue;
      }

      if (!info.has_sequence_header) {
        return fail("block record before the sequence header");
      }
      const std::uint32_t payload_words = (word0 >> 16) & 0xfff;
      if (payload_words < 2) {
        return fail("block record with payload_words " + std::to_string(payload_words) + " at byte " + std::to_string(position));
      }
      const std::size_t bytes = std::size_t(payload_words) * 4;
      if (bytes > frame.size() - position) {
        return fail("block record runs past the end of the frame at byte " + std::to_string(position));
      }
      const std::uint32_t block_index = word1 >> 8;
      if (info.block_records != 0 && block_index < last_block_index) {
        ++info.out_of_order_records;
      }
      last_block_index = block_index;
      ++info.block_records;
      if (shard_payload) {
        const bool coarse = block_index < coarse_blocks;
        if (coarse && seen_fine) {
          ++info.late_coarse_records;
        }
        if (note_alignment(position, bytes)) {
          // Each level's oversized records precede its ordinary ones.
          if (coarse ? seen_ordinary_coarse : seen_ordinary_fine) {
            ++info.late_oversized_records;
          }
        } else if (coarse) {
          seen_ordinary_coarse = true;
        } else {
          seen_ordinary_fine = true;
        }
        if (!coarse) {
          seen_fine = true;
        } else {
          info.critical_bytes = position + bytes;
        }
      }
      info.stripped.insert(info.stripped.end(), record, record + bytes);
      position += bytes;
    }

    if (!info.has_sequence_header) {
      return fail("frame has no sequence header");
    }
    info.valid = true;
    return info;
  }

  std::vector<bool> record_start_shards(std::span<const std::uint8_t> frame, std::size_t shard_payload) {
    if (!shard_payload || frame.empty()) {
      return {};
    }
    std::vector<bool> starts((frame.size() + FRAME_HEADER_BYTES + shard_payload - 1) / shard_payload, false);
    for (std::size_t position = 0; position < frame.size();) {
      if (frame.size() - position < BLOCK_HEADER_BYTES) {
        return {};
      }
      if ((position + FRAME_HEADER_BYTES) % shard_payload == 0 || position == 0) {
        starts[(position + FRAME_HEADER_BYTES) / shard_payload] = true;
      }
      const std::uint32_t word0 = read_u32(frame.data() + position);
      std::size_t bytes;
      if (word0 == PADDING_MAGIC) {
        bytes = PADDING_RECORD_MIN_BYTES + std::size_t(read_u32(frame.data() + position + sizeof(std::uint32_t))) * 4;
      } else if ((word0 >> 31) != 0) {
        bytes = SEQUENCE_HEADER_BYTES;
      } else {
        bytes = std::size_t((word0 >> 16) & 0xfff) * 4;
        if (bytes < BLOCK_HEADER_BYTES) {
          return {};
        }
      }
      if (bytes > frame.size() - position) {
        return {};
      }
      position += bytes;
    }
    return starts;
  }
}  // namespace pyrowave::policy
