
#include <core/Utils.h>

namespace core {

    /* COMMON FUNCTIONS */

    std::string to_mem_string(uint64_t mem) {
        if (mem >= 1e9 && mem % (uint64_t)1e9 == 0)
            return std::to_string(mem / (uint64_t)1e9) + "GB";
        else if (mem >= pow(2, 30) && mem % (uint64_t)pow(2, 30) == 0)
            return std::to_string(mem / (uint64_t)pow(2, 30)) + "GiB";
        else if (mem >= 1e6 && mem % (uint64_t)1e6 == 0)
            return std::to_string(mem / (uint64_t)1e6) + "MB";
        else if (mem >= pow(2, 20) && mem % (uint64_t)pow(2, 20) == 0)
            return std::to_string(mem / (uint64_t)pow(2, 20)) + "MiB";
        else if (mem >= 1e3 && mem % (uint64_t)1e3 == 0)
            return std::to_string(mem / (uint64_t)1e3) + "KB";
        else if (mem >= pow(2, 10) && mem % (uint64_t)pow(2, 10) == 0)
            return std::to_string(mem / (uint64_t)pow(2, 10)) + "KiB";
        else
            return std::to_string(mem) + "B";
    }

    /* Only encode the values when get less number of bits */
    uint16_t generateBoothEncodingEntry(uint16_t n) {
        uint32_t padded_n = n << 2;
        std::string bitstream = std::bitset<16 + 2>(padded_n).to_string();
        uint16_t booth_encoding = 0;
        bool booth = false;
        for(int i = 0; i < 16; i++) {
            std::string w = bitstream.substr(0,3);
            booth_encoding <<= 1;
            if(w == "000" || w == "001") {
                assert(!booth);
            } else if(w == "010") {
                if (booth) booth_encoding |= 0x1;
            } else if(w == "011") {
                if (booth) booth_encoding |= 0x1;
            } else if(w == "100") {
                if (!booth) booth_encoding |= 0x1;
                else { booth_encoding |= 0x1; booth = false;}
            } else if(w == "101") {
                if (!booth) booth_encoding |= 0x1;
            } else if(w == "110") {
                if (!booth) booth_encoding |= 0x1;
            } else if(w == "111") {
                if (!booth) { booth_encoding |= 0x2;  booth = true; }
            }
            bitstream = bitstream.substr(1);
        }
        return booth_encoding;
    }

    std::vector<uint16_t> generateBoothEncodingTable(const int MAX_VALUES = 32768) {
        std::vector<uint16_t> booth_table ((unsigned)MAX_VALUES, 0);
        for(int n = 0; n < MAX_VALUES; n++)
            booth_table[n] = generateBoothEncodingEntry((uint16_t)n);
        return booth_table;
    }

    uint16_t booth_encoding(uint16_t value) {
        const static std::vector<uint16_t> booth_table = generateBoothEncodingTable();
        return booth_table[value];
    }

    std::vector<uint8_t> generateEffectualBitsTable(const int MAX_VALUES = 65535) {
        std::vector<uint8_t> effectual_bits_table ((unsigned)MAX_VALUES, 0);
        for(int n = 0; n < MAX_VALUES; n++) {

            auto tmp_n = n;
            uint8_t effectual_bits = 0;
            while (tmp_n) {
                effectual_bits += tmp_n & 1;
                tmp_n >>= 1;
            }

            effectual_bits_table[n] = effectual_bits;
        }
        return effectual_bits_table;
    }

    uint8_t effectualBits(uint16_t value) {
        const static std::vector<uint8_t> effectual_bits_table = generateEffectualBitsTable();
        return effectual_bits_table[value];
    }

    std::vector<std::tuple<uint8_t,uint8_t>> generateMinMaxTable(const int MAX_VALUES = 32768) {
        std::vector<std::tuple<uint8_t,uint8_t>> min_max_table ((unsigned)MAX_VALUES, std::tuple<uint8_t,uint8_t>());
        min_max_table[0] = {16,0};
        for(int n = 1; n < MAX_VALUES; n++) {

            auto tmp_n = n;
            uint8_t count = 0;
            std::vector<uint8_t> offsets;
            while (tmp_n) {
                auto current_bit = tmp_n & 1;
                if(current_bit) offsets.push_back(count);
                tmp_n >>= 1;
                count++;
            }

            auto min_act_bit = offsets[0];
            auto max_act_bit = offsets[offsets.size()-1];

            min_max_table[n] = {min_act_bit, max_act_bit};
        }
        return min_max_table;
    }

    std::tuple<uint8_t,uint8_t> minMax(uint16_t value) {
        const static std::vector<std::tuple<uint8_t,uint8_t>> min_max_table = generateMinMaxTable();
        return min_max_table[value];
    }

}
