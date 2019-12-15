
#include <core/Utils.h>

namespace core {

    /* COMMON FUNCTIONS */

    template <typename T>
    base::Network<T> read_training(const std::string &network_name, uint32_t batch, uint32_t epoch,
            uint32_t traces_mode, bool QUIET) {

        // Read the network
        base::Network<T> network;
        interface::NetReader<T> reader = interface::NetReader<T>(network_name, batch, epoch, QUIET);
        network = reader.read_network_trace_params();

        bool forward = (traces_mode & 0x1u) != 0;
        bool backward = (traces_mode & 0x2u) != 0;
        bool accelerator_backward = (traces_mode & 0x4u) != 0;
        network.setForkward(forward);
        network.setBackward(backward);

        // Forward traces
        if(forward) {
            reader.read_training_weights_npy(network);
            reader.read_training_activations_npy(network);
        }

        // Backward traces
        if(backward) {
            reader.read_training_weight_gradients_npy(network);
            reader.read_training_input_gradients_npy(network);
            reader.read_training_output_activation_gradients_npy(network);
        }

        // Backward traces accelerators
        if (accelerator_backward) {
            reader.read_training_output_activation_gradients_npy(network);
        }

        return network;

    }

    bool iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
            int &x_counter, int &y_counter, int max_windows) {
        list_x.clear();
        list_y.clear();
        int current_windows = 0;
        while(y_counter < out_y) {
            while(x_counter < out_x) {
                list_x.push_back(x_counter);
                list_y.push_back(y_counter);
                current_windows++;
                x_counter++;
                if(current_windows >= max_windows)
                    return true;
            }
            x_counter = 0;
            y_counter++;
        }
        if(current_windows > 0)
            return true;

        y_counter = 0;
        return false;
    }

    std::tuple<uint8_t,uint8_t,uint8_t> split_bfloat16(float number) {
        bfloat16 bf_number = { .f = number };
        auto sign = (uint8_t)bf_number.field.sign;
        auto exponent = (uint8_t)bf_number.field.exponent;
        auto mantissa = (uint8_t)bf_number.field.mantissa;
        return std::make_tuple(sign,exponent,mantissa);
    }

    float cast_bfloat16(float number) {
        bfloat16 bf_number = { .f = number };
        bf_number.field.truncated_mantissa = 0;
        return bf_number.f;
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

    std::vector<uint8_t> generateEffectualBitsTableTMP(const int MAX_VALUES = 65535) {
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
        const static std::vector<uint8_t> effectual_bits_table = generateEffectualBitsTableTMP();
        return effectual_bits_table[value];
    }

    std::vector<std::tuple<uint8_t,uint8_t>> generateMinMaxTableTMP(const int MAX_VALUES = 32768) {
        std::vector<std::tuple<uint8_t,uint8_t>> min_max_table ((unsigned)MAX_VALUES, std::tuple<uint8_t,uint8_t>());
        min_max_table[0] = std::make_tuple(16,0);
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

            min_max_table[n] = std::make_tuple(min_act_bit, max_act_bit);
        }
        return min_max_table;
    }

    std::tuple<uint8_t,uint8_t> minMax(uint16_t value) {
        const static std::vector<std::tuple<uint8_t,uint8_t>> min_max_table = generateMinMaxTableTMP();
        return min_max_table[value];
    }

    bool check_act_bits(const std::vector<std::queue<uint8_t>> &offsets) {
        for (const auto &act_bits : offsets) {
            if (!act_bits.empty()) return true;
        }
        return false;
    }

    uint16_t sign_magnitude(short two_comp, uint16_t mask) {
        bool neg = two_comp < 0;
        int max_value = mask - 1;
        auto sign_mag = (uint16_t)abs(two_comp);
        sign_mag = (uint16_t)(sign_mag > max_value ? max_value : sign_mag);
        sign_mag = neg ? sign_mag | mask : sign_mag;
        return sign_mag;
    }

}
