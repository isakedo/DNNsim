
#include <core/Simulator.h>

namespace core {

    /* COMMON FUNCTIONS */

    template <typename T>
    base::Network<T> Simulator<T>::read_training(const std::string &network_name, uint32_t batch, uint32_t epoch,
            uint32_t traces_mode) {

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

    template <typename T>
    bool Simulator<T>::iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
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

    template <typename T>
    std::tuple<uint8_t,uint8_t,uint8_t> Simulator<T>::split_bfloat16(float number) {
        bfloat16 bf_number = { .f = number };
        auto sign = (uint8_t)bf_number.field.sign;
        auto exponent = (uint8_t)bf_number.field.exponent;
        auto mantissa = (uint8_t)bf_number.field.mantissa;
        return std::make_tuple(sign,exponent,mantissa);
    }

    template <typename T>
    float Simulator<T>::cast_bfloat16(float number) {
        bfloat16 bf_number = { .f = number };
        bf_number.field.truncated_mantissa = 0;
        return bf_number.f;
    }

    /* Only encode the values when get less number of bits */
    uint16_t generateBoothEncoding(uint16_t n) {
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

    std::vector<uint16_t> generateBoothTable(const int MAX_VALUES = 32768) {
        std::vector<uint16_t> booth_table ((unsigned)MAX_VALUES, 0);
        for(int n = 0; n < MAX_VALUES; n++)
            booth_table[n] = generateBoothEncoding((uint16_t)n);
        return booth_table;
    }

    template <typename T>
    uint16_t Simulator<T>::booth_encoding(uint16_t value) {
        const static std::vector<uint16_t> booth_table = generateBoothTable();
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

    template <typename T>
    uint8_t Simulator<T>::effectualBits(uint16_t value) {
        const static std::vector<uint8_t> effectual_bits_table = generateEffectualBitsTable();
        return effectual_bits_table[value];
    }

    std::vector<std::tuple<uint8_t,uint8_t>> generateMinMaxTable(const int MAX_VALUES = 32768) {
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

            min_max_table[n] = std::make_tuple(min_act_bit,max_act_bit);
        }
        return min_max_table;
    }

    template <typename T>
    std::tuple<uint8_t,uint8_t> Simulator<T>::minMax(uint16_t value) {
        const static std::vector<std::tuple<uint8_t,uint8_t>> min_max_table = generateMinMaxTable();
        return min_max_table[value];
    }

    template <typename T>
    bool Simulator<T>::check_act_bits(const std::vector<std::queue<uint8_t>> &offsets) {
        for (const auto &act_bits : offsets) {
            if (!act_bits.empty()) return true;
        }
        return false;
    }

    template <typename T>
    uint16_t Simulator<T>::sign_magnitude(short two_comp, uint16_t mask) {
        bool neg = two_comp < 0;
        int max_value = mask - 1;
        auto sign_mag = (uint16_t)abs(two_comp);
        sign_mag = (uint16_t)(sign_mag > max_value ? max_value : sign_mag);
        sign_mag = neg ? sign_mag | mask : sign_mag;
        return sign_mag;
    }

    /* POTENTIALS */

    template <typename T>
    void Simulator<T>::potentials(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch) {

        // Initialize statistics
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(),
                arch->filename_pot());

        auto work_reduction = stats.register_double_t("work_reduction", 0, sys::Average);
        auto speedup = stats.register_double_t("speedup", 0, sys::Average);
        auto par_mult = stats.register_double_t("parallel_multiplication", 0, sys::AverageTotal);
        auto bit_multiplications = stats.register_uint_t("bit_multiplications", 0, sys::AverageTotal);
        auto act_precision = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_precision = stats.register_uint_t("weights_precision", 0, sys::Average);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            arch->dataConversion(act, layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();

            base::Array<T> wgt = layer.getWeights();
            arch->dataConversion(wgt, layer.getWgtPrecision());
            if(conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            uint64_t batch_size, act_channels, Nx, Ny, R;
            if (lstm) {
                R = act_shape[0];
                batch_size = act_shape[1];
                act_channels = act_shape[2];
                Nx = 1;
                Ny = 1;
            } else {
                R = 1;
                batch_size = act_shape[0];
                act_channels = act_shape[1];
                Nx = act_shape[2];
                Ny = act_shape[3];
            }

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            auto groups = act_channels / wgt_channels;
            auto it_per_group = num_filters / groups;

            auto network_bits = network.getNetwork_bits();
            auto act_prec = layer.getActPrecision();
            auto wgt_prec = layer.getWgtPrecision();

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                                     num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                if (conv) {

                    for(int m = 0; m < num_filters; m++) {

                        // Two towers alexnet
                        int start_group = 0;
                        if(m >= it_per_group)
                            start_group = (int)wgt_channels;

                        // Fix for MobileNet
                        if(wgt_channels == 1 && act_channels != 1)
                            start_group = m;

                        for(int x = 0; x < out_x; x++) {
                            for(int y = 0; y < out_y; y++) {
                                for (int i = 0; i < Kx; i++) {
                                    for (int j = 0; j < Ky; j++) {
                                        for (int k = 0; k < wgt_channels; k++) {
                                            T act_bits = act.get(n, start_group + k, stride * x + i, stride * y + j);
                                            T wgt_bits = wgt.get(m, k, i, j);
                                            bit_counter += arch->computeBits(act_bits, wgt_bits, act_prec, wgt_prec,
                                                    network_bits);
                                        }
                                    }
                                }
                            }
                        }
                    }

                } else {

                    for (int r = 0; r < R; r++) {
                        for (int m = 0; m < num_filters; m++) {
                            for (int k = 0; k < wgt_channels; k++) {
                                T act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                                T wgt_bits = wgt.get(m, k);
                                bit_counter += arch->computeBits(act_bits, wgt_bits, act_prec, wgt_prec, network_bits);
                            }
                        }
                    }

                }

                bit_multiplications->value[layer_it][n] = bit_counter;
                work_reduction->value[layer_it][n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
                speedup->value[layer_it][n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
                par_mult->value[layer_it][n] = parallel_mult;
                act_precision->value[layer_it][n] = layer.getActPrecision();
                wgt_precision->value[layer_it][n] = layer.getWgtPrecision();
            }

        }

        //Dump statistics
        stats.dump_csv(network.getName(), network.getLayersName(), arch->header_pot(network.getName()), this->QUIET);

    }


    INITIALISE_DATA_TYPES(Simulator);

}
