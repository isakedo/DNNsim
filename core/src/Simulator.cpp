
#include <core/Simulator.h>

namespace core {

    /* AUXILIAR FUNCTIONS */

    template <typename T>
    base::Network<T> read_training(const std::string &network_name, uint32_t batch, uint32_t epoch,
            uint32_t decoder_states, uint32_t traces_mode, bool QUIET) {

        // Read the network
        base::Network<T> network;
        interface::NetReader<T> reader = interface::NetReader<T>(network_name, batch, epoch, QUIET);
        network = reader.read_network_trace_params();
        if(decoder_states > 0) network.duplicate_decoder_layers(decoder_states);

        bool forward = (traces_mode & 0x1u) != 0;
        bool backward = (traces_mode & 0x2u) != 0;
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
        return network;

    }

    /* COMMON FUNCTIONS */

    template <typename T>
    bool Simulator<T>::iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
            int &x_counter, int &y_counter, int max_windows) {
        list_x.clear();
        list_y.clear();
        int current_windows = 0;
        while(x_counter < out_x) {
            while(y_counter < out_y) {
                list_x.push_back(x_counter);
                list_y.push_back(y_counter);
                current_windows++;
                y_counter++;
                if(current_windows >= max_windows)
                    return true;
            }
            y_counter = 0;
            x_counter++;
        }
        if(current_windows > 0)
            return true;

        x_counter = 0;
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

    /* DATA CALCULATIONS */

    template <typename T>
    void Simulator<T>::sparsity(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "sparsity";
        sys::Stats stats = sys::Stats(network.getNumLayers(), 1, filename);

        auto act_sparsity = stats.register_double_t("act_sparsity", 0, sys::Average);
        auto zero_act = stats.register_uint_t("zero_act", 0, sys::AverageTotal);
        auto total_act = stats.register_uint_t("total_act", 0, sys::AverageTotal);
        auto wgt_sparsity = stats.register_double_t("wgt_sparsity", 0, sys::Average);
        auto zero_wgt = stats.register_uint_t("zero_wgt", 0, sys::AverageTotal);
        auto total_wgt = stats.register_uint_t("total_wgt", 0, sys::AverageTotal);

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];

            uint64_t batch_zero_act = 0;
            const auto &act = layer.getActivations();
            for(uint64_t i = 0; i < act.getMax_index(); i++) {
                const auto data = act.get(i);
                if(data == 0) batch_zero_act++;
            }

            uint64_t batch_zero_wgt = 0;
            const auto &wgt = layer.getWeights();
            for(uint64_t i = 0; i < wgt.getMax_index(); i++) {
                const auto data = wgt.get(i);
                if(data == 0) batch_zero_wgt++;
            }

            act_sparsity->value[layer_it][0] = batch_zero_act / (double)act.getMax_index() * 100.;
            zero_act->value[layer_it][0] = batch_zero_act;
            total_act->value[layer_it][0] = act.getMax_index();
            wgt_sparsity->value[layer_it][0] = batch_zero_wgt / (double)wgt.getMax_index() * 100.;
            zero_wgt->value[layer_it][0] = batch_zero_wgt;
            total_wgt->value[layer_it][0] = wgt.getMax_index();

        }

        //Dump statistics
        std::string header = "Value sparsity for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, QUIET);

    }

    template <typename uint16_t>
    void Simulator<uint16_t>::bit_sparsity(const base::Network<uint16_t> &network) {

        // Initialize statistics
        std::string filename = "bit_sparsity";
        sys::Stats stats = sys::Stats(network.getNumLayers(), 1, filename);

        auto act_sparsity = stats.register_double_t("act_bit_sparsity", 0, sys::Average);
        auto zero_act = stats.register_uint_t("zero_act_bits", 0, sys::AverageTotal);
        auto total_act = stats.register_uint_t("total_act_bits", 0, sys::AverageTotal);
        auto wgt_sparsity = stats.register_double_t("wgt_bit_sparsity", 0, sys::Average);
        auto zero_wgt = stats.register_uint_t("zero_wgt_bits", 0, sys::AverageTotal);
        auto total_wgt = stats.register_uint_t("total_wgt_bits", 0, sys::AverageTotal);

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

            const base::Layer<uint16_t> &layer = network.getLayers()[layer_it];

            uint64_t zero_act_bits = 0;
            const auto &act = layer.getActivations();
            for(uint64_t i = 0; i < act.getMax_index(); i++) {
                const auto bits = act.get(i);
                uint8_t ones = effectualBits(bits);
                zero_act_bits += (16 - ones);
            }

            uint64_t zero_wgt_bits = 0;
            const auto &wgt = layer.getWeights();
            for(uint64_t i = 0; i < wgt.getMax_index(); i++) {
                const auto bits = wgt.get(i);
                uint8_t ones = effectualBits(bits);
                zero_wgt_bits += (16 - ones);
            }

            act_sparsity->value[layer_it][0] = zero_act_bits / (double)(act.getMax_index() * 16.) * 100.;
            zero_act->value[layer_it][0] = zero_act_bits;
            total_act->value[layer_it][0] = act.getMax_index() * 16.;
            wgt_sparsity->value[layer_it][0] = zero_wgt_bits / (double)(wgt.getMax_index() * 16.) * 100.;
            zero_wgt->value[layer_it][0] = zero_wgt_bits;
            total_wgt->value[layer_it][0] = wgt.getMax_index() * 16.;

        }

        //Dump statistics
        std::string header = "Bit sparsity for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template <typename T>
	void Simulator<T>::training_sparsity(const sys::Batch::Simulate &simulate, int epochs) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string filename = "training_value_sparsity";
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        // Forward stats
        auto fw_act_sparsity = stats.register_double_t("fw_act_sparsity", 0, sys::Average);
        auto fw_zero_act = stats.register_uint_t("fw_zero_act", 0, sys::AverageTotal);
        auto fw_total_act = stats.register_uint_t("fw_total_act", 0, sys::AverageTotal);
        auto fw_wgt_sparsity = stats.register_double_t("fw_wgt_sparsity", 0, sys::Average);
        auto fw_zero_wgt = stats.register_uint_t("fw_zero_wgt", 0, sys::AverageTotal);
        auto fw_total_wgt = stats.register_uint_t("fw_total_wgt", 0, sys::AverageTotal);

        auto bw_in_grad_sparsity = stats.register_double_t("bw_in_grad_sparsity", 0, sys::Average);
        auto bw_zero_in_grad = stats.register_uint_t("bw_zero_in_grad", 0, sys::AverageTotal);
        auto bw_total_in_grad = stats.register_uint_t("bw_total_in_grad", 0, sys::AverageTotal);
        auto bw_wgt_grad_sparsity = stats.register_double_t("bw_wgt_grad_sparsity", 0, sys::Average);
        auto bw_zero_wgt_grad = stats.register_uint_t("bw_zero_wgt_grad", 0, sys::AverageTotal);
        auto bw_total_wgt_grad = stats.register_uint_t("bw_total_wgt_grad", 0, sys::AverageTotal);
        auto bw_out_grad_sparsity = stats.register_double_t("bw_out_grad_sparsity", 0, sys::Average);
        auto bw_zero_out_grad = stats.register_uint_t("bw_zero_out_grad", 0, sys::AverageTotal);
        auto bw_total_out_grad = stats.register_uint_t("bw_total_out_grad", 0, sys::AverageTotal);

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

	    for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<float> network;
            network = read_training<float>(simulate.network, simulate.batch, epoch,
                    simulate.decoder_states, traces_mode, QUIET);

            for (int layer_it = 0; layer_it < network.getLayers().size(); layer_it++) {

                const base::Layer<float> &layer = network.getLayers()[layer_it];

                // Forward
                if (network.getForward()) {
                    uint64_t zero_act = 0;
                    const auto &act = layer.getActivations();
                    for (uint64_t i = 0; i < act.getMax_index(); i++) {
                        const auto data = act.get(i);
                        if (data == 0) zero_act++;
                    }

                    uint64_t zero_wgt = 0;
                    const auto &wgt = layer.getWeights();
                    for (uint64_t i = 0; i < wgt.getMax_index(); i++) {
                        const auto data = wgt.get(i);
                        if (data == 0) zero_wgt++;
                    }

                    fw_act_sparsity->value[layer_it][epoch] = zero_act / (double) act.getMax_index() * 100.;
                    fw_zero_act->value[layer_it][epoch] = zero_act;
                    fw_total_act->value[layer_it][epoch] = act.getMax_index();
                    fw_wgt_sparsity->value[layer_it][epoch] = zero_wgt / (double) wgt.getMax_index() * 100.;
                    fw_zero_wgt->value[layer_it][epoch] = zero_wgt;
                    fw_total_wgt->value[layer_it][epoch] = wgt.getMax_index();
                }

                //Backward
                if (network.getBackward()) {
                    uint64_t zero_act_grad = 0;
                    const auto &act_grad = layer.getInputGradients();
                    if (layer_it != 0) {
                        for (uint64_t i = 0; i < act_grad.getMax_index(); i++) {
                            const auto data = act_grad.get(i);
                            if (data == 0) zero_act_grad++;
                        }
                    }

                    uint64_t zero_wgt_grad = 0;
                    const auto &wgt_grad = layer.getWeightGradients();
                    for (uint64_t i = 0; i < wgt_grad.getMax_index(); i++) {
                        const auto data = wgt_grad.get(i);
                        if (data == 0) zero_wgt_grad++;
                    }

                    uint64_t zero_out_act_grad = 0;
                    const auto &out_act_grad = layer.getOutputGradients();
                    for (uint64_t i = 0; i < out_act_grad.getMax_index(); i++) {
                        const auto data = out_act_grad.get(i);
                        if (data == 0) zero_out_act_grad++;
                    }

                    if (layer_it != 0) {
                        bw_in_grad_sparsity->value[layer_it][epoch] =
                                zero_act_grad / (double) act_grad.getMax_index() * 100.;
                        bw_zero_in_grad->value[layer_it][epoch] = zero_act_grad;
                        bw_total_in_grad->value[layer_it][epoch] = act_grad.getMax_index();
                    }
                    bw_wgt_grad_sparsity->value[layer_it][epoch] =
                            zero_wgt_grad / (double) wgt_grad.getMax_index() * 100.;
                    bw_zero_wgt_grad->value[layer_it][epoch] = zero_wgt_grad;
                    bw_total_wgt_grad->value[layer_it][epoch] = wgt_grad.getMax_index();
                    bw_out_grad_sparsity->value[layer_it][epoch] = zero_out_act_grad /
                            (double) out_act_grad.getMax_index() * 100.;
                    bw_zero_out_grad->value[layer_it][epoch] = zero_out_act_grad;
                    bw_total_out_grad->value[layer_it][epoch] = out_act_grad.getMax_index();
                }

            }

        }

        //Dump statistics
        std::string header = "Value sparsity for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

	}

    template <typename T>
    void Simulator<T>::training_bit_sparsity(const sys::Batch::Simulate &simulate, int epochs, bool mantissa) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string task_name = mantissa ? "mantissa_bit_sparsity" : "exponent_bit_sparsity";
        std::string filename = "training_" + task_name;
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        // Forward stats
        auto fw_act_sparsity = stats.register_double_t("fw_act_bit_sparsity", 0, sys::Average);
        auto fw_zero_act = stats.register_uint_t("fw_zero_act_bits", 0, sys::AverageTotal);
        auto fw_total_act = stats.register_uint_t("fw_total_act_bits", 0, sys::AverageTotal);
        auto fw_wgt_sparsity = stats.register_double_t("fw_wgt_bit_sparsity", 0, sys::Average);
        auto fw_zero_wgt = stats.register_uint_t("fw_zero_wgt_bits", 0, sys::AverageTotal);
        auto fw_total_wgt = stats.register_uint_t("fw_total_wgt_bits", 0, sys::AverageTotal);

        auto bw_in_grad_sparsity = stats.register_double_t("bw_in_grad_bit_sparsity", 0, sys::Average);
        auto bw_zero_in_grad = stats.register_uint_t("bw_zero_in_grad_bits", 0, sys::AverageTotal);
        auto bw_total_in_grad = stats.register_uint_t("bw_total_in_grad_bits", 0, sys::AverageTotal);
        auto bw_wgt_grad_sparsity = stats.register_double_t("bw_wgt_grad_bit_sparsity", 0, sys::Average);
        auto bw_zero_wgt_grad = stats.register_uint_t("bw_zero_wgt_grad_bits", 0, sys::AverageTotal);
        auto bw_total_wgt_grad = stats.register_uint_t("bw_total_wgt_grad_bits", 0, sys::AverageTotal);
        auto bw_out_grad_sparsity = stats.register_double_t("bw_out_grad_bit_sparsity", 0, sys::Average);
        auto bw_zero_out_grad = stats.register_uint_t("bw_zero_out_grad_bits", 0, sys::AverageTotal);
        auto bw_total_out_grad = stats.register_uint_t("bw_total_out_grad_bits", 0, sys::AverageTotal);

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

        const auto MAX_ONES = mantissa ? 7. : 8.;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<float> network;
            network = read_training<float>(simulate.network, simulate.batch, epoch,
                    simulate.decoder_states, traces_mode, QUIET);

            for (int layer_it = 0; layer_it < network.getLayers().size(); layer_it++) {

                const base::Layer<float> &layer = network.getLayers()[layer_it];

                // Forward
                if (network.getForward()) {
                    uint64_t zero_act_bits = 0;
                    const auto &act = layer.getActivations();
                    for(uint64_t i = 0; i < act.getMax_index(); i++) {
                        const auto data_float = act.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        auto ones = effectualBits(bin_value);
                        zero_act_bits += (MAX_ONES - ones);
                    }

                    uint64_t zero_wgt_bits = 0;
                    const auto &wgt = layer.getWeights();
                    for(uint64_t i = 0; i < wgt.getMax_index(); i++) {
                        const auto data_float = wgt.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        auto ones = effectualBits(bin_value);
                        zero_wgt_bits += (MAX_ONES - ones);
                    }

                    fw_act_sparsity->value[layer_it][epoch] = zero_act_bits / (double) act.getMax_index() * 100.;
                    fw_zero_act->value[layer_it][epoch] = zero_act_bits;
                    fw_total_act->value[layer_it][epoch] = act.getMax_index();
                    fw_wgt_sparsity->value[layer_it][epoch] = zero_wgt_bits / (double) wgt.getMax_index() * 100.;
                    fw_zero_wgt->value[layer_it][epoch] = zero_wgt_bits;
                    fw_total_wgt->value[layer_it][epoch] = wgt.getMax_index();
                }

                //Backward
                if (network.getBackward()) {
                    uint64_t zero_act_grad_bits = 0;
                    const auto &act_grad = layer.getInputGradients();
                    if(layer_it != 0) {
                        for (uint64_t i = 0; i < act_grad.getMax_index(); i++) {
                            const auto data_float = act_grad.get(i);
                            auto data_bfloat = this->split_bfloat16(data_float);
                            auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                            auto ones = effectualBits(bin_value);
                            zero_act_grad_bits += (MAX_ONES - ones);
                        }
                    }

                    uint64_t zero_wgt_grad_bits = 0;
                    const auto &wgt_grad = layer.getWeightGradients();
                    for(uint64_t i = 0; i < wgt_grad.getMax_index(); i++) {
                        const auto data_float = wgt_grad.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        auto ones = effectualBits(bin_value);
                        zero_wgt_grad_bits += (MAX_ONES - ones);
                    }

                    uint64_t zero_out_act_grad_bits = 0;
                    const auto &out_act_grad = layer.getOutputGradients();
                    for (uint64_t i = 0; i < out_act_grad.getMax_index(); i++) {
                        const auto data_float = out_act_grad.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        auto ones = effectualBits(bin_value);
                        zero_out_act_grad_bits += (MAX_ONES - ones);
                    }

                    if (layer_it != 0) {
                        bw_in_grad_sparsity->value[layer_it][epoch] =
                                zero_act_grad_bits / (double) act_grad.getMax_index() * 100.;
                        bw_zero_in_grad->value[layer_it][epoch] = zero_act_grad_bits;
                        bw_total_in_grad->value[layer_it][epoch] = act_grad.getMax_index();
                    }
                    bw_wgt_grad_sparsity->value[layer_it][epoch] =
                            zero_wgt_grad_bits / (double) wgt_grad.getMax_index() * 100.;
                    bw_zero_wgt_grad->value[layer_it][epoch] = zero_wgt_grad_bits;
                    bw_total_wgt_grad->value[layer_it][epoch] = wgt_grad.getMax_index();
                    bw_out_grad_sparsity->value[layer_it][epoch] = zero_out_act_grad_bits /
                                                                   (double) out_act_grad.getMax_index() * 100.;
                    bw_zero_out_grad->value[layer_it][epoch] = zero_out_act_grad_bits;
                    bw_total_out_grad->value[layer_it][epoch] = out_act_grad.getMax_index();
                }

            }

        }

        //Dump statistics
        std::string header = "Bit sparsity for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }


    /*template <typename T>
    void Simulator<T>::training_distribution(const Network<T> &network, sys::Statistics::Stats &stats,
            int epoch, int epochs, bool mantissa) {

        const auto MAX_VALUE = mantissa ? 128 : 256;

        if(epoch == 0) {
            stats.task_name = mantissa ? "mantissa_distribution" : "exponent_distribution";
            stats.net_name = network.getName();
            stats.arch = "None";
            stats.mantissa_data = mantissa;

            stats.fw_act_values = std::vector<std::vector<std::vector<uint64_t>>>(MAX_VALUE);
            stats.fw_wgt_values = std::vector<std::vector<std::vector<uint64_t>>>(MAX_VALUE);
            stats.bw_in_grad_values = std::vector<std::vector<std::vector<uint64_t>>>(MAX_VALUE);
            stats.bw_wgt_grad_values = std::vector<std::vector<std::vector<uint64_t>>>(MAX_VALUE);
            stats.bw_out_grad_values = std::vector<std::vector<std::vector<uint64_t>>>(MAX_VALUE);
        }

        for(int layer_it = 0; layer_it < network.getLayers().size(); layer_it++) {

            const Layer<T> &layer = network.getLayers()[layer_it];

            if(epoch == 0) {
                stats.layers.push_back(layer.getName());

                for(int i = 0; i < MAX_VALUE; i++) {
                    stats.fw_act_values[i].emplace_back(std::vector<uint64_t>(epochs, 0));
                    stats.fw_wgt_values[i].emplace_back(std::vector<uint64_t>(epochs, 0));
                    stats.bw_in_grad_values[i].emplace_back(std::vector<uint64_t>(epochs, 0));
                    stats.bw_wgt_grad_values[i].emplace_back(std::vector<uint64_t>(epochs, 0));
                    stats.bw_out_grad_values[i].emplace_back(std::vector<uint64_t>(epochs, 0));
                }
            }

            // Forward
            if(network.getForward()) {
                const auto &act = layer.getActivations();
                for(uint64_t i = 0; i < act.getMax_index(); i++) {
                    const auto data_float = act.get(i);
                    auto data_bfloat = this->split_bfloat16(data_float);
                    auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                    stats.fw_act_values[bin_value][layer_it][epoch]++;
                }

                const auto &wgt = layer.getWeights();
                for(uint64_t i = 0; i < wgt.getMax_index(); i++) {
                    const auto data_float = wgt.get(i);
                    auto data_bfloat = this->split_bfloat16(data_float);
                    auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                    stats.fw_wgt_values[bin_value][layer_it][epoch]++;
                }
            }

            //Backward
            if(network.getBackward()) {
                const auto &act_grad = layer.getInputGradients();
                if(layer_it != 0) {
                    for (uint64_t i = 0; i < act_grad.getMax_index(); i++) {
                        const auto data_float = act_grad.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        stats.bw_in_grad_values[bin_value][layer_it][epoch]++;
                    }
                }

                const auto &wgt_grad = layer.getWeightGradients();
                for(uint64_t i = 0; i < wgt_grad.getMax_index(); i++) {
                    const auto data_float = wgt_grad.get(i);
                    auto data_bfloat = this->split_bfloat16(data_float);
                    auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                    stats.bw_wgt_grad_values[bin_value][layer_it][epoch]++;
                }

                const auto &out_act_grad = layer.getOutputGradients();
                for (uint64_t i = 0; i < out_act_grad.getMax_index(); i++) {
                    const auto data_float = out_act_grad.get(i);
                    auto data_bfloat = this->split_bfloat16(data_float);
                    auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                    stats.bw_out_grad_values[bin_value][layer_it][epoch]++;
                }
            }

        }

    }*/

    INITIALISE_DATA_TYPES(Simulator);

}
