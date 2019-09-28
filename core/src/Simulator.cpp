
#include <core/Simulator.h>

namespace core {

    /* COMMON FUNCTIONS */

    template <typename T>
    base::Network<T> Simulator<T>::read_training(const std::string &network_name, uint32_t batch, uint32_t epoch,
            uint32_t decoder_states, uint32_t traces_mode, bool accelerator) {

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
            if (!accelerator) {
                reader.read_training_weight_gradients_npy(network);
                reader.read_training_input_gradients_npy(network);
            }
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
    void Simulator<T>::information(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "information";
        sys::Stats stats = sys::Stats(network.getNumLayers(), 1, filename);

        auto type = stats.register_string_t("Type", sys::No_Measure);

        auto batch = stats.register_uint_t("Batch", 0, sys::Average);
        auto act_channels = stats.register_uint_t("Act Channels", 0, sys::Average);
        auto Nx = stats.register_uint_t("Act Width", 0, sys::Average);
        auto Ny = stats.register_uint_t("Act Height", 0, sys::Average);
        auto R = stats.register_uint_t("Times", 0, sys::Average);

        auto filters = stats.register_uint_t("Num Filters", 0, sys::Average);
        auto wgt_channels = stats.register_uint_t("Wgt Channels", 0, sys::Average);
        auto Kx = stats.register_uint_t("Wgt Width", 0, sys::Average);
        auto Ky = stats.register_uint_t("Wgt Height", 0, sys::Average);

        auto out_x = stats.register_uint_t("Output Width", 0, sys::Average);
        auto out_y = stats.register_uint_t("Output Height", 0, sys::Average);

        auto padding = stats.register_uint_t("Padding", 0, sys::Average);
        auto stride = stats.register_uint_t("Stride", 0, sys::Average);
        auto act_precision = stats.register_uint_t("Act Precision", 0, sys::Average);
        auto wgt_precision = stats.register_uint_t("Wgt Precision", 0, sys::Average);

        auto act_size = stats.register_double_t("Act Size (MB)", 0, sys::AverageTotal);
        auto act_row_size = stats.register_double_t("Act Row Size (MB)", 0, sys::AverageTotal);
        auto wgt_size = stats.register_double_t("Wgt Size (MB)", 0, sys::AverageTotal);
        auto wgt_set_size = stats.register_double_t("Wgt Working Set Size (MB)", 0, sys::AverageTotal);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();
            if(fc) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            type->value[layer_it][0] = layer.getType();

            if(lstm) {
                R->value[layer_it][0] = act_shape[0];
                batch->value[layer_it][0] = act_shape[1];
                act_channels->value[layer_it][0] = act_shape[2];
                Nx->value[layer_it][0] = 1;
                Ny->value[layer_it][0] = 1;
            } else {
                R->value[layer_it][0] = 1;
                batch->value[layer_it][0] = act_shape[0];
                act_channels->value[layer_it][0] = act_shape[1];
                Nx->value[layer_it][0] = act_shape[2];
                Ny->value[layer_it][0] = act_shape[3];
            }

            filters->value[layer_it][0] = wgt_shape[0];
            wgt_channels->value[layer_it][0] = wgt_shape[1];
            Kx->value[layer_it][0] = wgt_shape[2];
            Ky->value[layer_it][0] = wgt_shape[3];

            padding->value[layer_it][0] = layer.getPadding();
            stride->value[layer_it][0] = layer.getStride();
            act_precision->value[layer_it][0] = layer.getActPrecision();
            wgt_precision->value[layer_it][0] = layer.getWgtPrecision();

            out_x->value[layer_it][0] = (Nx->value[layer_it][0] - Kx->value[layer_it][0])/stride->value[layer_it][0] + 1;
            out_y->value[layer_it][0] = (Ny->value[layer_it][0] - Ky->value[layer_it][0])/stride->value[layer_it][0] + 1;

            act_size->value[layer_it][0] = act_channels->value[layer_it][0] * Nx->value[layer_it][0] *
                    Ny->value[layer_it][0] * R->value[layer_it][0] * network.getNetwork_bits() / 8000000.;
            act_row_size->value[layer_it][0] = act_channels->value[layer_it][0] * Nx->value[layer_it][0] *
                    Ky->value[layer_it][0] * network.getNetwork_bits() / 8000000.;
            wgt_size->value[layer_it][0] = filters->value[layer_it][0] * wgt_channels->value[layer_it][0] *
                    Kx->value[layer_it][0] * Ky->value[layer_it][0] * network.getNetwork_bits() / 8000000.;
            wgt_set_size->value[layer_it][0] = 16 * wgt_channels->value[layer_it][0] * Kx->value[layer_it][0] *
                    Ky->value[layer_it][0] * network.getNetwork_bits() / 8000000.;
        }

        //Dump statistics
        std::string header = "Information for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, QUIET);


    }

    template <typename T>
    void Simulator<T>::sparsity(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "value_sparsity";
        sys::Stats stats = sys::Stats(network.getNumLayers(), 1, filename);

        auto act_sparsity = stats.register_double_t("act_sparsity", 0, sys::Special);
        auto zero_act = stats.register_uint_t("zero_act", 0, sys::AverageTotal);
        auto total_act = stats.register_uint_t("total_act", 0, sys::AverageTotal);
        auto wgt_sparsity = stats.register_double_t("wgt_sparsity", 0, sys::Special);
        auto zero_wgt = stats.register_uint_t("zero_wgt", 0, sys::AverageTotal);
        auto total_wgt = stats.register_uint_t("total_wgt", 0, sys::AverageTotal);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

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

        act_sparsity->special_value = sys::get_total(zero_act->value) / (double)sys::get_total(total_act->value) * 100.;
        wgt_sparsity->special_value = sys::get_total(zero_wgt->value) / (double)sys::get_total(total_wgt->value) * 100.;

        //Dump statistics
        std::string header = "Value sparsity for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, QUIET);

    }

    template <typename uint16_t>
    void Simulator<uint16_t>::bit_sparsity(const base::Network<uint16_t> &network) {

        // Initialize statistics
        std::string filename = "bit_sparsity";
        sys::Stats stats = sys::Stats(network.getNumLayers(), 1, filename);

        auto act_sparsity = stats.register_double_t("act_bit_sparsity", 0, sys::Special);
        auto zero_act = stats.register_uint_t("zero_act_bits", 0, sys::AverageTotal);
        auto total_act = stats.register_uint_t("total_act_bits", 0, sys::AverageTotal);
        auto wgt_sparsity = stats.register_double_t("wgt_bit_sparsity", 0, sys::Special);
        auto zero_wgt = stats.register_uint_t("zero_wgt_bits", 0, sys::AverageTotal);
        auto total_wgt = stats.register_uint_t("total_wgt_bits", 0, sys::AverageTotal);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

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

        act_sparsity->special_value = sys::get_total(zero_act->value) / (double)sys::get_total(total_act->value) * 100.;
        wgt_sparsity->special_value = sys::get_total(zero_wgt->value) / (double)sys::get_total(total_wgt->value) * 100.;

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
        auto fw_act_sparsity = stats.register_double_t("fw_act_sparsity", 0, sys::Special);
        auto fw_zero_act = stats.register_uint_t("fw_zero_act", 0, sys::AverageTotal);
        auto fw_total_act = stats.register_uint_t("fw_total_act", 0, sys::AverageTotal);
        auto fw_wgt_sparsity = stats.register_double_t("fw_wgt_sparsity", 0, sys::Special);
        auto fw_zero_wgt = stats.register_uint_t("fw_zero_wgt", 0, sys::AverageTotal);
        auto fw_total_wgt = stats.register_uint_t("fw_total_wgt", 0, sys::AverageTotal);

        // Backward stats
        auto bw_in_grad_sparsity = stats.register_double_t("bw_in_grad_sparsity", 0, sys::Special);
        auto bw_zero_in_grad = stats.register_uint_t("bw_zero_in_grad", 0, sys::AverageTotal);
        auto bw_total_in_grad = stats.register_uint_t("bw_total_in_grad", 0, sys::AverageTotal);
        auto bw_wgt_grad_sparsity = stats.register_double_t("bw_wgt_grad_sparsity", 0, sys::Special);
        auto bw_zero_wgt_grad = stats.register_uint_t("bw_zero_wgt_grad", 0, sys::AverageTotal);
        auto bw_total_wgt_grad = stats.register_uint_t("bw_total_wgt_grad", 0, sys::AverageTotal);
        auto bw_out_grad_sparsity = stats.register_double_t("bw_out_grad_sparsity", 0, sys::Special);
        auto bw_zero_out_grad = stats.register_uint_t("bw_zero_out_grad", 0, sys::AverageTotal);
        auto bw_total_out_grad = stats.register_uint_t("bw_total_out_grad", 0, sys::AverageTotal);

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

	    for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, traces_mode, false);

            if(!QUIET) std::cout << "Starting simulation training sparsity for epoch " << epoch << std::endl;

            for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                const base::Layer<T> &layer = network.getLayers()[layer_it];

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

        fw_act_sparsity->special_value = sys::get_total(fw_zero_act->value) / (double)sys::get_total(fw_total_act->value) * 100.;
        fw_wgt_sparsity->special_value = sys::get_total(fw_zero_wgt->value) / (double)sys::get_total(fw_total_wgt->value) * 100.;
        bw_in_grad_sparsity->special_value = sys::get_total(bw_zero_in_grad->value) / (double)sys::get_total(bw_total_in_grad->value) * 100.;
        bw_wgt_grad_sparsity->special_value = sys::get_total(bw_zero_wgt_grad->value) / (double)sys::get_total(bw_total_wgt_grad->value) * 100.;
        bw_out_grad_sparsity->special_value = sys::get_total(bw_zero_out_grad->value) / (double)sys::get_total(bw_total_out_grad->value) * 100.;

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
        auto fw_act_sparsity = stats.register_double_t("fw_act_bit_sparsity", 0, sys::Special);
        auto fw_zero_act = stats.register_uint_t("fw_zero_act_bits", 0, sys::AverageTotal);
        auto fw_total_act = stats.register_uint_t("fw_total_act_bits", 0, sys::AverageTotal);
        auto fw_wgt_sparsity = stats.register_double_t("fw_wgt_bit_sparsity", 0, sys::Special);
        auto fw_zero_wgt = stats.register_uint_t("fw_zero_wgt_bits", 0, sys::AverageTotal);
        auto fw_total_wgt = stats.register_uint_t("fw_total_wgt_bits", 0, sys::AverageTotal);

        // Backward stats
        auto bw_in_grad_sparsity = stats.register_double_t("bw_in_grad_bit_sparsity", 0, sys::Special);
        auto bw_zero_in_grad = stats.register_uint_t("bw_zero_in_grad_bits", 0, sys::AverageTotal);
        auto bw_total_in_grad = stats.register_uint_t("bw_total_in_grad_bits", 0, sys::AverageTotal);
        auto bw_wgt_grad_sparsity = stats.register_double_t("bw_wgt_grad_bit_sparsity", 0, sys::Special);
        auto bw_zero_wgt_grad = stats.register_uint_t("bw_zero_wgt_grad_bits", 0, sys::AverageTotal);
        auto bw_total_wgt_grad = stats.register_uint_t("bw_total_wgt_grad_bits", 0, sys::AverageTotal);
        auto bw_out_grad_sparsity = stats.register_double_t("bw_out_grad_bit_sparsity", 0, sys::Special);
        auto bw_zero_out_grad = stats.register_uint_t("bw_zero_out_grad_bits", 0, sys::AverageTotal);
        auto bw_total_out_grad = stats.register_uint_t("bw_total_out_grad_bits", 0, sys::AverageTotal);

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

        const auto MAX_ONES = mantissa ? 7. : 8.;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, traces_mode, false);

            if(!QUIET) std::cout << "Starting simulation training bit sparsity for epoch " << epoch << std::endl;

            for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                const base::Layer<T> &layer = network.getLayers()[layer_it];

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

                    fw_act_sparsity->value[layer_it][epoch] = zero_act_bits / (double)(act.getMax_index() * MAX_ONES) * 100.;
                    fw_zero_act->value[layer_it][epoch] = zero_act_bits;
                    fw_total_act->value[layer_it][epoch] = act.getMax_index() * MAX_ONES;
                    fw_wgt_sparsity->value[layer_it][epoch] = zero_wgt_bits / (double)(wgt.getMax_index()  * MAX_ONES) * 100.;
                    fw_zero_wgt->value[layer_it][epoch] = zero_wgt_bits;
                    fw_total_wgt->value[layer_it][epoch] = wgt.getMax_index() * MAX_ONES;
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
                                zero_act_grad_bits / (double)(act_grad.getMax_index() * MAX_ONES) * 100.;
                        bw_zero_in_grad->value[layer_it][epoch] = zero_act_grad_bits;
                        bw_total_in_grad->value[layer_it][epoch] = act_grad.getMax_index() * MAX_ONES;
                    }
                    bw_wgt_grad_sparsity->value[layer_it][epoch] =
                            zero_wgt_grad_bits / (double)(wgt_grad.getMax_index() * MAX_ONES) * 100.;
                    bw_zero_wgt_grad->value[layer_it][epoch] = zero_wgt_grad_bits;
                    bw_total_wgt_grad->value[layer_it][epoch] = wgt_grad.getMax_index() * MAX_ONES;
                    bw_out_grad_sparsity->value[layer_it][epoch] = zero_out_act_grad_bits /
                            (double)(out_act_grad.getMax_index()  * MAX_ONES) * 100.;
                    bw_zero_out_grad->value[layer_it][epoch] = zero_out_act_grad_bits;
                    bw_total_out_grad->value[layer_it][epoch] = out_act_grad.getMax_index() * MAX_ONES;
                }

            }

        }

        fw_act_sparsity->special_value = sys::get_total(fw_zero_act->value) / (double)sys::get_total(fw_total_act->value) * 100.;
        fw_wgt_sparsity->special_value = sys::get_total(fw_zero_wgt->value) / (double)sys::get_total(fw_total_wgt->value) * 100.;
        bw_in_grad_sparsity->special_value = sys::get_total(bw_zero_in_grad->value) / (double)sys::get_total(bw_total_in_grad->value) * 100.;
        bw_wgt_grad_sparsity->special_value = sys::get_total(bw_zero_wgt_grad->value) / (double)sys::get_total(bw_total_wgt_grad->value) * 100.;
        bw_out_grad_sparsity->special_value = sys::get_total(bw_zero_out_grad->value) / (double)sys::get_total(bw_total_out_grad->value) * 100.;

        //Dump statistics
        std::string header = mantissa ? "Mantissa" : "Exponent";
        header += " Bit sparsity for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }


    template <typename T>
    void Simulator<T>::training_distribution(const sys::Batch::Simulate &simulate, int epochs, bool mantissa) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string task_name = mantissa ? "mantissa_distribution" : "exponent_distribution";
        std::string filename = "training_" + task_name;
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        int min_range = mantissa ? 0 : -127;
        int max_range = mantissa ? 127 : 128;
        auto fw_act_values = stats.register_uint_dist_t("Forward Activations Distribution", min_range, max_range, 0, sys::AverageTotal);
        auto fw_wgt_values = stats.register_uint_dist_t("Forward Weights Distribution", min_range, max_range, 0, sys::AverageTotal);
        auto bw_in_grad_values = stats.register_uint_dist_t("Backward Input Gradients Distribution", min_range, max_range, 0, sys::AverageTotal);
        auto bw_wgt_grad_values = stats.register_uint_dist_t("Backward Weight Gradients Distribution", min_range, max_range, 0, sys::AverageTotal);
        auto bw_out_grad_values = stats.register_uint_dist_t("Backward Output Gradients Distribution", min_range, max_range, 0, sys::AverageTotal);

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, traces_mode, false);

            if(!QUIET) std::cout << "Starting simulation training distribution for epoch " << epoch << std::endl;

            for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                const base::Layer<T> &layer = network.getLayers()[layer_it];

                // Forward
                if (network.getForward()) {
                    const auto &act = layer.getActivations();
                    for(uint64_t i = 0; i < act.getMax_index(); i++) {
                        const auto data_float = act.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        fw_act_values->value[bin_value][layer_it][epoch]++;
                    }

                    const auto &wgt = layer.getWeights();
                    for(uint64_t i = 0; i < wgt.getMax_index(); i++) {
                        const auto data_float = wgt.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        fw_wgt_values->value[bin_value][layer_it][epoch]++;
                    }
                }

                //Backward
                if (network.getBackward()) {
                    const auto &act_grad = layer.getInputGradients();
                    if(layer_it != 0) {
                        for (uint64_t i = 0; i < act_grad.getMax_index(); i++) {
                            const auto data_float = act_grad.get(i);
                            auto data_bfloat = this->split_bfloat16(data_float);
                            auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                            bw_in_grad_values->value[bin_value][layer_it][epoch]++;
                        }
                    }

                    const auto &wgt_grad = layer.getWeightGradients();
                    for(uint64_t i = 0; i < wgt_grad.getMax_index(); i++) {
                        const auto data_float = wgt_grad.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        bw_wgt_grad_values->value[bin_value][layer_it][epoch]++;
                    }

                    const auto &out_act_grad = layer.getOutputGradients();
                    for (uint64_t i = 0; i < out_act_grad.getMax_index(); i++) {
                        const auto data_float = out_act_grad.get(i);
                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto bin_value = mantissa ? std::get<2>(data_bfloat) : std::get<1>(data_bfloat);
                        bw_out_grad_values->value[bin_value][layer_it][epoch]++;
                    }
                }

            }

        }

        //Dump statistics
        std::string header = mantissa ? "Mantissa" : "Exponent";
        header += " Distribution for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    INITIALISE_DATA_TYPES(Simulator);

}
