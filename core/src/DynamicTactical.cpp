
#include <core/DynamicTactical.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t DynamicTactical<T>::computeDynamicTacticalBitsPE(T first, T second, bool first_value) {

        #ifdef ZERO_COUNT
        if(first_value && first == 0) return 0;
        else if(!first_value && second == 0) return 0;
        #else
        if(first_value && first == 0) return 0;
        else if(!first_value && second == 0) return 0;
        #endif
        else return 256;

    }

    /* CYCLES */

    template <typename T>
    void DynamicTactical<T>::run(const sys::Batch::Simulate &simulate, int epochs) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, this->QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string arch = "DynamicTactical";
        std::string filename = arch + "_cycles";
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        // Forward stats

        // Backward stats

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else traces_mode = 5;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, traces_mode);

            if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles forward simulation for epoch "
                                       << epoch << std::endl;

            for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                const base::Layer<float> &layer = network.getLayers()[layer_it];

                // Fil weight buffer

                // Fil activations buffer

                // Array of future dram positions. Use wards to calculate dram coming streams

                // Check results

            }

        }

        //Dump statistics
        std::string header = "DynamicTactical Number of Cycles for " + network_model.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(N_TILES) + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    /* CYCLES */

    template <typename T>
    void DynamicTactical<T>::potentials(const sys::Batch::Simulate &simulate, int epochs) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, this->QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string arch = "DynamicTactical";
        std::string filename = arch + "_cycles";
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        // Forward stats
        auto act_work_reduction = stats.register_double_t("Forward Activations Work Reduction", 0, sys::Average);
        auto act_speedup = stats.register_double_t("Forward Activations Speedup", 0, sys::Average);
        auto act_par_mult = stats.register_double_t("Forward Activations Multiplications", 0, sys::AverageTotal);
        auto act_bit_multiplications = stats.register_uint_t("Forward Activations Bit Multiplications", 0, sys::AverageTotal);

        auto wgt_work_reduction = stats.register_double_t("Forward Weights Work Reduction", 0, sys::Average);
        auto wgt_speedup = stats.register_double_t("Forward Weights Speedup", 0, sys::Average);
        auto wgt_par_mult = stats.register_double_t("Forward Weights Multiplications", 0, sys::AverageTotal);
        auto wgt_bit_multiplications = stats.register_uint_t("Forward Weights Bit Multiplications", 0, sys::AverageTotal);

        // Backward stats
        auto bw_wgt_work_reduction = stats.register_double_t("Backward Weights Work Reduction", 0, sys::Average);
        auto bw_wgt_speedup = stats.register_double_t("Backward Weights Speedup", 0, sys::Average);
        auto bw_wgt_par_mult = stats.register_double_t("Backward Weights Multiplications", 0, sys::AverageTotal);
        auto bw_wgt_bit_multiplications = stats.register_uint_t("Backward Weights Bit Multiplications", 0, sys::AverageTotal);

        auto wgt_out_grad_work_reduction = stats.register_double_t("Backward Output Gradients Work Reduction", 0, sys::Average);
        auto wgt_out_grad_speedup = stats.register_double_t("Backward Output Gradients Speedup", 0, sys::Average);
        auto wgt_out_grad_par_mult = stats.register_double_t("Backward Output Gradients Multiplications", 0, sys::AverageTotal);
        auto wgt_out_grad_bit_multiplications = stats.register_uint_t("Backward Output Gradients Bit Multiplications", 0, sys::AverageTotal);

        auto bw_act_work_reduction = stats.register_double_t("Backward Activations Work Reduction", 0, sys::Average);
        auto bw_act_speedup = stats.register_double_t("Backward Activations Speedup", 0, sys::Average);
        auto bw_act_par_mult = stats.register_double_t("Backward Activations Multiplications", 0, sys::AverageTotal);
        auto bw_act_bit_multiplications = stats.register_uint_t("Backward Activations Bit Multiplications", 0, sys::AverageTotal);

        auto act_out_grad_work_reduction = stats.register_double_t("Backward Output Gradients Work Reduction", 0, sys::Average);
        auto act_out_grad_speedup = stats.register_double_t("Backward Output Gradients Speedup", 0, sys::Average);
        auto act_out_grad_par_mult = stats.register_double_t("Backward Output Gradients Multiplications", 0, sys::AverageTotal);
        auto act_out_grad_bit_multiplications = stats.register_uint_t("Backward Output Gradients Bit Multiplications", 0, sys::AverageTotal);

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else traces_mode = 5;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, traces_mode);

            // Forward pass
            if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles simulation for epoch " << epoch << std::endl;

            for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                const base::Layer<float> &layer = network.getLayers()[layer_it];
                bool conv = layer.getType() == "Convolution";
                bool fc = layer.getType() == "InnerProduct";

                base::Array<T> act = layer.getActivations();
                act.powers_of_two_representation(layer.getActPrecision());
                if(fc && act.getDimensions() == 4) act.reshape_to_2D();

                base::Array<T> wgt = layer.getWeights();
                wgt.powers_of_two_representation(layer.getWgtPrecision());
                if(conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

                int padding = layer.getPadding();
                int stride = layer.getStride();

                if (conv) act.zero_pad(padding);

                const std::vector<size_t> &act_shape = act.getShape();
                const std::vector<size_t> &wgt_shape = wgt.getShape();

                auto batch_size = act_shape[0];
                auto act_channels = act_shape[1];
                auto Nx = act_shape[2];
                auto Ny = act_shape[3];

                auto num_filters = wgt_shape[0];
                auto wgt_channels = wgt_shape[1];
                auto Kx = wgt_shape[2];
                auto Ky = wgt_shape[3];

                long out_x = (Nx - Kx)/stride + 1;
                long out_y = (Ny - Ky)/stride + 1;

                auto groups = act_channels / wgt_channels;
                auto it_per_group = num_filters / groups;

                auto network_bits = network.getNetwork_bits();

                // Operations
                uint64_t fw_parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                                         num_filters * wgt_channels;

                // Forward convolution A * W
                for(int n = 0; n < batch_size; n++) {

                    if (simulate.only_backward)
                        continue;

                    double MAX_BITS = network_bits * network_bits;
                    uint64_t act_bit_counter = 0;
                    uint64_t wgt_bit_counter = 0;

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
                                                auto act_bits = act.get(n, start_group + k, stride * x + i,
                                                        stride * y + j);
                                                auto wgt_bits = wgt.get(m, k, i, j);
                                                act_bit_counter += computeDynamicTacticalBitsPE(act_bits, wgt_bits, true);
                                                wgt_bit_counter += computeDynamicTacticalBitsPE(act_bits, wgt_bits, false);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    } else {
                        for (int m = 0; m < num_filters; m++) {
                            for (int k = 0; k < wgt_channels; k++) {
                                auto act_bits = act.get(n, k);
                                auto wgt_bits = wgt.get(m, k);
                                act_bit_counter += computeDynamicTacticalBitsPE(act_bits, wgt_bits, true);
                                wgt_bit_counter += computeDynamicTacticalBitsPE(act_bits, wgt_bits, false);
                            }
                        }
                    }

                    act_bit_multiplications->value[layer_it][n] = act_bit_counter;
                    act_work_reduction->value[layer_it][n] = 100 - ((double)act_bit_counter / (double)fw_parallel_mult
                            / MAX_BITS * 100);
                    act_speedup->value[layer_it][n] = (double)fw_parallel_mult * MAX_BITS / (double)act_bit_counter;
                    act_par_mult->value[layer_it][n] = fw_parallel_mult;

                    wgt_bit_multiplications->value[layer_it][n] = wgt_bit_counter;
                    wgt_work_reduction->value[layer_it][n] = 100 - ((double)wgt_bit_counter / (double)fw_parallel_mult
                            / MAX_BITS * 100);
                    wgt_speedup->value[layer_it][n] = (double)fw_parallel_mult * MAX_BITS / (double)wgt_bit_counter;
                    wgt_par_mult->value[layer_it][n] = fw_parallel_mult;

                }

                if (simulate.only_forward)
                    continue;

                // Backward Convolution W * G

                // Backward Convolution A * G


            }




        }

        //Dump statistics
        std::string header = "DynamicTactical Potentials for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    template class DynamicTactical<float>;

}
