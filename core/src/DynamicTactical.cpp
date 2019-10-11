
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

        // Memory metadata
        address_map wgt_address_map;
        address_map wgt_grad_address_map;
        address_map act_address_map;

        // Simulate epochs
        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, 5);

            // Setup memory addresses
            if (epoch == 0) {

                // Weight and weight gradients addresses
                uint64_t wgt_next_addr = 0;
                uint64_t wgt_base_addr = 0x00000000;

                uint64_t wgt_grad_next_addr = 0;
                uint64_t wgt_grad_base_addr = 0x20000000;

                for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                    const base::Layer<float> &layer = network.getLayers()[layer_it];

                    base::Array<T> wgt = layer.getWeights();
                    if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

                    const std::vector<size_t> &wgt_shape = wgt.getShape();

                    auto num_filters = wgt_shape[0];
                    auto wgt_channels = wgt_shape[1];
                    auto Kx = wgt_shape[2];
                    auto Ky = wgt_shape[3];

                    // Weights addresses
                    wgt_address_map = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(num_filters,
                            std::vector<std::vector<std::vector<uint64_t>>>(Ky, std::vector<std::vector<uint64_t>>(Kx,
                            std::vector<uint64_t>(ceil(wgt_channels / 4.)))));

                    // Filter fourth
                    for (int m = 0; m < num_filters; ++m) {

                        // Column third
                        for (int y = 0; y < Ky; ++y) {

                            // Row second
                            for (int x = 0; x < Kx; ++x) {

                                // Store channel-first
                                for (int k = 0; k < wgt_channels; k += 4) {
                                    wgt_address_map[m][y][x][k/4] = wgt_base_addr + wgt_next_addr;
                                    wgt_next_addr += 0x40; // Align to 64 bits
                                }
                            }
                        }
                    }

                    // Weight gradients addresses
                    wgt_grad_address_map = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(num_filters,
                            std::vector<std::vector<std::vector<uint64_t>>>(Ky, std::vector<std::vector<uint64_t>>(Kx,
                            std::vector<uint64_t>(ceil(wgt_channels / 4.)))));

                    // Filter fourth
                    for (int m = 0; m < num_filters; ++m) {

                        // Column third
                        for (int y = 0; y < Ky; ++y) {

                            // Row second
                            for (int x = 0; x < Kx; ++x) {

                                // Store channel-first
                                for (int k = 0; k < wgt_channels; k += 4) {
                                    wgt_grad_address_map[m][y][x][k/4] = wgt_grad_base_addr + wgt_grad_next_addr;
                                    wgt_grad_next_addr += 0x40; // Align to 64 bits
                                }
                            }
                        }
                    }

                }

                // Activations addresses (Only first layer is read from off-chip)
                uint64_t act_next_addr = 0;
                uint64_t act_base_addr = 0x40000000;

                const base::Layer<float> &layer = network.getLayers()[0];
                bool fc = layer.getType() == "InnerProduct";

                base::Array<T> act = layer.getActivations();
                if(fc && act.getDimensions() == 4) act.reshape_to_2D();
                if(fc) act.reshape_to_4D();

                const std::vector<size_t> &act_shape = act.getShape();

                auto batch_size = act_shape[0];
                auto act_channels = act_shape[1];
                auto Nx = act_shape[2];
                auto Ny = act_shape[3];

                act_address_map = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(batch_size,
                        std::vector<std::vector<std::vector<uint64_t>>>(Ny, std::vector<std::vector<uint64_t>>(Nx,
                        std::vector<uint64_t>(ceil(act_channels / 4.)))));

                // Image fourth
                for (int n = 0; n < batch_size; ++n) {

                    // Column third
                    for (int y = 0; y < Ny; ++y) {

                        // Row second
                        for (int x = 0; x < Nx; ++x) {

                            // Store channel-first
                            for (int k = 0; k < act_channels; k += 4) {
                                act_address_map[n][y][x][k/4] = act_base_addr + act_next_addr;
                                act_next_addr += 0x40; // Align to 64 bits
                            }
                        }
                    }
                }

            }

            if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles simulation for epoch "
                                       << epoch << std::endl;

            auto num_batches = this->FAST_MODE ? 1 : network.getBatches();
            for (int batch = 0; batch < num_batches; ++batch) {

                // Forward pass
                for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                    if (simulate.only_backward)
                        continue;

                    const base::Layer<float> &layer = network.getLayers()[layer_it];
                    bool conv = layer.getType() == "Convolution";
                    bool fc = layer.getType() == "InnerProduct";

                    // Preprocess traces: Activations, Weights, Output Gradients
                    base::Array<T> act = layer.getActivations();
                    if(fc && act.getDimensions() == 4) act.reshape_to_2D();
                    if(fc) act.reshape_to_4D();
                    act.get_batch(batch);

                    base::Array<T> wgt = layer.getWeights();
                    if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

                    base::Array<T> out_grad = layer.getOutputGradients();
                    if(fc) out_grad.reshape_to_4D();

                    int padding = layer.getPadding();
                    int stride = layer.getStride();

                    const std::vector<size_t> &act_shape = act.getShape();
                    const std::vector<size_t> &wgt_shape = wgt.getShape();
                    const std::vector<size_t> &out_grad_shape = out_grad.getShape();

                    // Activations
                    auto act_channels = act_shape[1];
                    auto Nx = act_shape[2];
                    auto Ny = act_shape[3];

                    // Weights
                    auto num_filters = wgt_shape[0];
                    auto wgt_channels = wgt_shape[1];
                    auto Kx = wgt_shape[2];
                    auto Ky = wgt_shape[3];

                    // Output Gradients
                    auto Ox = out_grad_shape[2];
                    auto Oy = out_grad_shape[3];

                    bool asym_pad = false;
                    if (conv && padding > 0) {
                        asym_pad = ((Nx - Kx + 2 * padding)/(double)stride + 1) != Ox;
                    }

                    if (conv && padding > 0) asym_pad ? act.asym_zero_pad(padding) : act.zero_pad(padding);

                    const std::vector<size_t> &act_shape_pad = act.getShape();

                    auto Nx_pad = act_shape_pad[2];
                    auto Ny_pad = act_shape_pad[3];

                    // Simulate: Forward convolution A * W
                    auto sim_output_activations = std::vector<std::vector<std::vector<float>>>(num_filters,
                            std::vector<std::vector<float>>(Ox, std::vector<float>(Oy, 0)));


                    // Check correctness of the outputs
                    if (this->CHECK) {

                        auto output_activations = std::vector<std::vector<std::vector<float>>>(num_filters,
                                std::vector<std::vector<float>>(Ox, std::vector<float>(Oy, 0)));

                        // Actual convolution
                        for (int m = 0; m < num_filters; ++m) {

                            // Fix for MobileNet
                            int start_group = 0;
                            if(wgt_channels == 1 && act_channels != 1)
                                start_group = m;

                            // Number of Windows
                            for (int x = 0; x < Ox; ++x) {
                                for (int y = 0; y < Oy; ++y) {

                                    float sum = 0;

                                    // Window dimension
                                    for (int i = 0; i < Kx; ++i) {
                                        for (int j = 0; j < Ky; ++j) {
                                            for (int k = 0; k < wgt_channels; ++k) {
                                                sum += this->cast_bfloat16(act.get(0, start_group + k, stride * x + i,
                                                        stride * y + j)) * this->cast_bfloat16(wgt.get(m, k, i, j));
                                            }
                                        }
                                    }

                                    output_activations[m][x][y] = sum;
                                }
                            }
                        }

                        // Check values
                        /*for (int ch = 0; ch < num_filters; ++ch) {
                            for (int x = 0; x < out_x; ++x) {
                                for (int y = 0; y < out_y; ++y) {
                                    auto actual_value = output_activations[ch][x][y];
                                    auto sim_value = sim_output_activations[ch][x][y];
                                    if (actual_value != sim_value)
                                        throw std::runtime_error("Forward convolution wrong value.");
                                }
                            }
                        }*/

                    } // Check results

                } // Forward pass

                // Backward pass
                for (int layer_it = network.getNumLayers() - 1; layer_it >= 0; layer_it--) {

                    if (simulate.only_forward)
                        continue;

                    const base::Layer<float> &layer = network.getLayers()[layer_it];
                    bool conv = layer.getType() == "Convolution";
                    bool fc = layer.getType() == "InnerProduct";

                    // Preprocess traces: Activations, Weights, Output Gradients
                    base::Array<T> act = layer.getActivations();
                    if (fc && act.getDimensions() == 4) act.reshape_to_2D();
                    if (fc) act.reshape_to_4D();
                    act.get_batch(batch);

                    base::Array<T> wgt = layer.getWeights();
                    if (wgt.getDimensions() == 2) wgt.reshape_to_4D();

                    base::Array<T> out_grad = layer.getOutputGradients();
                    if (fc) out_grad.reshape_to_4D();
                    out_grad.get_batch(batch);

                    int padding = layer.getPadding();
                    int stride = layer.getStride();

                    const std::vector<size_t> &act_shape = act.getShape();
                    const std::vector<size_t> &wgt_shape = wgt.getShape();
                    const std::vector<size_t> &out_grad_shape = out_grad.getShape();

                    // Activations
                    auto act_channels = act_shape[1];
                    auto Nx = act_shape[2];
                    auto Ny = act_shape[3];

                    // Weights
                    auto num_filters = wgt_shape[0];
                    auto wgt_channels = wgt_shape[1];
                    auto Kx = wgt_shape[2];
                    auto Ky = wgt_shape[3];

                    // Output Gradients
                    auto out_channels = out_grad_shape[1];
                    auto Ox = out_grad_shape[2];
                    auto Oy = out_grad_shape[3];

                    bool asym_pad = false;
                    if (conv && padding > 0) {
                        asym_pad = ((Nx - Kx + 2 * padding) / (double) stride + 1) != Ox;
                    }

                    if (conv && padding > 0) asym_pad ? act.asym_zero_pad(padding) : act.zero_pad(padding);

                    const std::vector<size_t> &act_shape_pad = act.getShape();

                    auto Nx_pad = act_shape_pad[2];
                    auto Ny_pad = act_shape_pad[3];

                    // Backward pass - Calculate Weight gradients
                    if (conv) out_grad.dilate_out_grad(stride, Nx_pad, Kx);

                    const std::vector<size_t> &out_grad_shape_dil = out_grad.getShape();

                    auto Ox_dil = out_grad_shape_dil[2];
                    auto Oy_dil = out_grad_shape_dil[3];

                    // Check window size
                    if ((Nx_pad - Ox_dil + 1) != Kx)
                        throw std::runtime_error("Weight gradients incorrect window sizes");

                    // Simulate: Backward convolution A * G = WG
                    auto sim_weight_gradients = std::vector<std::vector<std::vector<std::vector<float>>>>(num_filters,
                            std::vector<std::vector<std::vector<float>>>(wgt_channels,
                            std::vector<std::vector<float>>(Kx, std::vector<float>(Ky, 0))));

                    // Check correctness of the outputs
                    if (this->CHECK) {

                        auto weight_gradients = std::vector<std::vector<std::vector<std::vector<float>>>>(num_filters,
                                std::vector<std::vector<std::vector<float>>>(wgt_channels,
                                std::vector<std::vector<float>>(Kx, std::vector<float>(Ky, 0))));

                        for (int o = 0; o < out_channels; ++o) {
                            for (int k = 0; k < act_channels; ++k) {

                                // Number of Windows
                                for (int x = 0; x < Kx; ++x) {
                                    for (int y = 0; y < Ky; ++y) {

                                        float sum = 0;

                                        // Window dimensions
                                        for (int i = 0; i < Ox_dil; ++i) {
                                            for (int j = 0; j < Oy_dil; ++j) {
                                                sum += this->cast_bfloat16(out_grad.get(0, o, i, j)) *
                                                        this->cast_bfloat16(act.get(0, k, x + i, y + j));
                                            }
                                        }

                                        weight_gradients[o][k][x][y] += sum;

                                    }
                                }
                            }

                        }

                        // Check values
                        /*for (int m = 0; m < num_filters; ++m) {
                            for (int ch = 0; ch < wgt_channels; ++ch) {
                                for (int x = 0; x < Kx; +false+x) {
                                    for (int y = 0; y < Ky; ++y) {
                                        auto actual_value = weight_gradients[m][ch][x][y];
                                        auto sim_value = sim_weight_gradients[m][ch][x][y];
                                        if (actual_value != sim_value)
                                            throw std::runtime_error("Backward weight gradients convolution wrong value.");
                                    }
                                }
                            }
                        }*/

                    } // Check results

                    // Backward pass - Calculate Input gradients
                    if (layer_it == 0)
                        continue;

                    if (conv && padding > 0)
                        asym_pad ? out_grad.asym_zero_pad(padding + stride - 1) :
                        out_grad.zero_pad(padding + stride - 1);

                    const std::vector<size_t> &out_grad_shape_pad = out_grad.getShape();

                    auto Ox_pad = out_grad_shape_pad[2];
                    auto Oy_pad = out_grad_shape_pad[3];

                    wgt.rotate_180deg();
                    wgt.reshape_channel_wise(out_channels);

                    const std::vector<size_t> &wgt_shape_rot = wgt.getShape();

                    auto num_filters_rot = wgt_shape[0];
                    auto wgt_channels_rot = wgt_shape[1];

                    // Check window size
                    if (wgt_channels_rot != out_channels)
                        throw std::runtime_error("Wrong weights rotated channels");

                    // Check window size
                    if ((Ox_pad - Kx + 1) != Nx)
                        throw std::runtime_error("Input gradients incorrect window sizes");

                    // Simulate: Backward convolution W * G = IG
                    auto sim_input_gradients = std::vector<std::vector<std::vector<float>>>(act_channels,
                            std::vector<std::vector<float>>(Nx, std::vector<float>(Ny, 0)));

                    // Check correctness of the outputs
                    if (this->CHECK) {

                        auto input_gradients = std::vector<std::vector<std::vector<float>>>(act_channels,
                                std::vector<std::vector<float>>(Nx, std::vector<float>(Ny, 0)));

                        // Actual convolution
                        for (int m = 0; m < num_filters_rot; ++m) {

                            // Number of Windows
                            for (int x = 0; x < Nx; ++x) {
                                for (int y = 0; y < Ny; ++y) {

                                    float sum = 0;

                                    // Windows dimension
                                    for (int i = 0; i < Kx; ++i) {
                                        for (int j = 0; j < Ky; ++j) {
                                            for (int k = 0; k < wgt_channels_rot; ++k) {
                                                sum += this->cast_bfloat16(out_grad.get(0, k, x + i, y + j)) *
                                                        this->cast_bfloat16(wgt.get(m, k, i, j));
                                            }
                                        }
                                    }

                                    input_gradients[m][x][y] = sum;
                                }
                            }
                        }

                        // Check values
                        /*for (int ch = 0; ch < act_channels; ++ch) {
                            for (int x = 0; x < Nx; ++x) {
                                for (int y = 0; y < Ny; ++y) {
                                    auto actual_value = input_gradients[ch][x][y];
                                    auto sim_value = sim_oinput_gradients[ch][x][y];
                                    if (actual_value != sim_value)
                                        throw std::runtime_error("Backward input gradients convolution wrong value.");
                                }
                            }
                        }*/

                    } // Check results

                } // Backward pass

            } // Batch

        } // Epoch

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

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, 5);

            if(!this->QUIET) std::cout << "Starting Dynamic Tactical potentials simulation for epoch "
                                       << epoch << std::endl;

            for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                const base::Layer<float> &layer = network.getLayers()[layer_it];
                bool conv = layer.getType() == "Convolution";
                bool fc = layer.getType() == "InnerProduct";

                // Preprocess traces: Activations, Weights, Output Gradients
                base::Array<T> act = layer.getActivations();
                if(fc && act.getDimensions() == 4) act.reshape_to_2D();
                if(fc) act.reshape_to_4D();

                base::Array<T> wgt = layer.getWeights();
                if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

                base::Array<T> out_grad = layer.getOutputGradients();
                if(fc) out_grad.reshape_to_4D();

                int padding = layer.getPadding();
                int stride = layer.getStride();

                const std::vector<size_t> &act_shape = act.getShape();
                const std::vector<size_t> &wgt_shape = wgt.getShape();
                const std::vector<size_t> &out_grad_shape = out_grad.getShape();

                // Activations
                auto batch_size = act_shape[0];
                auto act_channels = act_shape[1];
                auto Nx = act_shape[2];
                auto Ny = act_shape[3];
                if(this->FAST_MODE) batch_size = 1;

                // Weights
                auto num_filters = wgt_shape[0];
                auto wgt_channels = wgt_shape[1];
                auto Kx = wgt_shape[2];
                auto Ky = wgt_shape[3];

                // Output Gradients
                auto out_batch_size = out_grad_shape[0];
                auto out_channels = out_grad_shape[1];
                auto Ox = out_grad_shape[2];
                auto Oy = out_grad_shape[3];
                if(this->FAST_MODE) out_batch_size = 1;

                bool asym_pad = false;
                if (conv && padding > 0) {
                    asym_pad = ((Nx - Kx + 2 * padding)/(double)stride + 1) != Ox;
                }

                if (conv && padding > 0) asym_pad ? act.asym_zero_pad(padding) : act.zero_pad(padding);

                const std::vector<size_t> &act_shape_pad = act.getShape();

                auto Nx_pad = act_shape_pad[2];
                auto Ny_pad = act_shape_pad[3];

                // Forward pass

                auto network_bits = network.getNetwork_bits();

                // Operations
                uint64_t fw_parallel_mult = num_filters * Ox * Oy * Kx * Ky * wgt_channels;

                // Forward convolution A * W
                for (int n = 0; n < batch_size; ++n) {

                    if (simulate.only_backward)
                        continue;

                    double MAX_BITS = network_bits * network_bits;
                    uint64_t act_bit_counter = 0;
                    uint64_t wgt_bit_counter = 0;

                    // Actual convolution
                    for (int m = 0; m < num_filters; ++m) {

                        // Fix for MobileNet
                        int start_group = 0;
                        if(wgt_channels == 1 && act_channels != 1)
                            start_group = m;

                        // Number of Windows
                        for (int x = 0; x < Ox; ++x) {
                            for (int y = 0; y < Oy; ++y) {

                                // Window dimension
                                for (int i = 0; i < Kx; ++i) {
                                    for (int j = 0; j < Ky; ++j) {
                                        for (int k = 0; k < wgt_channels; ++k) {
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

                // Backward pass - Calculate Weight gradients
                if (conv) out_grad.dilate_out_grad(stride, Nx_pad, Kx);

                const std::vector<size_t> &out_grad_shape_dil = out_grad.getShape();

                auto Ox_dil = out_grad_shape_dil[2];
                auto Oy_dil = out_grad_shape_dil[3];

                // Check window size
                if ((Nx_pad - Ox_dil + 1) != Kx)
                    throw std::runtime_error("Weight gradients incorrect window sizes");

                // Operations
                uint64_t bw_act_parallel_mult = out_channels * act_channels * Kx * Ky * Ox_dil * Oy_dil;

                // Backward convolution A * G
                for (int n = 0; n < out_batch_size; ++n) {

                    double MAX_BITS = network_bits * network_bits;
                    uint64_t act_bit_counter = 0;
                    uint64_t out_grad_bit_counter = 0;

                    for (int o = 0; o < out_channels; ++o) {
                        for (int k = 0; k < act_channels; ++k) {

                            // Number of Windows
                            for (int x = 0; x < Kx; ++x) {
                                for (int y = 0; y < Ky; ++y) {

                                    // Window dimensions
                                    for (int i = 0; i < Ox_dil; ++i) {
                                        for (int j = 0; j < Oy_dil; ++j) {
                                            auto act_bits = act.get(n, k, x + i, y + j);
                                            auto out_grad_bits = out_grad.get(n, o, i, j);
                                            act_bit_counter += computeDynamicTacticalBitsPE(act_bits, out_grad_bits, true);
                                            out_grad_bit_counter += computeDynamicTacticalBitsPE(act_bits, out_grad_bits, false);
                                        }
                                    }

                                }
                            }
                        }

                    }

                    bw_act_bit_multiplications->value[layer_it][n] = act_bit_counter;
                    bw_act_work_reduction->value[layer_it][n] = 100 - ((double)act_bit_counter /
                            (double)bw_act_parallel_mult / MAX_BITS * 100);
                    bw_act_speedup->value[layer_it][n] = (double)bw_act_parallel_mult * MAX_BITS /
                            (double)act_bit_counter;
                    bw_act_par_mult->value[layer_it][n] = bw_act_parallel_mult;

                    act_out_grad_bit_multiplications->value[layer_it][n] = out_grad_bit_counter;
                    act_out_grad_work_reduction->value[layer_it][n] = 100 - ((double)out_grad_bit_counter /
                            (double)bw_act_parallel_mult / MAX_BITS * 100);
                    act_out_grad_speedup->value[layer_it][n] = (double)bw_act_parallel_mult * MAX_BITS /
                            (double)out_grad_bit_counter;
                    act_out_grad_par_mult->value[layer_it][n] = bw_act_parallel_mult;

                }

                // Backward pass - Calculate Input gradients
                if (conv && padding > 0) asym_pad ? out_grad.asym_zero_pad(padding + stride - 1) :
                        out_grad.zero_pad(padding + stride - 1);

                const std::vector<size_t> &out_grad_shape_pad = out_grad.getShape();

                auto Ox_pad = out_grad_shape_pad[2];
                auto Oy_pad = out_grad_shape_pad[3];

                wgt.rotate_180deg();
                wgt.reshape_channel_wise(out_channels);

                const std::vector<size_t> &wgt_shape_rot = wgt.getShape();

                auto num_filters_rot = wgt_shape[0];
                auto wgt_channels_rot = wgt_shape[1];

                // Check window size
                if (wgt_channels_rot != out_channels)
                    throw std::runtime_error("Wrong weights rotated channels");

                // Check window size
                if ((Ox_pad - Kx + 1) != Nx)
                    throw std::runtime_error("Input gradients incorrect window sizes");

                // Operations
                uint64_t bw_wgt_parallel_mult = num_filters * Nx * Ny * Kx * Ky * wgt_channels;

                // Backward convolution W * G
                for (int n = 0; n < out_batch_size; ++n) {

                    double MAX_BITS = network_bits * network_bits;
                    uint64_t wgt_bit_counter = 0;
                    uint64_t out_grad_bit_counter = 0;

                    for (int m = 0; m < num_filters_rot; ++m) {

                        // Number of Windows
                        for (int x = 0; x < Nx; ++x) {
                            for (int y = 0; y < Ny; ++y) {

                                // Windows dimension
                                for (int i = 0; i < Kx; ++i) {
                                    for (int j = 0; j < Ky; ++j) {
                                        for (int k = 0; k < wgt_channels_rot; ++k) {
                                            auto wgt_bits = wgt.get(m, k, i, j);
                                            auto out_grad_bits = out_grad.get(n, k, x + i, y + j);
                                            wgt_bit_counter += computeDynamicTacticalBitsPE(wgt_bits, out_grad_bits, true);
                                            out_grad_bit_counter += computeDynamicTacticalBitsPE(wgt_bits, out_grad_bits, false);
                                        }
                                    }
                                }

                            }
                        }
                    }

                    bw_wgt_bit_multiplications->value[layer_it][n] = wgt_bit_counter;
                    bw_wgt_work_reduction->value[layer_it][n] = 100 - ((double)wgt_bit_counter /
                            (double)bw_wgt_parallel_mult / MAX_BITS * 100);
                    bw_wgt_speedup->value[layer_it][n] = (double)fw_parallel_mult * MAX_BITS /
                            (double)bw_wgt_parallel_mult;
                    bw_wgt_par_mult->value[layer_it][n] = bw_wgt_parallel_mult;

                    wgt_out_grad_bit_multiplications->value[layer_it][n] = out_grad_bit_counter;
                    wgt_out_grad_work_reduction->value[layer_it][n] = 100 - ((double)out_grad_bit_counter /
                            (double)bw_wgt_parallel_mult / MAX_BITS * 100);
                    wgt_out_grad_speedup->value[layer_it][n] = (double)bw_wgt_parallel_mult * MAX_BITS /
                            (double)out_grad_bit_counter;
                    wgt_out_grad_par_mult->value[layer_it][n] = bw_wgt_parallel_mult;

                }

            }

        }


        //Dump statistics
        std::string header = "DynamicTactical Potentials for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    template class DynamicTactical<float>;

}
