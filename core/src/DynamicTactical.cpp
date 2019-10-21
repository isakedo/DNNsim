
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

    bool check_zero_line(const std::vector<value_mux> &schedule) {
        for(auto tuple : schedule) {
            auto value = std::get<0>(tuple);
            if(value != 0) return false;
        }
        return true;
    }

    template <typename T>
    void DynamicTactical<T>::promote(schedule_buffer &schedule, value_index ineffectual, value_index candidate) {

        // Ineffectual
        auto inef_time = std::get<0>(ineffectual);
        auto inef_lane = std::get<1>(ineffectual);

        // Candidate
        auto cand_time = std::get<0>(candidate);
        auto cand_lane = std::get<1>(candidate);

        // Swap
        auto ineffectual_tuple = schedule[inef_time][inef_lane];
        schedule[inef_time][inef_lane] = schedule[cand_time][cand_lane];
        schedule[cand_time][cand_lane] = ineffectual_tuple;
    }

    template <typename T>
    std::vector<value_index> DynamicTactical<T>::search(const schedule_buffer &schedule, value_index value_idx,
            int max_time) {

        auto time = std::get<0>(value_idx);
        auto lane = std::get<1>(value_idx);
        int upper_bound = (lane / N_LANES) * N_LANES;
        int lower_bound = ((lane / N_LANES) + 1) * N_LANES;
        std::vector<value_index> effectual_candidates;

        auto next_time = time + 1;
        if(next_time >= max_time) return effectual_candidates;

        // Search effectual values in search space
        for (int s = 0; s < SEARCH_MAP.size(); ++s) {
            auto search_space = SEARCH_MAP[s];
            auto time_h = time + std::get<0>(search_space);
            auto lane_d = lane + std::get<1>(search_space);
            if(time_h >= max_time) continue;
            lane_d = (lane_d) < upper_bound ? N_LANES + lane_d : lane_d; // Wrap around
            lane_d = (lane_d) >= lower_bound ? lane_d - N_LANES : lane_d; // Wrap around
            auto value_tuple = schedule[time_h][lane_d];
            auto value_bits = std::get<0>(value_tuple);
            if(value_bits != 0) effectual_candidates.push_back(std::make_tuple(time_h, lane_d));
        }

        return effectual_candidates;
    }

    template <typename T>
    void DynamicTactical<T>::original_schedule(schedule_buffer &schedule) {

        auto max_time = schedule.size();
        auto groups = schedule.front().size() / N_LANES;
        auto tmp_schedule = schedule;
        schedule.clear();

        int skip = 0;
        for (int time = 0; time < max_time; ++time) {

            // Skip lines of zeroes
            if (skip < LOOKAHEAD_H && check_zero_line(tmp_schedule[time])) {
                skip++;
                continue;
            }
            skip = 0;

            for (int group = 0; group < groups; ++group) {

                int overlap = 1;
                while(overlap > 0) {

                    // Get ineffectual values
                    int init_lane = group * N_LANES;
                    std::vector<value_index> ineffectual_values;
                    for(int lane = init_lane; lane < init_lane + N_LANES; lane++) {
                        auto value_tuple = tmp_schedule[time][lane];
                        auto value_bits = std::get<0>(value_tuple);
                        if(value_bits == 0) ineffectual_values.emplace_back(std::make_tuple(time, lane));
                    }

                    // Num of candidates for each ineffectual values
                    overlap = -1;
                    std::vector<uint16_t> num_candidates (N_LANES, 0);
                    std::vector<std::vector<value_index>> effectual_candidates (N_LANES, std::vector<value_index>());
                    for(auto inef_idx : ineffectual_values) {
                        auto lane = std::get<1>(inef_idx);
                        effectual_candidates[lane % N_LANES] = search(tmp_schedule, inef_idx, max_time);
                        if(!effectual_candidates[lane % N_LANES].empty()) {
                            auto effectual_num_candidates = (uint16_t)effectual_candidates[lane % N_LANES].size();
                            num_candidates[lane % N_LANES] = effectual_num_candidates;
                            if (effectual_num_candidates > overlap) overlap = effectual_num_candidates;
                        }
                    }

                    // Promote less flexible candidates first
                    for(auto inef_idx : ineffectual_values) {
                        auto lane = std::get<1>(inef_idx);
                        if(num_candidates[lane % N_LANES] == overlap) {
                            //Promote value
                            auto cand_idx = effectual_candidates[lane % N_LANES].front();
                            promote(tmp_schedule, inef_idx, cand_idx);
                            break;
                        }
                    }

                } // Optimal promotion loop

            } // Group
            schedule.emplace_back(tmp_schedule[time]);
        } // Time

    }

    /* CYCLES */

    template <typename T>
    void DynamicTactical<T>::run(const sys::Batch::Simulate &simulate, int epochs) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, this->QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string arch = "DynamicTactical";
        int mux_entries = LOOKAHEAD_H + LOOKASIDE_D + 1;
        std::string filename = "DynamicTactical_L" + std::to_string(this->N_LANES) + "_C" +
                std::to_string(this->N_COLUMNS) + "_R" + std::to_string(this->N_ROWS) + "_T" +
                std::to_string(this->N_TILES) + "_" + this->SEARCH_SHAPE + std::to_string(mux_entries) + "("
                + std::to_string(this->LOOKAHEAD_H) + "-" + std::to_string(this->LOOKASIDE_D) + ")" + "_cycles";
        sys::Stats stats = sys::Stats(epochs, network_model.getNumLayers(), filename);
        stats.setTraining(true);

        // Forward stats
        auto fw_compute_cycles = stats.register_uint_t("Forward compute cycles", 0, sys::Total);
        auto fw_base_compute_cycles = stats.register_uint_t("Forward base compute cycles", 0, sys::Total);
        auto fw_speedup = stats.register_double_t("Forward speedup", 0, sys::Special);
        auto fw_ideal_compute_cycles = stats.register_uint_t("Forward ideal compute cycles", 0, sys::Total);
        auto fw_exploited_sparsity = stats.register_double_t("Forward exploited sparsity", 0, sys::Special);

        // Backward stats
        auto bw_wgt_act_compute_cycles = stats.register_uint_t("Backward Weights A.S compute cycles", 0, sys::Total);
        auto bw_wgt_act_base_compute_cycles = stats.register_uint_t("Backward Weights A.S base compute cycles", 0, sys::Total);
        auto bw_wgt_act_speedup = stats.register_double_t("Backward Weights A.S speedup", 0, sys::Special);
        auto bw_wgt_act_ideal_compute_cycles = stats.register_uint_t("Backward Weights A.S ideal compute cycles", 0, sys::Total);
        auto bw_wgt_act_exploited_sparsity = stats.register_double_t("Backward Weights A.S exploited sparsity", 0, sys::Special);

        auto bw_wgt_out_compute_cycles = stats.register_uint_t("Backward Weights G.S compute cycles", 0, sys::Total);
        auto bw_wgt_out_base_compute_cycles = stats.register_uint_t("Backward Weights G.S base compute cycles", 0, sys::Total);
        auto bw_wgt_out_speedup = stats.register_double_t("Backward Weights G.S speedup", 0, sys::Special);
        auto bw_wgt_out_ideal_compute_cycles = stats.register_uint_t("Backward Weights G.S ideal compute cycles", 0, sys::Total);
        auto bw_wgt_out_exploited_sparsity = stats.register_double_t("Backward Weights G.S exploited sparsity", 0, sys::Special);

        auto bw_wgt_compute_cycles = stats.register_uint_t("Backward Weights compute cycles", 0, sys::Total);
        auto bw_wgt_speedup = stats.register_double_t("Backward Weights speedup", 0, sys::Special);

        auto bw_in_compute_cycles = stats.register_uint_t("Backward Input compute cycles", 0, sys::Total);
        auto bw_in_base_compute_cycles = stats.register_uint_t("Backward Input base compute cycles", 0, sys::Total);
        auto bw_in_speedup = stats.register_double_t("Backward Input speedup", 0, sys::Special);
        auto bw_in_ideal_compute_cycles = stats.register_uint_t("Backward Input ideal compute cycles", 0, sys::Total);
        auto bw_in_exploited_sparsity = stats.register_double_t("Backward Input exploited sparsity", 0, sys::Special);

        // Simulate epochs
        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, 5);

            if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles simulation for epoch "
                                       << epoch << std::endl;

            auto num_batches = this->FAST_MODE ? 1 : network.getBatches();
            int batch;

            auto max_threads = omp_get_max_threads();
            omp_set_num_threads(std::min(max_threads, this->N_THREADS));
            #pragma omp parallel for private(batch)
            for (batch = 0; batch < num_batches; ++batch) {

                // Forward pass
                for (int layer_it = 0; layer_it < network.getNumLayers(); layer_it++) {

                    if (simulate.only_backward)
                        continue;

                    const base::Layer<float> &layer = network.getLayers()[layer_it];
                    bool conv = layer.getType() == "Convolution";
                    bool fc = layer.getType() == "InnerProduct";

                    printf("Forward: Epoch: %d, Batch %d, Layer %s\n", epoch, batch, layer.getName().c_str());

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

                    if (conv && padding > 0) {
                        if (Kx < Ky) act.zero_pad_y(padding);
                        else if (Ky < Kx) act.zero_pad_x(padding);
                        else if (((Nx - Kx + 2 * padding)/(double)stride + 1) != Ox) act.asym_zero_pad(padding);
                        else act.zero_pad(padding);
                    }

                    const std::vector<size_t> &act_shape_pad = act.getShape();

                    auto Nx_pad = act_shape_pad[2];
                    auto Ny_pad = act_shape_pad[3];

                    // Check window size
                    if ((Nx_pad - Kx)/stride + 1 != Ox)
                        throw std::runtime_error("Output activations incorrect X window sizes");
                    if ((Ny_pad - Ky)/stride + 1 != Oy)
                        throw std::runtime_error("Output activations incorrect Y window sizes");

                    // Simulate: Forward convolution A * W
                    auto sim_output_activations = std::vector<std::vector<std::vector<double>>>(num_filters,
                            std::vector<std::vector<double>>(Ox, std::vector<double>(Oy, 0)));

                    // Stats
                    uint64_t batch_compute_cycles = 0;
                    uint64_t batch_base_compute_cycles = 0;
                    uint64_t batch_ideal_compute_cycles = 0;

                    // Generate weight buffer
                    auto num_filter_sets = (uint64_t)ceil(num_filters / (double)N_COLUMNS);

                    auto round_wgt_channels = (int)ceil(wgt_channels / (double)N_LANES) * N_LANES;
                    auto time_per_filter = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)N_LANES);

                    non_schedule_buffer weight_buffer = non_schedule_buffer(num_filter_sets,
                            std::vector<std::vector<float>>(time_per_filter,
                            std::vector<float>(N_COLUMNS * N_LANES, 0.0f)));

                    int set_wgt = -1;
                    for(int m = 0; m < num_filters; m++) {

                        if ((m % N_COLUMNS) == 0)
                            set_wgt++;

                        int time = 0;
                        for (int y = 0; y < Ky; ++y) {
                            for (int x = 0; x < Kx; ++x) {
                                for (int k = 0; k < wgt_channels; k += N_LANES) {
                                    int index = 0;
                                    for(int channel = k; channel < std::min((uint64_t)k + N_LANES,
                                            wgt_channels); ++channel) {
                                        auto wgt_bits = wgt.get(m, channel, x, y);
                                        int pos = (m % N_COLUMNS) * N_LANES + index;
                                        weight_buffer[set_wgt][time][pos] = wgt_bits;
                                        index++;
                                        if(index == N_LANES) {
                                            time++;
                                            index = 0;
                                        }
                                    }
                                    if(index != 0)
                                        time++;
                                }
                            }
                        }

                    }

                    std::vector<int> x_windows, y_windows;
                    int x_counter = 0, y_counter = 0;
                    while(this->iterateWindows(Ox, Oy, x_windows, y_windows, x_counter, y_counter, N_ROWS)) {

                        // Generate activation buffer
                        auto round_act_channels = (int)ceil(act_channels / (double)N_LANES) * N_LANES;
                        auto time_per_window = (uint64_t)ceil(round_act_channels * Kx * Ky / (double)N_LANES);

                        schedule_buffer activation_buffer = schedule_buffer(time_per_window,
                                std::vector<value_mux>(x_windows.size() * N_LANES, std::make_tuple(0.0f, 0, 0)));

                        uint64_t ideal_time_per_window = 0;

                        for (int w = 0; w < x_windows.size(); ++w) {
                            auto x_window = x_windows[w] * stride;
                            auto y_window = y_windows[w] * stride;

                            uint64_t non_zeroes = 0;

                            int time = 0;
                            for (int y = 0; y < Ky; ++y) {
                                for (int x = 0; x < Kx; ++x) {
                                    for (int k = 0; k < act_channels; k += N_LANES) {
                                        int index = 0;
                                        for (int channel = k; channel < std::min((uint64_t)k + N_LANES,
                                                act_channels); ++channel) {
                                            auto act_bits = act.get(0, channel, x_window + x, y_window + y);
                                            int pos = w * N_LANES + index;
                                            activation_buffer[time][pos] = std::make_tuple(act_bits, time, index);
                                            index++;
                                            if(index == N_LANES) {
                                                time++;
                                                index = 0;
                                            }
                                            if (act_bits != 0) non_zeroes++;
                                        }
                                        if (index != 0)
                                            time++;
                                    }
                                }
                            }
                            auto ideal_time = (uint64_t)ceil(non_zeroes / (double)N_LANES);
                            if (ideal_time > ideal_time_per_window)
                                ideal_time_per_window = ideal_time;

                        }

                        // Schedule buffer
                        original_schedule(activation_buffer);

                        for (int set = 0; set < num_filter_sets; ++set) {

                            batch_compute_cycles += activation_buffer.size();
                            batch_base_compute_cycles += time_per_window;
                            batch_ideal_compute_cycles += ideal_time_per_window;

                            if (this->CHECK) {
                                for (const auto &time_buffer : activation_buffer) {

                                    for (int w = 0; w < x_windows.size(); ++w) {
                                        auto window_idx = w * N_LANES;
                                        auto x_window = x_windows[w];
                                        auto y_window = y_windows[w];

                                        for (int f = 0; f < N_COLUMNS; ++f) {
                                            auto filter_idx = f * N_LANES;
                                            auto filter = set * N_COLUMNS + f;

                                            if (filter >= num_filters)
                                                continue;

                                            for (int lane = 0; lane < N_LANES; ++lane) {

                                                auto act_bits = std::get<0>(time_buffer[window_idx + lane]);
                                                auto time_h = std::get<1>(time_buffer[window_idx + lane]);
                                                auto lane_d = std::get<2>(time_buffer[window_idx + lane]);

                                                auto wgt_bits = weight_buffer[set][time_h][filter_idx + lane_d];

                                                sim_output_activations[filter][x_window][y_window] += act_bits * wgt_bits;

                                            } // Multiply 16 weights and 16 activations

                                        } // Filter
                                    } // Window
                                } //Time of the buffer
                            } // Check

                        } // Filter sets

                    } // Window sets

                    #pragma omp critical
                    {
                        fw_compute_cycles->value[epoch][layer_it] += batch_compute_cycles;
                        fw_base_compute_cycles->value[epoch][layer_it] += batch_base_compute_cycles;
                        fw_ideal_compute_cycles->value[epoch][layer_it] += batch_ideal_compute_cycles;
                    }

                    // Check correctness of the outputs
                    if (this->CHECK) {

                        auto output_activations = std::vector<std::vector<std::vector<double>>>(num_filters,
                                std::vector<std::vector<double>>(Ox, std::vector<double>(Oy, 0)));

                        // Actual convolution
                        for (int m = 0; m < num_filters; ++m) {

                            // Fix for MobileNet
                            int start_group = 0;
                            if(wgt_channels == 1 && act_channels != 1)
                                start_group = m;

                            // Number of Windows
                            for (int x = 0; x < Ox; ++x) {
                                for (int y = 0; y < Oy; ++y) {

                                    double sum = 0;

                                    // Window dimension
                                    for (int j = 0; j < Ky; ++j) {
                                        for (int i = 0; i < Kx; ++i) {
                                            for (int k = 0; k < wgt_channels; ++k) {
                                                sum += act.get(0, start_group + k, stride * x + i,
                                                        stride * y + j) * wgt.get(m, k, i, j);
                                            }
                                        }
                                    }

                                    output_activations[m][x][y] = sum;
                                }
                            }
                        }

                        // Check values
                        for (int ch = 0; ch < num_filters; ++ch) {
                            for (int x = 0; x < Ox; ++x) {
                                for (int y = 0; y < Oy; ++y) {
                                    auto actual_value = output_activations[ch][x][y];
                                    auto sim_value = sim_output_activations[ch][x][y];
                                    auto error = (actual_value - sim_value) / sim_value;
                                    if (abs(error) > 1e-10)
                                        throw std::runtime_error("Forward convolution wrong value.");
                                }
                            }
                        }

                    } // Check results

                } // Forward pass

                // Backward pass
                for (int layer_it = 75; layer_it >= 0; layer_it--) {

                    if (simulate.only_forward)
                        continue;

                    const base::Layer<float> &layer = network.getLayers()[layer_it];
                    bool conv = layer.getType() == "Convolution";
                    bool fc = layer.getType() == "InnerProduct";

                    printf("Backward: Epoch: %d, Batch %d, Layer %s\n", epoch, batch, layer.getName().c_str());

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

                    int pad_type = -1;
                    if (conv && padding > 0) {
                        if (Kx < Ky) pad_type = 1;
                        else if (Ky < Kx) pad_type = 2;
                        else if (((Nx - Kx + 2 * padding)/(double)stride + 1) != Ox) pad_type = 3;
                        else pad_type = 4;
                    }

                    if (pad_type == 1) act.zero_pad_y(padding);
                    else if (pad_type == 2) act.zero_pad_x(padding);
                    else if (pad_type == 3) act.asym_zero_pad(padding);
                    else if (pad_type == 4) act.zero_pad(padding);

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
                        throw std::runtime_error("Weight gradients incorrect X window sizes");
                    if ((Ny_pad - Oy_dil + 1) != Ky)
                        throw std::runtime_error("Weight gradients incorrect Y window sizes");

                    // Simulate: Backward convolution A * G = WG: Activations sparsity
                    auto sim_weight_act_gradients = std::vector<std::vector<std::vector<std::vector<double>>>>(
                            num_filters, std::vector<std::vector<std::vector<double>>>(wgt_channels,
                            std::vector<std::vector<double>>(Kx, std::vector<double>(Ky, 0))));

                    // Stats
                    uint64_t batch_wgt_act_compute_cycles = 0;
                    uint64_t batch_wgt_act_base_compute_cycles = 0;
                    uint64_t batch_wgt_act_ideal_compute_cycles = 0;

                    // Generate output gradients buffer
                    auto spatial_pad = (uint64_t)ceil(Ox_dil * Ox_dil/ (double)N_LANES) * N_LANES;

                    auto num_out_grad_sets = (uint64_t)ceil(out_channels / (double)N_COLUMNS);
                    auto time_per_out_grad_channel = (uint64_t)ceil(spatial_pad / (double)N_LANES);

                    non_schedule_buffer out_gradients_buffer = non_schedule_buffer(num_out_grad_sets,
                            std::vector<std::vector<float>>(time_per_out_grad_channel,
                            std::vector<float>(N_COLUMNS * N_LANES, 0.0f)));

                    std::vector<uint64_t> ideal_time_per_out_grad_channel (ceil(out_channels/(double)N_COLUMNS), 0);

                    int set_out = -1;
                    for(int o = 0; o < out_channels; ++o) {

                        if ((o % N_COLUMNS) == 0)
                            set_out++;

                        uint64_t non_zeroes = 0;

                        int index = 0;
                        int time = 0;
                        for (int y = 0; y < Oy_dil; ++y) {
                            for (int x = 0; x < Ox_dil; ++x) {
                                auto out_bits = out_grad.get(0, o, x, y);
                                int pos = (o % N_COLUMNS) * N_LANES + index;
                                out_gradients_buffer[set_out][time][pos] = out_bits;
                                index++;
                                if(index == N_LANES) {
                                    time++;
                                    index = 0;
                                }
                                if (out_bits != 0) non_zeroes++;
                            }
                        }
                        auto ideal_time = (uint64_t)ceil(non_zeroes / (double)N_LANES);
                        if (ideal_time > ideal_time_per_out_grad_channel[set_out])
                            ideal_time_per_out_grad_channel[set_out] = ideal_time;
                    }


                    for (int window = 0; window < (Kx * Ky); ++window) {
                        auto x_window = window % Kx;
                        auto y_window = window / Kx;

                        for (int k = 0; k < act_channels; k += N_ROWS) {

                            // Generate activation buffer
                            auto time_per_act_channel = (uint64_t)ceil(spatial_pad / (double)N_LANES);

                            schedule_buffer activation_buffer = schedule_buffer(time_per_act_channel,
                                    std::vector<value_mux>(N_ROWS * N_LANES, std::make_tuple(0.0f, 0, 0)));

                            uint64_t ideal_time_per_act_channel = 0;

                            for(int act_channel = k; act_channel < std::min((uint64_t)k + N_ROWS, act_channels);
                                    ++act_channel) {

                                uint64_t non_zeroes = 0;

                                int index = 0;
                                int time = 0;
                                for (int y = 0; y < Oy_dil; ++y) {
                                    for (int x = 0; x < Ox_dil; ++x) {
                                        auto act_bits = act.get(0, act_channel, x_window + x, y_window + y);
                                        int pos = (act_channel % N_ROWS) * N_LANES + index;
                                        activation_buffer[time][pos] = std::make_tuple(act_bits, time, index);
                                        index++;
                                        if(index == N_LANES) {
                                            time++;
                                            index = 0;
                                        }
                                        if (act_bits != 0) non_zeroes++;
                                    }
                                }
                                auto ideal_time = (uint64_t)ceil(non_zeroes / (double)N_LANES);
                                if (ideal_time > ideal_time_per_act_channel)
                                    ideal_time_per_act_channel = ideal_time;
                            }

                            // Schedule buffer
                            original_schedule(activation_buffer);

                            for (int set = 0; set < num_out_grad_sets; ++set) {

                                batch_wgt_act_compute_cycles += activation_buffer.size();
                                batch_wgt_act_base_compute_cycles += time_per_act_channel;
                                batch_wgt_act_ideal_compute_cycles += ideal_time_per_act_channel;

                                if (this->CHECK) {
                                    for (const auto &time_buffer : activation_buffer) {

                                        for (int a = 0; a < N_ROWS; ++a) {
                                            auto act_channel_idx = a * N_LANES;
                                            auto act_channel = k + a;

                                            if (act_channel >= act_channels)
                                                continue;

                                            for (int o = 0; o < N_COLUMNS; ++o) {
                                                auto out_grad_channel_idx = o * N_LANES;
                                                auto out_grad_channel = set * N_COLUMNS + o;

                                                if (out_grad_channel >= out_channels)
                                                    continue;

                                                for (int lane = 0; lane < N_LANES; ++lane) {

                                                    auto act_bits = std::get<0>(time_buffer[act_channel_idx + lane]);
                                                    auto time_h = std::get<1>(time_buffer[act_channel_idx + lane]);
                                                    auto lane_d = std::get<2>(time_buffer[act_channel_idx + lane]);

                                                    auto out_bits = out_gradients_buffer
                                                            [set][time_h][out_grad_channel_idx + lane_d];

                                                    sim_weight_act_gradients
                                                    [out_grad_channel][act_channel][x_window][y_window]
                                                            += act_bits * out_bits;

                                                } // Multiply 16 output gradients and 16 activations

                                            } // Output Gradients
                                        } // Window
                                    } // Time of the buffers
                                } // Check

                            } // Output Gradients Channels sets

                        } // Activations Channels sets

                    } // Windows

                    #pragma omp critical
                    {
                        bw_wgt_act_compute_cycles->value[epoch][layer_it] += batch_wgt_act_compute_cycles;
                        bw_wgt_act_base_compute_cycles->value[epoch][layer_it] += batch_wgt_act_base_compute_cycles;
                        bw_wgt_act_ideal_compute_cycles->value[epoch][layer_it] += batch_wgt_act_ideal_compute_cycles;
                    }

                    out_gradients_buffer.clear();

                    // Simulate: Backward convolution A * G = WG: Gradients sparsity
                    auto sim_weight_out_gradients = std::vector<std::vector<std::vector<std::vector<double>>>>(
                            num_filters, std::vector<std::vector<std::vector<double>>>(wgt_channels,
                            std::vector<std::vector<double>>(Kx, std::vector<double>(Ky, 0))));

                    // Stats
                    uint64_t batch_wgt_out_compute_cycles = 0;
                    uint64_t batch_wgt_out_base_compute_cycles = 0;
                    uint64_t batch_wgt_out_ideal_compute_cycles = 0;

                    // Schedule output gradients
                    auto num_out_grad_sets_sch = (uint64_t)ceil(out_channels / (double)N_ROWS);
                    auto time_per_out_grad_channel_sch = (uint64_t)ceil(spatial_pad / (double)N_LANES);

                    std::vector<schedule_buffer> out_gradients_buffer_sch = std::vector<schedule_buffer>(
                            num_out_grad_sets_sch, schedule_buffer(time_per_out_grad_channel_sch,
                            std::vector<value_mux>(N_ROWS * N_LANES, std::make_tuple(0.0f, 0, 0))));

                    std::vector<uint64_t> ideal_time_per_out_grad_channel_sch (ceil(out_channels/(double)N_ROWS), 0);

                    int set_out_sch = -1;
                    for(int o = 0; o < out_channels; ++o) {

                        if ((o % N_ROWS) == 0)
                            set_out_sch++;

                        uint64_t non_zeroes = 0;

                        int index = 0;
                        int time = 0;
                        for (int y = 0; y < Oy_dil; ++y) {
                            for (int x = 0; x < Ox_dil; ++x) {
                                auto out_bits = out_grad.get(0, o, x, y);
                                int pos = (o % N_ROWS) * N_LANES + index;
                                out_gradients_buffer_sch[set_out_sch][time][pos] = std::make_tuple(out_bits, time, index);
                                index++;
                                if(index == N_LANES) {
                                    time++;
                                    index = 0;
                                }
                                if (out_bits != 0) non_zeroes++;
                            }
                        }
                        auto ideal_time = (uint64_t)ceil(non_zeroes / (double)N_LANES);
                        if (ideal_time > ideal_time_per_out_grad_channel_sch[set_out_sch])
                            ideal_time_per_out_grad_channel_sch[set_out_sch] = ideal_time;
                    }

                    for (auto &gradients_channel_buffer : out_gradients_buffer_sch) {
                        original_schedule(gradients_channel_buffer);
                    }

                    for (int window = 0; window < (Kx * Ky); ++window) {
                        auto x_window = window % Kx;
                        auto y_window = window / Kx;

                        for (int k = 0; k < act_channels; k += N_COLUMNS) {

                            // Generate activation buffer
                            auto time_per_act_channel = (uint64_t)ceil(spatial_pad / (double)N_LANES);

                            std::vector<std::vector<float>> activation_buffer = std::vector<std::vector<float>>(
                                    time_per_act_channel, std::vector<float>(N_COLUMNS * N_LANES, 0.0f));

                            for(int act_channel = k; act_channel < std::min((uint64_t)k + N_COLUMNS, act_channels);
                                ++act_channel) {

                                int index = 0;
                                int time = 0;
                                for (int y = 0; y < Oy_dil; ++y) {
                                    for (int x = 0; x < Ox_dil; ++x) {
                                        auto act_bits = act.get(0, act_channel, x_window + x, y_window + y);
                                        int pos = (act_channel % N_COLUMNS) * N_LANES + index;
                                        activation_buffer[time][pos] = act_bits;
                                        index++;
                                        if(index == N_LANES) {
                                            time++;
                                            index = 0;
                                        }
                                    }
                                }

                            }

                            for (int set = 0; set < num_out_grad_sets_sch; ++set) {

                                batch_wgt_out_compute_cycles += out_gradients_buffer_sch[set].size();
                                batch_wgt_out_base_compute_cycles += time_per_out_grad_channel_sch;
                                batch_wgt_out_ideal_compute_cycles += ideal_time_per_out_grad_channel_sch[set];

                                if (this->CHECK) {
                                    for (const auto &time_buffer : out_gradients_buffer_sch[set]) {

                                        for (int a = 0; a < N_COLUMNS; ++a) {
                                            auto act_channel_idx = a * N_LANES;
                                            auto act_channel = k + a;

                                            if (act_channel >= act_channels)
                                                continue;

                                            for (int o = 0; o < N_ROWS; ++o) {
                                                auto out_grad_channel_idx = o * N_LANES;
                                                auto out_grad_channel = set * N_ROWS + o;

                                                if (out_grad_channel >= out_channels)
                                                    continue;

                                                for (int lane = 0; lane < N_LANES; ++lane) {

                                                    auto out_bits = std::get<0>(time_buffer[out_grad_channel_idx + lane]);
                                                    auto time_h = std::get<1>(time_buffer[out_grad_channel_idx + lane]);
                                                    auto lane_d = std::get<2>(time_buffer[out_grad_channel_idx + lane]);

                                                    auto act_bits = activation_buffer[time_h][act_channel_idx + lane_d];

                                                    sim_weight_out_gradients
                                                    [out_grad_channel][act_channel][x_window][y_window]
                                                            += act_bits * out_bits;

                                                } // Multiply 16 output gradients and 16 activations

                                            } // Output Gradients
                                        } // Window
                                    } // Time of the buffers
                                } // Check

                            } // Output Gradients Channels sets

                        } // Activations Channels sets

                    } // Windows

                    #pragma omp critical
                    {
                        bw_wgt_out_compute_cycles->value[epoch][layer_it] += batch_wgt_out_compute_cycles;
                        bw_wgt_out_base_compute_cycles->value[epoch][layer_it] += batch_wgt_out_base_compute_cycles;
                        bw_wgt_out_ideal_compute_cycles->value[epoch][layer_it] += batch_wgt_out_ideal_compute_cycles;
                    }

                    out_gradients_buffer_sch.clear();

                    // Check correctness of the outputs
                    if (this->CHECK) {

                        auto weight_gradients = std::vector<std::vector<std::vector<std::vector<double>>>>(num_filters,
                                std::vector<std::vector<std::vector<double>>>(wgt_channels,
                                std::vector<std::vector<double>>(Kx, std::vector<double>(Ky, 0))));

                        for (int o = 0; o < out_channels; ++o) {
                            for (int k = 0; k < act_channels; ++k) {

                                // Number of Windows
                                for (int x = 0; x < Kx; ++x) {
                                    for (int y = 0; y < Ky; ++y) {

                                        double sum = 0;

                                        // Window dimensions
                                        for (int j = 0; j < Oy_dil; ++j) {
                                            for (int i = 0; i < Ox_dil; ++i) {
                                                sum += out_grad.get(0, o, i, j) * act.get(0, k, x + i, y + j);
                                            }
                                        }

                                        weight_gradients[o][k][x][y] += sum;

                                    }
                                }
                            }

                        }

                        // Check values: Activations sparsity
                        for (int m = 0; m < num_filters; ++m) {
                            for (int ch = 0; ch < wgt_channels; ++ch) {
                                for (int x = 0; x < Kx; ++x) {
                                    for (int y = 0; y < Ky; ++y) {
                                        auto actual_value = weight_gradients[m][ch][x][y];
                                        auto sim_value = sim_weight_act_gradients[m][ch][x][y];
                                        auto error = (actual_value - sim_value) / sim_value;
                                        if (abs(error) > 1e-10)
                                            throw std::runtime_error("Backward weight gradients convolution "
                                                                     "(Act. Sparsity) wrong value.");
                                    }
                                }
                            }
                        }

                        // Check values: Gradients sparsity
                        for (int m = 0; m < num_filters; ++m) {
                            for (int ch = 0; ch < wgt_channels; ++ch) {
                                for (int x = 0; x < Kx; ++x) {
                                    for (int y = 0; y < Ky; ++y) {
                                        auto actual_value = weight_gradients[m][ch][x][y];
                                        auto sim_value = sim_weight_out_gradients[m][ch][x][y];
                                        auto error = (actual_value - sim_value) / sim_value;
                                        if (abs(error) > 1e-10)
                                            throw std::runtime_error("Backward weight gradients convolution "
                                                                     "(Grad. Sparsity) wrong value.");
                                    }
                                }
                            }
                        }

                    } // Check results

                    // Backward pass - Calculate Input gradients
                    if (layer_it == 0)
                        continue;

                    if (pad_type == 1) out_grad.zero_pad_y(padding + stride - 1);
                    else if (pad_type == 2) out_grad.zero_pad_x(padding + stride - 1);
                    else if (pad_type == 3) out_grad.asym_zero_pad(padding + stride - 1);
                    else if (pad_type == 4) out_grad.zero_pad(padding + stride - 1);
                    else if ((Ox - Kx + 1) != Nx) out_grad.zero_pad(Kx - 1);

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
                        throw std::runtime_error("Input gradients incorrect X window sizes");
                    if ((Oy_pad - Ky + 1) != Ny)
                        throw std::runtime_error("Input gradients incorrect Y window sizes");

                    // Simulate: Backward convolution W * G = IG
                    auto sim_input_gradients = std::vector<std::vector<std::vector<double>>>(act_channels,
                            std::vector<std::vector<double>>(Nx, std::vector<double>(Ny, 0)));

                    // Stats
                    uint64_t batch_in_compute_cycles = 0;
                    uint64_t batch_in_base_compute_cycles = 0;
                    uint64_t batch_in_ideal_compute_cycles = 0;

                    // Generate weight buffer
                    auto num_filter_sets = (uint64_t)ceil(num_filters_rot / (double)N_COLUMNS);

                    auto round_wgt_channels = (int)ceil(wgt_channels_rot / (double)N_LANES) * N_LANES;
                    auto time_per_filter = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)N_LANES);

                    non_schedule_buffer weight_buffer = non_schedule_buffer(num_filter_sets,
                            std::vector<std::vector<float>>(time_per_filter,
                            std::vector<float>(N_COLUMNS * N_LANES, 0.0f)));

                    int set_wgt = -1;
                    for(int m = 0; m < num_filters_rot; m++) {

                        if ((m % N_COLUMNS) == 0)
                            set_wgt++;

                        int time = 0;
                        for (int y = 0; y < Ky; ++y) {
                            for (int x = 0; x < Kx; ++x) {
                                for (int k = 0; k < wgt_channels_rot; k += N_LANES) {
                                    int index = 0;
                                    for(int channel = k; channel < std::min((uint64_t)k + N_LANES,
                                            wgt_channels_rot); channel++) {
                                        auto wgt_bits = wgt.get(m, channel, x, y);
                                        int pos = (m % N_COLUMNS) * N_LANES + index;
                                        weight_buffer[set_wgt][time][pos] = wgt_bits;
                                        index++;
                                        if(index == N_LANES) {
                                            time++;
                                            index = 0;
                                        }
                                    }
                                    if(index != 0)
                                        time++;
                                }
                            }
                        }

                    }

                    std::vector<int> x_windows, y_windows;
                    int x_counter = 0, y_counter = 0;
                    while(this->iterateWindows(Nx, Ny, x_windows, y_windows, x_counter, y_counter, N_ROWS)) {

                        // Generate gradients buffer
                        auto round_out_channels = (int)ceil(out_channels / (double)N_LANES) * N_LANES;
                        auto time_per_window = (uint64_t)ceil(round_out_channels * Kx * Ky / (double)N_LANES);

                        schedule_buffer gradients_buffer = schedule_buffer(time_per_window,
                                std::vector<value_mux>(x_windows.size() * N_LANES, std::make_tuple(0.0f, 0, 0)));

                        uint64_t ideal_time_per_window = 0;

                        for (int w = 0; w < x_windows.size(); ++w) {
                            auto x_window = x_windows[w];
                            auto y_window = y_windows[w];

                            uint64_t non_zeroes = 0;

                            int time = 0;
                            for (int y = 0; y < Ky; ++y) {
                                for (int x = 0; x < Kx; ++x) {
                                    for (int k = 0; k < out_channels; k += N_LANES) {
                                        int index = 0;
                                        for (int channel = k; channel < std::min((uint64_t)k + N_LANES,
                                                out_channels); channel++) {
                                            auto out_bits = out_grad.get(0, channel, x_window + x, y_window + y);
                                            int pos = w * N_LANES + index;
                                            gradients_buffer[time][pos] = std::make_tuple(out_bits, time, index);
                                            index++;
                                            if(index == N_LANES) {
                                                time++;
                                                index = 0;
                                            }
                                            if (out_bits != 0) non_zeroes++;
                                        }
                                        if (index != 0)
                                            time++;
                                    }
                                }
                            }

                            auto ideal_time = (uint64_t)ceil(non_zeroes / (double)N_LANES);
                            if (ideal_time > ideal_time_per_window)
                                ideal_time_per_window = ideal_time;

                        }

                        // Schedule buffer
                        original_schedule(gradients_buffer);

                        for (int set = 0; set < num_filter_sets; ++set) {

                            batch_in_compute_cycles += gradients_buffer.size();
                            batch_in_base_compute_cycles += time_per_window;
                            batch_in_ideal_compute_cycles += ideal_time_per_window;

                            if (this->CHECK) {
                                for (const auto &time_buffer : gradients_buffer) {

                                    for (int w = 0; w < x_windows.size(); ++w) {
                                        auto window_idx = w * N_LANES;
                                        auto x_window = x_windows[w];
                                        auto y_window = y_windows[w];

                                        for (int f = 0; f < N_COLUMNS; ++f) {
                                            auto filter_idx = f * N_LANES;
                                            auto filter = set * N_COLUMNS + f;

                                            if (filter >= num_filters_rot)
                                                continue;

                                            for (int lane = 0; lane < N_LANES; ++lane) {

                                                auto out_bits = std::get<0>(time_buffer[window_idx + lane]);
                                                auto time_h = std::get<1>(time_buffer[window_idx + lane]);
                                                auto lane_d = std::get<2>(time_buffer[window_idx + lane]);

                                                auto wgt_bits = weight_buffer[set][time_h][filter_idx + lane_d];

                                                sim_input_gradients[filter][x_window][y_window] += out_bits * wgt_bits;

                                            } // Multiply 16 weights and 16 output gradients

                                        } // Filter
                                    } // Window
                                } // Time of the buffers
                            } // Check

                        } // Filter sets

                    } // Window sets

                    #pragma omp critical
                    {
                        bw_in_compute_cycles->value[epoch][layer_it] += batch_in_compute_cycles;
                        bw_in_base_compute_cycles->value[epoch][layer_it] += batch_in_base_compute_cycles;
                        bw_in_ideal_compute_cycles->value[epoch][layer_it] += batch_in_ideal_compute_cycles;
                    }

                    // Check correctness of the outputs
                    if (this->CHECK) {

                        auto input_gradients = std::vector<std::vector<std::vector<double>>>(act_channels,
                                std::vector<std::vector<double>>(Nx, std::vector<double>(Ny, 0)));

                        // Actual convolution
                        for (int m = 0; m < num_filters_rot; ++m) {

                            // Number of Windows
                            for (int x = 0; x < Nx; ++x) {
                                for (int y = 0; y < Ny; ++y) {

                                    double sum = 0;

                                    // Windows dimension
                                    for (int j = 0; j < Ky; ++j) {
                                        for (int i = 0; i < Kx; ++i) {
                                            for (int k = 0; k < wgt_channels_rot; ++k) {
                                                sum += out_grad.get(0, k, x + i, y + j) * wgt.get(m, k, i, j);
                                            }
                                        }
                                    }

                                    input_gradients[m][x][y] = sum;
                                }
                            }
                        }

                        // Check values
                        for (int ch = 0; ch < act_channels; ++ch) {
                            for (int x = 0; x < Nx; ++x) {
                                for (int y = 0; y < Ny; ++y) {
                                    auto actual_value = input_gradients[ch][x][y];
                                    auto sim_value = sim_input_gradients[ch][x][y];
                                    auto error = (actual_value - sim_value) / sim_value;
                                    if (abs(error) > 1e-10)
                                        throw std::runtime_error("Backward input gradients convolution wrong value.");
                                }
                            }
                        }

                    } // Check results

                } // Backward pass

            } // Batch

            // Calculate special stats
            for (int layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

                fw_speedup->value[epoch][layer_it] = fw_base_compute_cycles->value[epoch][layer_it] /
                        (double)fw_compute_cycles->value[epoch][layer_it];
                fw_exploited_sparsity->value[epoch][layer_it] =
                        (fw_base_compute_cycles->value[epoch][layer_it] -
                        fw_compute_cycles->value[epoch][layer_it])
                        / (double)(fw_base_compute_cycles->value[epoch][layer_it] -
                        fw_ideal_compute_cycles->value[epoch][layer_it]) * 100.;
                if (isnan(fw_exploited_sparsity->value[epoch][layer_it]))
                    fw_exploited_sparsity->value[epoch][layer_it] = 0;

                bw_wgt_act_speedup->value[epoch][layer_it] = bw_wgt_act_base_compute_cycles->value[epoch][layer_it] /
                        (double)bw_wgt_act_compute_cycles->value[epoch][layer_it];
                bw_wgt_act_exploited_sparsity->value[epoch][layer_it] =
                        (bw_wgt_act_base_compute_cycles->value[epoch][layer_it] -
                         bw_wgt_act_compute_cycles->value[epoch][layer_it])
                        / (double)(bw_wgt_act_base_compute_cycles->value[epoch][layer_it] -
                        bw_wgt_act_ideal_compute_cycles->value[epoch][layer_it]) * 100.;
                if (isnan(bw_wgt_act_exploited_sparsity->value[epoch][layer_it]))
                    bw_wgt_act_exploited_sparsity->value[epoch][layer_it] = 0;

                bw_wgt_out_speedup->value[epoch][layer_it] = bw_wgt_out_base_compute_cycles->value[epoch][layer_it] /
                        (double)bw_wgt_out_compute_cycles->value[epoch][layer_it];
                bw_wgt_out_exploited_sparsity->value[epoch][layer_it] =
                        (bw_wgt_out_base_compute_cycles->value[epoch][layer_it] -
                         bw_wgt_out_compute_cycles->value[epoch][layer_it])
                        / (double)(bw_wgt_out_base_compute_cycles->value[epoch][layer_it] -
                        bw_wgt_out_ideal_compute_cycles->value[epoch][layer_it]) * 100.;
                if (isnan(bw_wgt_out_exploited_sparsity->value[epoch][layer_it]))
                    bw_wgt_out_exploited_sparsity->value[epoch][layer_it] = 0;

                bw_wgt_compute_cycles->value[epoch][layer_it] = std::min(bw_wgt_act_compute_cycles->value[epoch][layer_it],
                        bw_wgt_out_compute_cycles->value[epoch][layer_it]);
                bw_wgt_speedup->value[epoch][layer_it] = bw_wgt_out_base_compute_cycles->value[epoch][layer_it] /
                        (double)bw_wgt_compute_cycles->value[epoch][layer_it];

                if (layer_it != 0) {
                    bw_in_speedup->value[epoch][layer_it] = bw_in_base_compute_cycles->value[epoch][layer_it] /
                            (double) bw_in_compute_cycles->value[epoch][layer_it];
                    bw_in_exploited_sparsity->value[epoch][layer_it] =
                            (bw_in_base_compute_cycles->value[epoch][layer_it] -
                             bw_in_compute_cycles->value[epoch][layer_it])
                            / (double)(bw_in_base_compute_cycles->value[epoch][layer_it] -
                            bw_in_ideal_compute_cycles->value[epoch][layer_it]) * 100.;
                    if (isnan(bw_in_exploited_sparsity->value[epoch][layer_it]))
                        bw_in_exploited_sparsity->value[epoch][layer_it] = 0;
                }

            }

            fw_speedup->special_value_vector.push_back(sys::get_total(fw_base_compute_cycles->value[epoch]) /
                    (double)sys::get_total(fw_compute_cycles->value[epoch]));
            fw_exploited_sparsity->special_value_vector.push_back((
                    sys::get_total(fw_base_compute_cycles->value[epoch]) -
                    sys::get_total(fw_compute_cycles->value[epoch])) /
                    (double)(sys::get_total(fw_base_compute_cycles->value[epoch]) -
                    sys::get_total(fw_ideal_compute_cycles->value[epoch])) * 100.);

            bw_wgt_act_speedup->special_value_vector.push_back(
                    sys::get_total(bw_wgt_act_base_compute_cycles->value[epoch]) /
                    (double)sys::get_total(bw_wgt_act_compute_cycles->value[epoch]));
            bw_wgt_act_exploited_sparsity->special_value_vector.push_back(
                    (sys::get_total(bw_wgt_act_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_wgt_act_compute_cycles->value[epoch])) /
                    (double)(sys::get_total(bw_wgt_act_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_wgt_act_ideal_compute_cycles->value[epoch])) * 100.);

            bw_wgt_out_speedup->special_value_vector.push_back(
                    sys::get_total(bw_wgt_out_base_compute_cycles->value[epoch]) /
                    (double)sys::get_total(bw_wgt_out_compute_cycles->value[epoch]));
            bw_wgt_out_exploited_sparsity->special_value_vector.push_back(
                    (sys::get_total(bw_wgt_out_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_wgt_out_compute_cycles->value[epoch])) /
                    (double)(sys::get_total(bw_wgt_out_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_wgt_out_ideal_compute_cycles->value[epoch])) * 100.);

            bw_wgt_speedup->special_value_vector.push_back(
                    sys::get_total(bw_wgt_out_base_compute_cycles->value[epoch]) /
                    (double)sys::get_total(bw_wgt_compute_cycles->value[epoch]));

            bw_in_speedup->special_value_vector.push_back(sys::get_total(bw_in_base_compute_cycles->value[epoch]) /
                    (double)sys::get_total(bw_in_compute_cycles->value[epoch]));
            bw_in_exploited_sparsity->special_value_vector.push_back(
                    (sys::get_total(bw_in_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_in_compute_cycles->value[epoch])) /
                    (double)(sys::get_total(bw_in_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_in_ideal_compute_cycles->value[epoch])) * 100.);


        } // Epoch

        fw_speedup->special_value = sys::get_total(fw_base_compute_cycles->value) /
                (double)sys::get_total(fw_compute_cycles->value);
        fw_exploited_sparsity->special_value =
                (sys::get_total(fw_base_compute_cycles->value) -
                sys::get_total(fw_compute_cycles->value)) /
                (double)(sys::get_total(fw_base_compute_cycles->value) -
                sys::get_total(fw_ideal_compute_cycles->value)) * 100.;

        bw_wgt_act_speedup->special_value = sys::get_total(bw_wgt_act_base_compute_cycles->value) /
                (double)sys::get_total(bw_wgt_act_compute_cycles->value);
        bw_wgt_act_exploited_sparsity->special_value =
                (sys::get_total(bw_wgt_act_base_compute_cycles->value) -
                sys::get_total(bw_wgt_act_compute_cycles->value)) /
                (double)(sys::get_total(bw_wgt_act_base_compute_cycles->value) -
                sys::get_total(bw_wgt_act_ideal_compute_cycles->value)) * 100.;

        bw_wgt_out_speedup->special_value = sys::get_total(bw_wgt_out_base_compute_cycles->value) /
                (double)sys::get_total(bw_wgt_out_compute_cycles->value);
        bw_wgt_out_exploited_sparsity->special_value =
                (sys::get_total(bw_wgt_out_base_compute_cycles->value) -
                sys::get_total(bw_wgt_out_compute_cycles->value)) /
                (double)(sys::get_total(bw_wgt_out_base_compute_cycles->value) -
                sys::get_total(bw_wgt_out_ideal_compute_cycles->value)) * 100.;

        bw_wgt_speedup->special_value = sys::get_total(bw_wgt_out_base_compute_cycles->value) /
                (double)sys::get_total(bw_wgt_compute_cycles->value);

        bw_in_speedup->special_value = sys::get_total(bw_in_base_compute_cycles->value) /
                (double)sys::get_total(bw_in_compute_cycles->value);
        bw_in_exploited_sparsity->special_value =
                (sys::get_total(bw_in_base_compute_cycles->value) -
                sys::get_total(bw_in_compute_cycles->value)) /
                (double)(sys::get_total(bw_in_base_compute_cycles->value) -
                sys::get_total(bw_in_ideal_compute_cycles->value)) * 100.;

        //Dump statistics
        std::string header = "DynamicTactical Number of Cycles for " + network_model.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(N_TILES) + "\n";
        header += "Search shape: " + std::string(1, this->SEARCH_SHAPE) + "\n";
        header += "Lookahead H: " + std::to_string(this->LOOKAHEAD_H) + "\n";
        header += "Lookaside D: " + std::to_string(this->LOOKASIDE_D) + "\n";

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
        sys::Stats stats = sys::Stats(epochs, network_model.getNumLayers(), filename);
        stats.setTraining(true);

        // Forward stats
        auto act_work_reduction = stats.register_double_t("Forward Activations Work Reduction", 0, sys::Average);
        auto act_speedup = stats.register_double_t("Forward Activations Speedup", 0, sys::Average);
        auto act_par_mult = stats.register_double_t("Forward Activations Multiplications", 0, sys::Total);
        auto act_bit_multiplications = stats.register_uint_t("Forward Activations Bit Multiplications", 0, sys::Total);

        auto wgt_work_reduction = stats.register_double_t("Forward Weights Work Reduction", 0, sys::Average);
        auto wgt_speedup = stats.register_double_t("Forward Weights Speedup", 0, sys::Average);
        auto wgt_par_mult = stats.register_double_t("Forward Weights Multiplications", 0, sys::Total);
        auto wgt_bit_multiplications = stats.register_uint_t("Forward Weights Bit Multiplications", 0, sys::Total);

        // Backward stats
        auto bw_wgt_work_reduction = stats.register_double_t("Backward Weights Work Reduction", 0, sys::Average);
        auto bw_wgt_speedup = stats.register_double_t("Backward Weights Speedup", 0, sys::Average);
        auto bw_wgt_par_mult = stats.register_double_t("Backward Weights Multiplications", 0, sys::Total);
        auto bw_wgt_bit_multiplications = stats.register_uint_t("Backward Weights Bit Multiplications", 0, sys::Total);

        auto wgt_out_grad_work_reduction = stats.register_double_t("Backward Output Gradients Work Reduction", 0, sys::Average);
        auto wgt_out_grad_speedup = stats.register_double_t("Backward Output Gradients Speedup", 0, sys::Average);
        auto wgt_out_grad_par_mult = stats.register_double_t("Backward Output Gradients Multiplications", 0, sys::Total);
        auto wgt_out_grad_bit_multiplications = stats.register_uint_t("Backward Output Gradients Bit Multiplications", 0, sys::Total);

        auto bw_act_work_reduction = stats.register_double_t("Backward Activations Work Reduction", 0, sys::Average);
        auto bw_act_speedup = stats.register_double_t("Backward Activations Speedup", 0, sys::Average);
        auto bw_act_par_mult = stats.register_double_t("Backward Activations Multiplications", 0, sys::Total);
        auto bw_act_bit_multiplications = stats.register_uint_t("Backward Activations Bit Multiplications", 0, sys::Total);

        auto act_out_grad_work_reduction = stats.register_double_t("Backward Output Gradients Work Reduction", 0, sys::Average);
        auto act_out_grad_speedup = stats.register_double_t("Backward Output Gradients Speedup", 0, sys::Average);
        auto act_out_grad_par_mult = stats.register_double_t("Backward Output Gradients Multiplications", 0, sys::Total);
        auto act_out_grad_bit_multiplications = stats.register_uint_t("Backward Output Gradients Bit Multiplications", 0, sys::Total);

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

                    act_bit_multiplications->value[n][layer_it] = act_bit_counter;
                    act_work_reduction->value[n][layer_it] = 100 - ((double)act_bit_counter / (double)fw_parallel_mult
                            / MAX_BITS * 100);
                    act_speedup->value[n][layer_it] = (double)fw_parallel_mult * MAX_BITS / (double)act_bit_counter;
                    act_par_mult->value[n][layer_it] = fw_parallel_mult;

                    wgt_bit_multiplications->value[n][layer_it] = wgt_bit_counter;
                    wgt_work_reduction->value[n][layer_it] = 100 - ((double)wgt_bit_counter / (double)fw_parallel_mult
                            / MAX_BITS * 100);
                    wgt_speedup->value[n][layer_it] = (double)fw_parallel_mult * MAX_BITS / (double)wgt_bit_counter;
                    wgt_par_mult->value[n][layer_it] = fw_parallel_mult;

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

                    bw_act_bit_multiplications->value[n][layer_it] = act_bit_counter;
                    bw_act_work_reduction->value[n][layer_it] = 100 - ((double)act_bit_counter /
                            (double)bw_act_parallel_mult / MAX_BITS * 100);
                    bw_act_speedup->value[n][layer_it] = (double)bw_act_parallel_mult * MAX_BITS /
                            (double)act_bit_counter;
                    bw_act_par_mult->value[n][layer_it] = bw_act_parallel_mult;

                    act_out_grad_bit_multiplications->value[n][layer_it] = out_grad_bit_counter;
                    act_out_grad_work_reduction->value[n][layer_it] = 100 - ((double)out_grad_bit_counter /
                            (double)bw_act_parallel_mult / MAX_BITS * 100);
                    act_out_grad_speedup->value[n][layer_it] = (double)bw_act_parallel_mult * MAX_BITS /
                            (double)out_grad_bit_counter;
                    act_out_grad_par_mult->value[n][layer_it] = bw_act_parallel_mult;

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

                    bw_wgt_bit_multiplications->value[n][layer_it]= wgt_bit_counter;
                    bw_wgt_work_reduction->value[n][layer_it] = 100 - ((double)wgt_bit_counter /
                            (double)bw_wgt_parallel_mult / MAX_BITS * 100);
                    bw_wgt_speedup->value[n][layer_it] = (double)fw_parallel_mult * MAX_BITS /
                            (double)bw_wgt_parallel_mult;
                    bw_wgt_par_mult->value[n][layer_it] = bw_wgt_parallel_mult;

                    wgt_out_grad_bit_multiplications->value[n][layer_it] = out_grad_bit_counter;
                    wgt_out_grad_work_reduction->value[n][layer_it] = 100 - ((double)out_grad_bit_counter /
                            (double)bw_wgt_parallel_mult / MAX_BITS * 100);
                    wgt_out_grad_speedup->value[n][layer_it] = (double)bw_wgt_parallel_mult * MAX_BITS /
                            (double)out_grad_bit_counter;
                    wgt_out_grad_par_mult->value[n][layer_it] = bw_wgt_parallel_mult;

                }

            }

        }


        //Dump statistics
        std::string header = "DynamicTactical Potentials for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    template class DynamicTactical<float>;

}
