
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

    /* SCHEDULER */

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

        int skip = 0;
        for (int time = 0; time < max_time; ++time) {

            // Skip lines of zeroes
            if (skip < LOOKAHEAD_H && check_zero_line(schedule[time])) {
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
                        auto value_tuple = schedule[time][lane];
                        auto value_bits = std::get<0>(value_tuple);
                        if(value_bits == 0) ineffectual_values.emplace_back(std::make_tuple(time, lane));
                    }

                    // Num of candidates for each ineffectual values
                    overlap = -1;
                    std::vector<uint16_t> num_candidates (N_LANES, 0);
                    std::vector<std::vector<value_index>> effectual_candidates (N_LANES, std::vector<value_index>());
                    for(auto inef_idx : ineffectual_values) {
                        auto lane = std::get<1>(inef_idx);
                        effectual_candidates[lane % N_LANES] = search(schedule, inef_idx, max_time);
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
                            promote(schedule, inef_idx, cand_idx);
                            break;
                        }
                    }

                } // Optimal promotion loop

            } // Group
        } // Time

    }

    /* CHECKING FUNCTIONS */

    template <typename T>
    void check_result_channel_first(const output_tensor &sim_output, const base::Array<T> &input,
            const base::Array<T> &wgt, uint64_t Ox, uint64_t Oy, int stride) {

        const std::vector<size_t> &in_shape = input.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        // Input values
        auto in_channels = in_shape[1];

        // Weights
        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        output_tensor output = output_tensor(1,std::vector<std::vector<std::vector<double>>>(num_filters,
                std::vector<std::vector<double>>(Ox, std::vector<double>(Oy, 0))));

        // Actual convolution
        for (int m = 0; m < num_filters; ++m) {

            // Fix for MobileNet
            int start_group = 0;
            if(wgt_channels == 1 && in_channels != 1)
                start_group = m;

            // Number of Windows
            for (int x = 0; x < Ox; ++x) {
                for (int y = 0; y < Oy; ++y) {

                    double sum = 0;

                    // Window dimension
                    for (int j = 0; j < Ky; ++j) {
                        for (int i = 0; i < Kx; ++i) {
                            for (int k = 0; k < wgt_channels; ++k) {
                                sum += input.get(0, start_group + k, stride * x + i, stride * y + j) *
                                        wgt.get(m, k, i, j);
                            }
                        }
                    }

                    output[0][m][x][y] = sum;
                }
            }
        }

        // Check values
        for (int ch = 0; ch < num_filters; ++ch) {
            for (int x = 0; x < Ox; ++x) {
                for (int y = 0; y < Oy; ++y) {
                    auto actual_value = output[0][ch][x][y];
                    auto sim_value = sim_output[0][ch][x][y];
                    auto error = (actual_value - sim_value) / sim_value;
                    if (abs(error) > 1e-10)
                        throw std::runtime_error("Channel First convolution wrong value.");
                }
            }
        }
    }

    template <typename T>
    void check_result_spatial(const output_tensor &sim_output, const base::Array<T> &act,
            const base::Array<T> &out_grad, uint64_t num_filters, uint64_t Kx, uint64_t Ky, uint64_t wgt_channels) {

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &out_grad_shape = out_grad.getShape();

        // Activations
        auto act_channels = act_shape[1];

        // Output gradients
        auto out_channels = out_grad_shape[1];
        auto Ox = out_grad_shape[2];
        auto Oy = out_grad_shape[3];

        output_tensor output = output_tensor(num_filters,std::vector<std::vector<std::vector<double>>>(wgt_channels,
                std::vector<std::vector<double>>(Kx, std::vector<double>(Ky, 0))));

        for (int o = 0; o < out_channels; ++o) {
            for (int k = 0; k < act_channels; ++k) {

                // Number of Windows
                for (int x = 0; x < Kx; ++x) {
                    for (int y = 0; y < Ky; ++y) {

                        double sum = 0;

                        // Window dimensions
                        for (int j = 0; j < Oy; ++j) {
                            for (int i = 0; i < Ox; ++i) {
                                sum += out_grad.get(0, o, i, j) * act.get(0, k, x + i, y + j);
                            }
                        }

                        output[o][k][x][y] += sum;

                    }
                }

            }
        }

        // Check values: Activations sparsity
        for (int m = 0; m < num_filters; ++m) {
            for (int ch = 0; ch < wgt_channels; ++ch) {
                for (int x = 0; x < Kx; ++x) {
                    for (int y = 0; y < Ky; ++y) {
                        auto actual_value = output[m][ch][x][y];
                        auto sim_value = sim_output[m][ch][x][y];
                        auto error = (actual_value - sim_value) / sim_value;
                        if (abs(error) > 1e-10)
                            throw std::runtime_error("Spatial 2D convolution wrong value.");
                    }
                }
            }
        }

    }

    /* CONVOLUTION FUNCTIONS */

    bank_map map_on_chip_activations(uint64_t X, uint64_t Y, uint64_t Ox, int stride, uint64_t BANKS) {

        bank_map values_bank_map = bank_map(Y, std::vector<int>(X, -1));

        int bank = 0;
        int bkp_bank = 0;
        for (int y = 0; y < Y; ++y) {
            for (int x = 0; x < X; ++x) {
                if (y % stride == 0 && x == 0)
                    bank = bkp_bank;
                values_bank_map[y][x] = bank;
                bank = (bank + 1) % BANKS;
                if (y % stride == 0 && x == Ox * stride - 1)
                    bkp_bank = bank;
            }
        }

        return values_bank_map;
    }

    bank_map map_on_chip_gradients(uint64_t X, uint64_t Y, uint64_t Ox, int stride, uint64_t BANKS) {

        bank_map values_bank_map = bank_map(Y, std::vector<int>(X, -1));

        int bank = 0;
        int bkp_bank = 0;
        for (int y = 0; y < Y; ++y) {
            for (int x = 0; x < X; ++x) {
                if (x == 0) bank = bkp_bank;
                values_bank_map[y][x] = bank;
                bank = (bank + 1) % BANKS;
                if (x == Ox / stride - 1) bkp_bank = bank;
            }
        }

        return values_bank_map;
    }

    template <typename T>
    void DynamicTactical<T>::channel_first_convolution(const base::Array<T> &input, const base::Array<T> &wgt,
            const bank_map &input_bank_map, uint64_t Ox, uint64_t Oy, int stride, conv_stats &stats,
            output_tensor &output) {

        const std::vector<size_t> &in_shape = input.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        // Input values
        auto in_channels = in_shape[1];

        // Weights
        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        // Prepare bank
        std::vector<int> read_requests (BANKS, 0);

        // Stats
        stats.cycles = 0;
        stats.compute_cycles = 0;
        stats.base_compute_cycles = 0;
        stats.ideal_compute_cycles = 0;
        stats.read_bank_conflicts = 0;

        // Generate weight buffer
        auto num_filter_sets = (uint64_t)ceil(num_filters / (double)N_COLUMNS);

        auto round_wgt_channels = (int)ceil(wgt_channels / (double)N_LANES) * N_LANES;
        auto time_per_filter = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)N_LANES);

        non_schedule_buffer weight_buffer = non_schedule_buffer(num_filter_sets,
                std::vector<std::vector<float>>(time_per_filter,std::vector<float>(N_COLUMNS * N_LANES, 0.0f)));

        int set_wgt = -1;
        for(int m = 0; m < num_filters; m++) {

            if ((m % N_COLUMNS) == 0)
                set_wgt++;

            int time = 0;
            for (int y = 0; y < Ky; ++y) {
                for (int x = 0; x < Kx; ++x) {
                    for (int k = 0; k < wgt_channels; k += N_LANES) {
                        int index = 0;
                        for(int channel = k; channel < std::min((uint64_t)k + N_LANES, wgt_channels); ++channel) {
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
            auto round_in_channels = (int)ceil(in_channels / (double)N_LANES) * N_LANES;
            auto time_per_window = (uint64_t)ceil(round_in_channels * Kx * Ky / (double)N_LANES);

            schedule_buffer window_buffer = schedule_buffer(time_per_window,
                    std::vector<value_mux>(x_windows.size() * N_LANES, std::make_tuple(0.0f, 0, 0)));

            bank_map window_bank_map = bank_map(time_per_window, std::vector<int>(N_ROWS, -1));

            uint64_t ideal_time_per_window = 0;

            for (int w = 0; w < x_windows.size(); ++w) {
                auto x_window = x_windows[w] * stride;
                auto y_window = y_windows[w] * stride;

                uint64_t non_zeroes = 0;

                int time = 0;
                for (int y = 0; y < Ky; ++y) {
                    for (int x = 0; x < Kx; ++x) {
                        for (int k = 0; k < in_channels; k += N_LANES) {
                            int index = 0;
                            for (int channel = k; channel < std::min((uint64_t)k + N_LANES, in_channels); ++channel) {
                                auto in_bits = input.get(0, channel, x_window + x, y_window + y);
                                int pos = w * N_LANES + index;
                                window_buffer[time][pos] = std::make_tuple(in_bits, time, index);
                                index++;
                                if(index == N_LANES) {
                                    window_bank_map[time][w] = input_bank_map[y_window + y][x_window + x];
                                    time++;
                                    index = 0;
                                }
                                if (in_bits != 0) non_zeroes++;
                            }
                            if (index != 0) {
                                window_bank_map[time][w] = input_bank_map[y_window + y][x_window + x];
                                time++;
                            }
                        }
                    }
                }
                auto ideal_time = (uint64_t)ceil(non_zeroes / (double)N_LANES);
                if (ideal_time > ideal_time_per_window)
                    ideal_time_per_window = ideal_time;

            }

            // Schedule buffer
            original_schedule(window_buffer);

            for (int set = 0; set < num_filter_sets; ++set) {

                stats.base_compute_cycles += time_per_window;
                stats.ideal_compute_cycles += ideal_time_per_window;

                int skip = 0;
                for (int time = 0; time < time_per_window; ++time) {

                    read_requests = std::vector<int>(BANKS, 0);

                    // Input requests
                    for (int w = 0; w < x_windows.size(); ++w) {
                        auto bank = window_bank_map[time][w];
                        if (bank >= 0) read_requests[bank]++;
                    }

                    auto read_bank_conflicts = *std::max_element(read_requests.begin(), read_requests.end()) - 1;
                    stats.read_bank_conflicts += read_bank_conflicts;
                    stats.cycles += read_bank_conflicts;

                    // Skip lines of zeroes
                    if (skip < LOOKAHEAD_H && check_zero_line(window_buffer[time])) {
                        skip++;
                        continue;
                    }
                    skip = 0;

                    stats.compute_cycles++;
                    stats.cycles++;

                    if (this->CHECK) {

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

                                    auto in_bits = std::get<0>(window_buffer[time][window_idx + lane]);
                                    auto time_h = std::get<1>(window_buffer[time][window_idx + lane]);
                                    auto lane_d = std::get<2>(window_buffer[time][window_idx + lane]);

                                    auto wgt_bits = weight_buffer[set][time_h][filter_idx + lane_d];

                                    output[0][filter][x_window][y_window] += in_bits * wgt_bits;

                                } // Multiply 16 weights and 16 window values

                            } // Filter
                        } // Window
                    } //Check

                } // Time of the buffers
            } // Filter sets
        } // Window sets

    }

    template <typename T>
    void DynamicTactical<T>::channel_first_dilated_convolution(const base::Array<T> &input, const base::Array<T> &wgt,
            const bank_map &input_bank_map, int stride, bool asym_pad, conv_stats &stats, output_tensor &output) {

        const std::vector<size_t> &in_shape = input.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        // Input values
        auto in_channels = in_shape[1];
        auto Wx = in_shape[2];
        auto Wy = in_shape[3];

        // Weights
        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        // Prepare bank
        std::vector<int> read_requests (BANKS, 0);

        // Stats
        stats.cycles = 0;
        stats.compute_cycles = 0;
        stats.base_compute_cycles = 0;
        stats.ideal_compute_cycles = 0;
        stats.read_bank_conflicts = 0;

        // Generate new kernels
        std::vector<bool> free_pos (Kx * Ky, true);
        auto next_pos = free_pos.begin();
        std::vector<std::vector<std::tuple<int, int>>> kernel_sets;
        do {

            int init_pos = std::distance(free_pos.begin(), next_pos);
            int pos_x = init_pos % Kx;
            int pos_y = init_pos / Kx;

            std::vector<std::tuple<int, int>> positions;
            while(pos_y < Ky) {
                while(pos_x < Kx) {
                    int index = pos_y * Kx + pos_x;
                    positions.emplace_back(std::make_tuple(pos_x, pos_y));
                    free_pos[index] = false;
                    pos_x += stride;
                }
                pos_x = init_pos % Kx;
                pos_y += stride;
            }

            kernel_sets.push_back(positions);
            next_pos = std::find(free_pos.begin(), free_pos.end(), true);

        } while(next_pos != free_pos.end());

        for (const auto &kernel_positions : kernel_sets) {

            std::list<int> kx, ky;
            for (const auto &index_tuple : kernel_positions) {
                kx.push_back(std::get<0>(index_tuple));
                ky.push_back(std::get<1>(index_tuple));
            }

            kx.sort();
            ky.sort();
            kx.unique();
            ky.unique();
            Kx = kx.size();
            Ky = ky.size();

            auto pad_x = std::get<0>(kernel_positions.front());
            auto pad_y = std::get<1>(kernel_positions.front());
            auto padw_x = asym_pad ? stride - 1 - pad_x : pad_x;
            auto padw_y = asym_pad ? stride - 1 - pad_y : pad_y;

            // Generate weight buffer
            auto num_filter_sets = (uint64_t)ceil(num_filters / (double)N_COLUMNS);

            auto round_wgt_channels = (int)ceil(wgt_channels / (double)N_LANES) * N_LANES;
            auto time_per_filter = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)N_LANES);

            non_schedule_buffer weight_buffer = non_schedule_buffer(num_filter_sets,
                    std::vector<std::vector<float>>(time_per_filter,std::vector<float>(N_COLUMNS * N_LANES, 0.0f)));

            int set_wgt = -1;
            for(int m = 0; m < num_filters; m++) {

                if ((m % N_COLUMNS) == 0)
                    set_wgt++;

                int time = 0;

                for (auto index_tuple : kernel_positions) {
                    auto x = std::get<0>(index_tuple);
                    auto y = std::get<1>(index_tuple);

                    for (int k = 0; k < wgt_channels; k += N_LANES) {
                        int index = 0;
                        for (int channel = k; channel < std::min((uint64_t) k + N_LANES, wgt_channels); ++channel) {
                            auto wgt_bits = wgt.get(m, channel, x, y);
                            int pos = (m % N_COLUMNS) * N_LANES + index;
                            weight_buffer[set_wgt][time][pos] = wgt_bits;
                            index++;
                            if (index == N_LANES) {
                                time++;
                                index = 0;
                            }
                        }
                        if (index != 0)
                            time++;
                    }
                }

            }

            auto Ox = asym_pad ? Wx - Kx - pad_x + 1 : Wx - Kx - 2 * pad_x + 1;
            auto Oy = asym_pad ? Wy - Ky - pad_y + 1 : Wy - Ky - 2 * pad_y + 1;
            pad_x = asym_pad ? 0 : pad_x;
            pad_y = asym_pad ? 0 : pad_y;

            std::vector<int> x_windows, y_windows;
            int x_counter = 0, y_counter = 0;
            while(this->iterateWindows(Ox, Oy, x_windows, y_windows, x_counter, y_counter, N_ROWS)) {

                // Generate activation buffer
                auto round_in_channels = (int)ceil(in_channels / (double)N_LANES) * N_LANES;
                auto time_per_window = (uint64_t)ceil(round_in_channels * Kx * Ky / (double)N_LANES);

                schedule_buffer window_buffer = schedule_buffer(time_per_window,
                        std::vector<value_mux>(x_windows.size() * N_LANES, std::make_tuple(0.0f, 0, 0)));

                bank_map window_bank_map = bank_map(time_per_window, std::vector<int>(N_ROWS, -1));

                uint64_t ideal_time_per_window = 0;

                for (int w = 0; w < x_windows.size(); ++w) {
                    auto x_window = x_windows[w];
                    auto y_window = y_windows[w];

                    uint64_t non_zeroes = 0;

                    int time = 0;
                    for (int y = 0; y < Ky; ++y) {
                        for (int x = 0; x < Kx; ++x) {
                            for (int k = 0; k < in_channels; k += N_LANES) {
                                int index = 0;
                                for (int channel = k; channel < std::min((uint64_t)k + N_LANES, in_channels); ++channel) {
                                    auto in_bits = input.get(0, channel, x_window + x + pad_x, y_window + y + pad_y);
                                    int pos = w * N_LANES + index;
                                    window_buffer[time][pos] = std::make_tuple(in_bits, time, index);
                                    index++;
                                    if(index == N_LANES) {
                                        window_bank_map[time][w] = input_bank_map[y_window + y + pad_y][x_window + x + pad_x];
                                        time++;
                                        index = 0;
                                    }
                                    if (in_bits != 0) non_zeroes++;
                                }
                                if (index != 0) {
                                    window_bank_map[time][w] = input_bank_map[y_window + y + pad_y][x_window + x + pad_x];
                                    time++;
                                }
                            }
                        }
                    }
                    auto ideal_time = (uint64_t)ceil(non_zeroes / (double)N_LANES);
                    if (ideal_time > ideal_time_per_window)
                        ideal_time_per_window = ideal_time;

                }

                // Schedule buffer
                original_schedule(window_buffer);

                for (int set = 0; set < num_filter_sets; ++set) {

                    stats.base_compute_cycles += time_per_window;
                    stats.ideal_compute_cycles += ideal_time_per_window;

                    int skip = 0;
                    for (int time = 0; time < time_per_window; ++time) {

                        read_requests = std::vector<int>(BANKS, 0);

                        // Input requests
                        for (int w = 0; w < x_windows.size(); ++w) {
                            auto bank = window_bank_map[time][w];
                            if (bank >= 0) read_requests[bank]++;
                        }

                        auto read_bank_conflicts = *std::max_element(read_requests.begin(), read_requests.end()) - 1;
                        stats.read_bank_conflicts += read_bank_conflicts;
                        stats.cycles += read_bank_conflicts;

                        // Skip lines of zeroes
                        if (skip < LOOKAHEAD_H && check_zero_line(window_buffer[time])) {
                            skip++;
                            continue;
                        }
                        skip = 0;

                        stats.compute_cycles++;
                        stats.cycles++;

                        if (this->CHECK) {

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

                                        auto in_bits = std::get<0>(window_buffer[time][window_idx + lane]);
                                        auto time_h = std::get<1>(window_buffer[time][window_idx + lane]);
                                        auto lane_d = std::get<2>(window_buffer[time][window_idx + lane]);

                                        auto wgt_bits = weight_buffer[set][time_h][filter_idx + lane_d];

                                        output[0][filter][stride * x_window + padw_x][stride * y_window + padw_y]
                                                += in_bits * wgt_bits;

                                    } // Multiply 16 weights and 16 window values

                                } // Filter
                            } // Window
                        } //Check

                    } // Time of the buffers
                } // Filter sets
            } // Window sets


        } // Kernel sets

    }

    template <typename T>
    void DynamicTactical<T>::spatial_convolution(const base::Array<T> &act, const base::Array<T> &out_grad,
            const bank_map &act_bank_map, const bank_map &out_bank_map, uint64_t Kx, uint64_t Ky, int stride,
            int pad_x, int pad_y, conv_stats &stats, output_tensor &output) {

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &out_grad_shape = out_grad.getShape();

        // Activations
        auto act_channels = act_shape[1];
        auto Nx = out_grad_shape[2];
        auto Ny = out_grad_shape[3];

        // Output gradients
        auto out_channels = out_grad_shape[1];
        auto Ox = out_grad_shape[2];
        auto Oy = out_grad_shape[3];

        uint64_t act_zeroes = 0;
        for (int ch = 0; ch < act_channels; ++ch) {
            for (int x = 0; x < Nx; ++x) {
                for (int y = 0; y < Ny; ++y) {
                    if (act.get(0, ch, x, y) == 0) act_zeroes++;
                }
            }
        }

        uint64_t out_grad_zeroes = 0;
        for (int ch = 0; ch < out_channels; ++ch) {
            for (int x = 0; x < Ox; ++x) {
                for (int y = 0; y < Oy; ++y) {
                    if (out_grad.get(0, ch, x, y) == 0) out_grad_zeroes++;
                }
            }
        }

        bool schedule_act = act_zeroes > out_grad_zeroes;
        auto OUT_SET_SIZE = schedule_act ? N_COLUMNS : N_ROWS;
        auto ACT_SET_SIZE = schedule_act ? N_ROWS : N_COLUMNS;

        // Prepare banks
        std::vector<int> act_read_requests (BANKS, 0);
        std::vector<int> out_read_requests (BANKS, 0);

        // Stats
        stats.cycles = 0;
        stats.compute_cycles = 0;
        stats.base_compute_cycles = 0;
        stats.ideal_compute_cycles = 0;
        stats.read_bank_conflicts = 0;

        // Generate output gradients buffer
        auto spatial_pad = (uint64_t)ceil(Ox * Ox / (double)N_LANES) * N_LANES;

        auto num_out_grad_sets = (uint64_t)ceil(out_channels / (double)OUT_SET_SIZE);
        auto time_per_out_grad_channel = (uint64_t)ceil(spatial_pad / (double)N_LANES);

        std::vector<schedule_buffer> out_gradients_buffer = std::vector<schedule_buffer>(
                num_out_grad_sets, schedule_buffer(time_per_out_grad_channel,
                std::vector<value_mux>(OUT_SET_SIZE * N_LANES, std::make_tuple(0.0f, 0, 0))));

        std::vector<bank_map> out_sets_bank_map = std::vector<bank_map>(num_out_grad_sets,
                bank_map(time_per_out_grad_channel, std::vector<int>(OUT_SET_SIZE, -1)));

        std::vector<uint64_t> ideal_time_per_out_grad_channel (ceil(out_channels/(double)OUT_SET_SIZE), 0);

        int set_out = -1;
        for(int o = 0; o < out_channels; ++o) {

            if ((o % OUT_SET_SIZE) == 0)
                set_out++;

            uint64_t non_zeroes = 0;

            int index = 0;
            int time = 0;
            for (int y = 0; y < Oy; ++y) {
                for (int x = 0; x < Ox; ++x) {
                    auto out_bits = out_grad.get(0, o, x, y);
                    int pos = (o % OUT_SET_SIZE) * N_LANES + index;
                    out_gradients_buffer[set_out][time][pos] = std::make_tuple(out_bits, time, index);
                    if ((o % OUT_SET_SIZE) == 0)
                        out_sets_bank_map[set_out][time][index] = out_bank_map[pad_y + y][pad_x + x];
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

        if (!schedule_act) {
            for (auto &gradients_channel_buffer : out_gradients_buffer) {
                original_schedule(gradients_channel_buffer);
            }
        }

        for (int window = 0; window < (Kx * Ky); ++window) {
            auto x_window = window % Kx;
            auto y_window = window / Kx;

            for (int k = 0; k < act_channels; k += ACT_SET_SIZE) {

                // Generate activation buffer
                auto time_per_act_channel = (uint64_t)ceil(spatial_pad / (double)N_LANES);

                schedule_buffer activation_buffer = schedule_buffer(time_per_act_channel,
                        std::vector<value_mux>(ACT_SET_SIZE * N_LANES, std::make_tuple(0.0f, 0, 0)));

                bank_map window_bank_map = bank_map(time_per_act_channel, std::vector<int>(ACT_SET_SIZE, -1));

                uint64_t ideal_time_per_act_channel = 0;

                for(int act_channel = k; act_channel < std::min((uint64_t)k + ACT_SET_SIZE, act_channels);
                        ++act_channel) {

                    uint64_t non_zeroes = 0;

                    int index = 0;
                    int time = 0;
                    for (int y = 0; y < Oy; ++y) {
                        for (int x = 0; x < Ox; ++x) {
                            auto act_bits = act.get(0, act_channel, x_window + x * stride, y_window + y * stride);
                            int pos = (act_channel % ACT_SET_SIZE) * N_LANES + index;
                            activation_buffer[time][pos] = std::make_tuple(act_bits, time, index);
                            if (act_channel == k)
                                window_bank_map[time][index] = act_bank_map[y_window + y * stride][x_window + x * stride];
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
                if (schedule_act) original_schedule(activation_buffer);

                for (int set = 0; set < num_out_grad_sets; ++set) {

                    if (schedule_act) {
                        stats.base_compute_cycles += time_per_act_channel;
                        stats.ideal_compute_cycles += ideal_time_per_act_channel;
                    } else {
                        stats.base_compute_cycles += time_per_out_grad_channel;
                        stats.ideal_compute_cycles += ideal_time_per_out_grad_channel[set];
                    }

                    int skip = 0;
                    for (int time = 0; time < time_per_act_channel; ++time) {

                        act_read_requests = std::vector<int>(BANKS, 0);
                        out_read_requests = std::vector<int>(BANKS, 0);

                        // Activations requests
                        for (int a = 0; a < ACT_SET_SIZE; ++a) {
                            auto bank = window_bank_map[time][a];
                            if (bank >= 0) act_read_requests[bank]++;
                        }

                        // Output gradients requests
                        for (int o = 0; o < OUT_SET_SIZE; ++o) {
                            auto bank = out_sets_bank_map[set_out][time][o];
                            if (bank >= 0) out_read_requests[bank]++;
                        }

                        auto act_bank_conflicts = *std::max_element(act_read_requests.begin(),
                                act_read_requests.end()) - 1;
                        auto out_bank_conflicts = *std::max_element(out_read_requests.begin(),
                                out_read_requests.end()) - 1;
                        auto read_bank_conflicts = std::max(act_bank_conflicts, out_bank_conflicts);
                        stats.read_bank_conflicts += read_bank_conflicts;
                        stats.cycles += read_bank_conflicts;

                        // Skip lines of zeroes
                        bool zero_line = schedule_act ? check_zero_line(activation_buffer[time]) :
                                check_zero_line(out_gradients_buffer[set][time]);
                        if (skip < LOOKAHEAD_H && zero_line) {
                            skip++;
                            continue;
                        }
                        skip = 0;

                        stats.compute_cycles++;
                        stats.cycles++;

                        if (this->CHECK) {

                            for (int a = 0; a < ACT_SET_SIZE; ++a) {
                                auto act_channel_idx = a * N_LANES;
                                auto act_channel = k + a;

                                if (act_channel >= act_channels)
                                    continue;

                                for (int o = 0; o < OUT_SET_SIZE; ++o) {
                                    auto out_grad_channel_idx = o * N_LANES;
                                    auto out_grad_channel = set * OUT_SET_SIZE + o;

                                    if (out_grad_channel >= out_channels)
                                        continue;

                                    for (int lane = 0; lane < N_LANES; ++lane) {

                                        float act_bits, out_bits;
                                        if (schedule_act) {
                                            act_bits = std::get<0>(activation_buffer[time][act_channel_idx + lane]);
                                            auto time_h = std::get<1>(activation_buffer[time][act_channel_idx + lane]);
                                            auto lane_d = std::get<2>(activation_buffer[time][act_channel_idx + lane]);

                                            out_bits = std::get<0>(out_gradients_buffer[set][time_h]
                                                    [out_grad_channel_idx + lane_d]);
                                        } else {
                                            out_bits = std::get<0>(out_gradients_buffer[set][time]
                                                    [out_grad_channel_idx + lane]);
                                            auto time_h = std::get<1>(out_gradients_buffer[set][time]
                                                    [out_grad_channel_idx + lane]);
                                            auto lane_d = std::get<2>(out_gradients_buffer[set][time]
                                                    [out_grad_channel_idx + lane]);

                                            act_bits = std::get<0>(activation_buffer[time_h][act_channel_idx + lane]);
                                        }

                                        output[out_grad_channel][act_channel][x_window][y_window]
                                                += act_bits * out_bits;

                                    } // Multiply 16 output gradients and 16 activations

                                } // Output Gradients
                            } // Window
                        } // Check

                    } // Time of the buffers
                } // Output Gradients Channels sets
            } // Activations Channels sets
        } // Windows

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
                std::to_string(this->N_TILES) + "_" + this->SEARCH_SHAPE + std::to_string(mux_entries) + "(" +
                std::to_string(this->LOOKAHEAD_H) + "-" + std::to_string(this->LOOKASIDE_D) + ")" + "_B" +
                std::to_string(BANKS) + "_cycles";
        sys::Stats stats = sys::Stats(epochs, network_model.getNumLayers(), filename);
        stats.setTraining(true);

        // Forward stats
        auto fw_cycles = stats.register_uint_t("Forward cycles", 0, sys::Total);
        auto fw_compute_cycles = stats.register_uint_t("Forward compute cycles", 0, sys::Total);
        auto fw_base_compute_cycles = stats.register_uint_t("Forward base compute cycles", 0, sys::Total);
        auto fw_speedup = stats.register_double_t("Forward speedup", 0, sys::Special);
        auto fw_ideal_compute_cycles = stats.register_uint_t("Forward ideal compute cycles", 0, sys::Total);
        auto fw_exploited_sparsity = stats.register_double_t("Forward exploited sparsity", 0, sys::Special);
        auto fw_read_conflicts = stats.register_uint_t("Forward read bank conflicts", 0, sys::Total);

        // Backward stats
        auto bw_wgt_cycles = stats.register_uint_t("Backward Weights cycles", 0, sys::Total);
        auto bw_wgt_compute_cycles = stats.register_uint_t("Backward Weights compute cycles", 0, sys::Total);
        auto bw_wgt_base_compute_cycles = stats.register_uint_t("Backward Weights base compute cycles", 0, sys::Total);
        auto bw_wgt_speedup = stats.register_double_t("Backward Weights speedup", 0, sys::Special);
        auto bw_wgt_ideal_compute_cycles = stats.register_uint_t("Backward Weights ideal compute cycles", 0, sys::Total);
        auto bw_wgt_exploited_sparsity = stats.register_double_t("Backward Weights exploited sparsity", 0, sys::Special);
        auto bw_wgt_read_conflicts = stats.register_uint_t("Backward Weights read bank conflicts", 0, sys::Total);

        auto bw_in_cycles = stats.register_uint_t("Backward Input cycles", 0, sys::Total);
        auto bw_in_compute_cycles = stats.register_uint_t("Backward Input compute cycles", 0, sys::Total);
        auto bw_in_base_compute_cycles = stats.register_uint_t("Backward Input base compute cycles", 0, sys::Total);
        auto bw_in_speedup = stats.register_double_t("Backward Input speedup", 0, sys::Special);
        auto bw_in_ideal_compute_cycles = stats.register_uint_t("Backward Input ideal compute cycles", 0, sys::Total);
        auto bw_in_exploited_sparsity = stats.register_double_t("Backward Input exploited sparsity", 0, sys::Special);
        auto bw_in_read_conflicts = stats.register_uint_t("Backward Input read bank conflicts", 0, sys::Total);

        // Simulate epochs
        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states, 5);

            if(!this->QUIET) std::cout << "Starting Dynamic Tactical cycles simulation for epoch "
                                       << epoch << std::endl;

            auto num_batches = this->FAST_MODE ? 1 : network.getBatches();
            int batch;

            auto max_threads = omp_get_max_threads();
            //omp_set_num_threads(std::min(max_threads, this->N_THREADS));
            //#pragma omp parallel for private(batch)
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
                        else if (((Nx - Kx + 2 * padding)/(double)stride + 1) != Ox) act.asym_right_zero_pad(padding);
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

                    // Generate bank map
                    auto act_bank_map = map_on_chip_activations(Nx_pad, Ny_pad, Ox, stride, BANKS);

                    // Simulate: Forward convolution A * W
                    output_tensor sim_output_activations = output_tensor(1,
                            std::vector<std::vector<std::vector<double>>>(num_filters,
                            std::vector<std::vector<double>>(Ox, std::vector<double>(Oy, 0))));

                    conv_stats batch_stats;
                    channel_first_convolution(act, wgt, act_bank_map, Ox, Oy, stride, batch_stats,
                            sim_output_activations);

                    #pragma omp critical
                    {
                        fw_cycles->value[epoch][layer_it] += batch_stats.cycles;
                        fw_compute_cycles->value[epoch][layer_it] += batch_stats.compute_cycles;
                        fw_base_compute_cycles->value[epoch][layer_it] += batch_stats.base_compute_cycles;
                        fw_ideal_compute_cycles->value[epoch][layer_it] += batch_stats.ideal_compute_cycles;
                        fw_read_conflicts->value[epoch][layer_it] += batch_stats.read_bank_conflicts;
                    }

                    // Check correctness of the outputs
                    if (this->CHECK)
                        check_result_channel_first(sim_output_activations, act, wgt, Ox, Oy, stride);

                } // Forward pass

                // Backward pass
                for (int layer_it = network.getNumLayers() - 1; layer_it >= 0; layer_it--) {

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
                    else if (pad_type == 3) act.asym_right_zero_pad(padding);
                    else if (pad_type == 4) act.zero_pad(padding);

                    const std::vector<size_t> &act_shape_pad = act.getShape();

                    auto Nx_pad = act_shape_pad[2];
                    auto Ny_pad = act_shape_pad[3];

                    // Backward pass - Calculate Weight gradients
                    auto pad = (Nx_pad - Kx) % stride;
                    auto Ox_dil = Ox + (Ox - 1) * (stride - 1) + pad;
                    auto Oy_dil = Oy + (Oy - 1) * (stride - 1) + pad;

                    // Check window size
                    if ((Nx_pad - Ox_dil + 1) != Kx)
                        throw std::runtime_error("Weight gradients incorrect X window sizes");
                    if ((Ny_pad - Oy_dil + 1) != Ky)
                        throw std::runtime_error("Weight gradients incorrect Y window sizes");

                    // Generate bank map
                    int pad_x = 0, pad_y = 0;
                    if (pad_type == 1) pad_y = 2 * padding;
                    else if (pad_type == 2) pad_x = 2 * padding;
                    else if (pad_type == 3) pad_x = pad_y = 2 * padding - 1;
                    else if (pad_type == 4) pad_x = pad_y = 2 * padding;
                    else if ((Ox - Kx + 1) != Nx) pad_x = pad_y = 2 * (Kx - 1) / stride;

                    auto pad_left = pad_type == 3 ? (pad_x + 1) / 2 - 1 : pad_x / 2;
                    auto pad_top = pad_type == 3 ? (pad_y + 1) / 2 - 1 : pad_y / 2;

                    auto act_bank_map = map_on_chip_activations(Nx_pad, Ny_pad, Ox, stride, BANKS);
                    auto out_bank_map = map_on_chip_gradients(Ox + pad_x, Oy + pad_y, Nx, stride, BANKS);

                    // Simulate: Backward convolution A * G = WG: Activations sparsity
                    output_tensor sim_weight_gradients = output_tensor(num_filters,
                            std::vector<std::vector<std::vector<double>>>(wgt_channels,
                            std::vector<std::vector<double>>(Kx, std::vector<double>(Ky, 0))));

                    conv_stats batch_stats;
                    spatial_convolution(act, out_grad, act_bank_map, out_bank_map, Kx, Ky, stride, pad_left, pad_top,
                            batch_stats, sim_weight_gradients);

                    #pragma omp critical
                    {
                        bw_wgt_cycles->value[epoch][layer_it] += batch_stats.cycles;
                        bw_wgt_compute_cycles->value[epoch][layer_it] += batch_stats.compute_cycles;
                        bw_wgt_base_compute_cycles->value[epoch][layer_it] += batch_stats.base_compute_cycles;
                        bw_wgt_ideal_compute_cycles->value[epoch][layer_it] += batch_stats.ideal_compute_cycles;
                        bw_wgt_read_conflicts->value[epoch][layer_it] += batch_stats.read_bank_conflicts;
                    }

                    // Backward pass - Calculate Input gradients
                    if (layer_it == 0)
                        continue;

                    if (pad_type == 1) {
                        out_grad.zero_pad_y(padding);
                        pad_y = 2 * (padding + stride - 1);
                    } else if (pad_type == 2) {
                        out_grad.zero_pad_x(padding);
                        pad_x = 2 * (padding + stride - 1);
                    } else if (pad_type == 3) {
                        out_grad.asym_left_zero_pad(padding);
                        pad_x = pad_y = 2 * (padding + stride - 1) - 1;
                    } else if (pad_type == 4) {
                        out_grad.zero_pad(padding);
                        pad_x = pad_y = 2 * (padding + stride - 1);
                    } else if ((Ox - Kx + 1) != Nx) {
                        out_grad.zero_pad((Kx - 1) / stride);
                        pad_x = pad_y = 2 * (Kx - 1);
                    }

                    auto Ox_pad = Ox_dil + pad_x;
                    auto Oy_pad = Oy_dil + pad_y;

                    wgt.rotate_180deg();
                    wgt.reshape_channel_wise(out_channels);

                    const std::vector<size_t> &wgt_shape_rot = wgt.getShape();

                    auto num_filters_rot = wgt_shape[0];
                    auto wgt_channels_rot = wgt_shape[1];

                    // Check window size
                    if (wgt_channels_rot != out_channels)
                        throw std::runtime_error("Wrong weights rotated channels");
                    if (num_filters_rot != act_channels)
                        throw std::runtime_error("Wrong weights rotated filters");

                    // Check window size
                    if ((Ox_pad - Kx + 1) != Nx)
                        throw std::runtime_error("Input gradients incorrect X window sizes");
                    if ((Oy_pad - Ky + 1) != Ny)
                        throw std::runtime_error("Input gradients incorrect Y window sizes");

                    // Simulate: Backward convolution W * G = IG
                    output_tensor sim_input_gradients = output_tensor(1,
                            std::vector<std::vector<std::vector<double>>>(num_filters_rot,
                            std::vector<std::vector<double>>(Nx, std::vector<double>(Ny, 0))));

                    if (stride > 1)
                        channel_first_dilated_convolution(out_grad, wgt, out_bank_map, stride, pad_type == 3,
                                batch_stats, sim_input_gradients);
                    else
                        channel_first_convolution(out_grad, wgt, out_bank_map, Nx, Ny, 1, batch_stats,
                                sim_input_gradients);

                    #pragma omp critical
                    {
                        bw_in_cycles->value[epoch][layer_it] += batch_stats.cycles;
                        bw_in_compute_cycles->value[epoch][layer_it] += batch_stats.compute_cycles;
                        bw_in_base_compute_cycles->value[epoch][layer_it] += batch_stats.base_compute_cycles;
                        bw_in_ideal_compute_cycles->value[epoch][layer_it] += batch_stats.ideal_compute_cycles;
                        bw_in_read_conflicts->value[epoch][layer_it] += batch_stats.read_bank_conflicts;
                    }

                    // Check correctness of the outputs
                    if (this->CHECK) {

                        // Reload output gradients without padding
                        out_grad = layer.getOutputGradients();
                        if (fc) out_grad.reshape_to_4D();
                        out_grad.get_batch(batch);

                        // Check weight gradients
                        if (conv) out_grad.dilate_out_grad(stride, Nx_pad, Kx);
                        check_result_spatial(sim_weight_gradients, act, out_grad, num_filters, Kx, Ky, wgt_channels);

                        // Check input gradients
                        if (pad_type == 1) out_grad.zero_pad_y(padding + stride - 1);
                        else if (pad_type == 2) out_grad.zero_pad_x(padding + stride - 1);
                        else if (pad_type == 3) out_grad.asym_left_zero_pad(padding + stride - 1);
                        else if (pad_type == 4) out_grad.zero_pad(padding + stride - 1);
                        else if ((Ox - Kx + 1) != Nx) out_grad.zero_pad(Kx - 1);
                        check_result_channel_first(sim_input_gradients, out_grad, wgt, Nx, Ny, 1);

                    }

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

                bw_wgt_speedup->value[epoch][layer_it] = bw_wgt_base_compute_cycles->value[epoch][layer_it] /
                        (double)bw_wgt_compute_cycles->value[epoch][layer_it];
                bw_wgt_exploited_sparsity->value[epoch][layer_it] =
                        (bw_wgt_base_compute_cycles->value[epoch][layer_it] -
                         bw_wgt_compute_cycles->value[epoch][layer_it])
                        / (double)(bw_wgt_base_compute_cycles->value[epoch][layer_it] -
                        bw_wgt_ideal_compute_cycles->value[epoch][layer_it]) * 100.;
                if (isnan(bw_wgt_exploited_sparsity->value[epoch][layer_it]))
                    bw_wgt_exploited_sparsity->value[epoch][layer_it] = 0;

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

            bw_wgt_speedup->special_value_vector.push_back(
                    sys::get_total(bw_wgt_base_compute_cycles->value[epoch]) /
                    (double)sys::get_total(bw_wgt_compute_cycles->value[epoch]));
            bw_wgt_exploited_sparsity->special_value_vector.push_back(
                    (sys::get_total(bw_wgt_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_wgt_compute_cycles->value[epoch])) /
                    (double)(sys::get_total(bw_wgt_base_compute_cycles->value[epoch]) -
                    sys::get_total(bw_wgt_ideal_compute_cycles->value[epoch])) * 100.);

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
        fw_exploited_sparsity->special_value = (sys::get_total(fw_base_compute_cycles->value) -
                sys::get_total(fw_compute_cycles->value)) /
                (double)(sys::get_total(fw_base_compute_cycles->value) -
                sys::get_total(fw_ideal_compute_cycles->value)) * 100.;

        bw_wgt_speedup->special_value = sys::get_total(bw_wgt_base_compute_cycles->value) /
                (double)sys::get_total(bw_wgt_compute_cycles->value);
        bw_wgt_exploited_sparsity->special_value = (sys::get_total(bw_wgt_base_compute_cycles->value) -
                sys::get_total(bw_wgt_compute_cycles->value)) /
                (double)(sys::get_total(bw_wgt_base_compute_cycles->value) -
                sys::get_total(bw_wgt_ideal_compute_cycles->value)) * 100.;

        bw_in_speedup->special_value = sys::get_total(bw_in_base_compute_cycles->value) /
                (double)sys::get_total(bw_in_compute_cycles->value);
        bw_in_exploited_sparsity->special_value = (sys::get_total(bw_in_base_compute_cycles->value) -
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
        header += "Number of banks: " + std::to_string(BANKS) + "\n";

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

                if (conv && padding > 0) asym_pad ? act.asym_right_zero_pad(padding) : act.zero_pad(padding);

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
                if (conv && padding > 0) asym_pad ? out_grad.asym_left_zero_pad(padding + stride - 1) :
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
