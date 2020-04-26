
#include <core/WindowFirstOutS.h>

namespace core {

    /* CYCLES */

    template <typename T>
    std::string WindowFirstOutS<T>::dataflow() {
        return "Window First Output Stationary";
    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph_conv_layer() {

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        auto last_act_index = (int)ceil(act_channels / (double)this->dram->getValuesPerBlock()) - 1;

        // Try to fit whole layer
        auto all_input = act_channels * Nx * Ny * this->dram->getDataSize();
        auto all_output = num_filters * this->out_x * this->out_y * this->dram->getDataSize();
        auto extra_rows = ceil(this->EF_COLUMNS / (double)this->out_x) - 1;

        auto working_set_windows = std::min(this->out_x * this->out_y, (int)this->EF_COLUMNS);
        auto worst_case_window_set = act_channels * Kx * Ky * working_set_windows * this->dram->getDataSize();

        auto all_filters = num_filters * wgt_channels * Kx * Ky * this->dram->getDataSize();

        auto working_set_filters = std::min((uint32_t)num_filters, this->EF_ROWS * this->arch->getTiles());
        auto filter_set = wgt_channels * Kx * Ky * working_set_filters * this->dram->getDataSize();

        // Check if all filters fit on-chip
        if (all_filters < this->gbuffer->getWgtSize()) {

            auto window_set_output = num_filters * this->dram->getDataSize();

            // Check if all activations fit on-chip
            if (all_input + all_output < this->gbuffer->getActSize()) {

            }

            // Check if set of windows fit on-chip
            else if (worst_case_window_set + window_set_output < this->gbuffer->getActSize()) {

            }

            // Check if a subset of channels fit on-chip
            else {

            }

        }

        // Check if a set of filters fit on-chip
        else if (filter_set < this->gbuffer->getWgtSize()) {

            auto window_set_output = working_set_filters * this->dram->getDataSize();

            // Check if all activations fit on-chip
            if (all_input + all_output < this->gbuffer->getActSize()) {

            }

            // Check if set of windows fit on-chip
            else if (worst_case_window_set + window_set_output < this->gbuffer->getActSize()) {

            }

            // Check if a subset of channels fit on-chip
            else {

            }

        }

        // Check if a subset of channels fit on-chip
        else {

            auto window_set_output = working_set_filters * this->dram->getDataSize();

            auto total_time_size = this->max_buffer_time * this->EF_LANES * this->dram->getDataSize();
            auto time_steps = (uint64_t)ceil(total_time_size / (double)this->gbuffer->getWgtSize());
            assert(time_steps != 1);

            // Check if all activations fit on-chip
            if (all_input + all_output < this->gbuffer->getActSize()) {

            }

            // Check if set of windows fit on-chip
            else if (worst_case_window_set + window_set_output < this->gbuffer->getActSize()) {

            }

            // Check if a subset of channels fit on-chip
            else {

            }

        }

    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph_linear_layer() {

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto recurrences = this->lstm ? act_shape[0] : 1;
        auto act_channels = this->lstm ? act_shape[2] : act_shape[1];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];

        auto last_act_index = (int)ceil(act_channels / (double)this->dram->getValuesPerBlock()) - 1;

        auto all_input_size = act_channels * this->dram->getDataSize();
        auto all_output_size = num_filters * this->dram->getDataSize();

        auto all_filters_size = num_filters * wgt_channels * this->dram->getDataSize();

        auto working_set_filters = std::min((uint32_t)num_filters, this->EF_ROWS * this->arch->getTiles());
        auto filter_set_size = wgt_channels * working_set_filters * this->dram->getDataSize();

        // Check if all filters fit on-chip
        if (all_filters_size < this->gbuffer->getWgtSize()) {

            // Check if all activations fit on-chip
            if (all_input_size + all_output_size < this->gbuffer->getActSize()) {

                this->next_layer_act_on_chip = true;
                this->on_chip_graph = std::vector<std::shared_ptr<typename Control<T>::Node>>(recurrences);

                for (int r = 0; r < recurrences; ++r) {

                    auto node = std::make_shared<typename OutputStationary<T>::NodeOutS>();

                    // Fill parameters
                    node->time_step = 0;
                    node->max_time = this->max_buffer_time;
                    node->recurrence = r;

                    // Fil filters
                    node->filter_sets = std::vector<int>(this->filter_sets, 0);
                    std::iota(node->filter_sets.begin(), node->filter_sets.end(), 0);

                    // First read activations
                    if (r == 0) {
                        if (!this->layer_act_on_chip) {
                            auto first_address = this->act_address_map[0][0][0];
                            auto last_address = this->act_address_map[0][0][last_act_index];
                            node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        }

                        auto first_address = std::get<0>(this->wgt_address_map.front());
                        auto last_address = std::get<1>(this->wgt_address_map.back());
                        node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                    }

                    this->on_chip_graph[r] = node;

                }

            } // all activations fit on-chip

            // Check if a subset of channels fit on-chip
            else {
                throw std::runtime_error("TODO");
            }

        }

        // Check if a set of filters fit on-chip
        else if (filter_set_size < this->gbuffer->getWgtSize()) {

            // Check if all activations fit on-chip
            if (all_input_size + all_output_size < this->gbuffer->getActSize()) {

                // TODO Fix this for fc8
                auto filter_sets_per_step = this->gbuffer->getWgtSize() / filter_set_size;
                auto total_filter_sets = ceil(num_filters / (double)working_set_filters);
                auto filter_steps = ceil(total_filter_sets / (double)filter_sets_per_step);
                assert(filter_steps != 1);

                this->next_layer_act_on_chip = true;
                this->on_chip_graph = std::vector<std::shared_ptr<typename Control<T>::Node>>(recurrences * filter_steps);

                for (int r = 0; r < recurrences; ++r) {
                    for (int fstep = 0; fstep < filter_steps; ++fstep) {

                        auto node = std::make_shared<typename OutputStationary<T>::NodeOutS>();

                        node->time_step = 0;
                        node->max_time = this->max_buffer_time;
                        node->recurrence = r;

                        // Fil filters
                        auto start_filter_set = fstep * filter_sets_per_step * this->arch->getTiles();
                        auto last_filter_set = std::min(filter_sets_per_step * this->arch->getTiles(),
                                this->filter_sets - start_filter_set);
                        node->filter_sets = std::vector<int>(last_filter_set, 0);
                        std::iota(node->filter_sets.begin(), node->filter_sets.end(),  start_filter_set);

                        // First read activations
                        if (r == 0 && fstep == 0 && !this->layer_act_on_chip) {
                            auto first_address = this->act_address_map[0][0][0];
                            auto last_address = this->act_address_map[0][0][last_act_index];
                            node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        }

                        auto first_address = std::get<0>(this->wgt_address_map[start_filter_set]);
                        auto last_address = std::get<1>(this->wgt_address_map[start_filter_set + last_filter_set - 1]);
                        node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));

                        if (r != 0 || fstep != 0)
                            node->evict_wgt = true;

                        this->on_chip_graph[r * filter_steps + fstep] = node;

                    }
                }


            } // all activations fit on-chip

            // Check if a subset of channels fit on-chip
            else {
                throw std::runtime_error("TODO");
            }

        }

        // Check if a subset of channels fit on-chip
        else {

            // Check if all activations fit on-chip
            if (all_input_size + all_output_size < this->gbuffer->getActSize()) {

                auto time_steps = ceil(filter_set_size / (double)this->gbuffer->getWgtSize());
                auto filter_steps = ceil(num_filters / (double)working_set_filters);
                auto max_time_per_step = this->max_buffer_time / time_steps;
                assert(time_steps != 1);

                this->next_layer_act_on_chip = true;
                this->on_chip_graph = std::vector<std::shared_ptr<typename Control<T>::Node>>(recurrences * filter_steps
                        * time_steps);

                for (int r = 0; r < recurrences; ++r) {
                    for (int fstep = 0; fstep < filter_steps; ++fstep) {
                        for (int tstep = 0; tstep < time_steps; ++tstep) {

                            auto node = std::make_shared<typename OutputStationary<T>::NodeOutS>();

                            node->time_step = tstep;
                            node->max_time = max_time_per_step;
                            node->recurrence = r;

                            // Fil filters
                            auto start_filter_set = fstep * this->arch->getTiles();
                            auto last_filter_set = std::min((uint64_t)this->arch->getTiles(),
                                    this->filter_sets - start_filter_set);
                            auto end_filter_set = start_filter_set + last_filter_set - 1;
                            node->filter_sets = std::vector<int>(last_filter_set, 0);
                            std::iota(node->filter_sets.begin(), node->filter_sets.end(),  start_filter_set);

                            // First read activations
                            if (r == 0 && fstep == 0 && tstep == 0 && !this->layer_act_on_chip) {
                                auto first_address = this->act_address_map[0][0][0];
                                auto last_address = this->act_address_map[0][0][last_act_index];
                                node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                            }

                            auto start_time = tstep * max_time_per_step;
                            auto end_time = std::min((uint64_t)((tstep + 1) * max_time_per_step),
                                    this->max_buffer_time) - 1;

                            uint64_t first_address = NULL_ADDR, last_address = NULL_ADDR;
                            // TODO Fix tactical
                            if (this->arch->schedule()) {
                                for (int tmp = start_time; tmp <= end_time; ++tmp) {
                                    if (this->wgt_address_buffer[start_filter_set][tmp].front() != NULL_ADDR) {
                                        first_address = this->wgt_address_buffer[start_filter_set][tmp].front();
                                        break;
                                    }
                                }

                                for (int tmp = end_time; tmp >= start_time ; --tmp) {
                                    if (this->wgt_address_buffer[end_filter_set][tmp].back() != NULL_ADDR) {
                                        last_address = this->wgt_address_buffer[end_filter_set] [tmp].back();
                                        break;
                                    }
                                }

                            } else {
                                first_address = this->wgt_address_buffer[start_filter_set][start_time].front();
                                last_address = this->wgt_address_buffer[end_filter_set][end_time].back();
                            }

                            assert(first_address != NULL_ADDR);
                            assert(last_address != NULL_ADDR);

                            node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));

                            if (r != 0 || fstep != 0 || tstep != 0)
                                node->evict_wgt = true;

                            this->on_chip_graph[r * filter_steps * time_steps + fstep * time_steps + tstep] = node;

                        }
                    }
                }

            } // all activations fit on-chip

            // Check if a subset of channels fit on-chip
            else {
                throw std::runtime_error("TODO");
            }

        }

    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph() {
        if (this->linear) generate_execution_graph_linear_layer();
        else generate_execution_graph_conv_layer();
    }

    template <typename T>
    void WindowFirstOutS<T>::configure_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, uint32_t act_prec, uint32_t wgt_prec, bool _linear,
            bool _lstm, int _stride) {

        OutputStationary<T>::configure_layer(_act, _wgt, act_prec, wgt_prec, _linear, _lstm, _stride);

        // Generate off-chip addresses and bank mapping
        this->generate_memory_maps();

        // Generate execution graph for on-chip memory
        generate_execution_graph();

    }

    template <typename T>
    bool WindowFirstOutS<T>::still_on_chip_data_conv_layer(std::vector<core::TileData<T>> &tiles_data) {

        // Select values from current node
        const auto &current_node = std::static_pointer_cast<typename OutputStationary<T>::NodeOutS>
                (this->on_chip_graph.front());
        const auto &window_sets = current_node->window_sets;
        const auto &filter_tile_sets = current_node->filter_sets;
        const auto &time_step = current_node->time_step;
        const auto &max_time = current_node->max_time;

        while (this->window_set_it < window_sets.size()) {

            // Fill window buffer
            if (!this->window_buffer_filled) {

                // Select windows
                auto window_idx = window_sets[this->window_set_it] * this->EF_COLUMNS;
                for (int c = 0; c < this->EF_COLUMNS; ++c) {

                    auto window = window_idx + c;
                    if (window >= (this->out_x * this->out_y))
                        continue;

                    auto x_window = window % this->out_x;
                    auto y_window = window / this->out_y;
                    this->windows.emplace_back(std::make_tuple(x_window, y_window));
                }

                this->fill_window_buffer();
                this->window_buffer_filled = true;
            }

            while (this->filter_set_it < filter_tile_sets.size()) {

                auto filter_set = filter_tile_sets[this->filter_set_it];

                // Filter set
                if (!this->filter_buffer_filled) {

                    this->filters = std::vector<std::vector<int>>(this->arch->getTiles(), std::vector<int>());

                    // Select filter for each tile
                    for (int t = 0; t < this->arch->getTiles(); ++t) {

                        auto filter_idx = (filter_set + t) * this->EF_ROWS;

                        auto num_filters = this->wgt->getShape()[0];
                        for (int r = 0; r < this->EF_ROWS; ++r) {
                            auto filter = filter_idx + r;
                            if (filter >= num_filters)
                                continue;
                            this->filters[t].push_back(filter);
                        }

                    }

                    this->filter_buffer_filled = true;
                }

                bool still_work = false;
                for (int t = 0; t < this->arch->getTiles(); ++t) {

                    tiles_data[t].valid = false;

                    if (this->filters[t].empty()) break;

                    while (this->time[t] < max_time) {
                        auto set_time = time_step * max_time + this->time[t];

                        if (set_time > this->max_buffer_time)
                            continue;

                        if (this->arch->schedule()) {

                            // Skip lines of zeroes
                            bool zero_line = this->scheduler->check_zero_line(this->weight_buffer
                                    [filter_set + t][set_time]);
                            if (this->skip[t] < this->scheduler->getLookaheadH() && zero_line) {
                                this->skip[t]++;
                                this->time[t]++;
                                continue;
                            }
                            this->skip[t] = 0;

                        }

                        auto num_act_rows = 1;
                        if (this->arch->schedule()) num_act_rows += this->scheduler->getLookaheadH();
                        tiles_data[t].act_row = BufferSet<T>(this->window_buffer.begin() + set_time,
                                std::min(this->window_buffer.begin() + set_time +
                                num_act_rows, this->window_buffer.end()));
                        if (t == 0) {
                            if (!this->layer_act_on_chip) {
                                tiles_data[t].act_addresses =
                                        AddressBufferSet(this->window_address_buffer.begin() + set_time,
                                        std::min(this->window_address_buffer.begin() + set_time +
                                        num_act_rows, this->window_address_buffer.end()));
                            }
                            tiles_data[t].act_banks = this->window_bank_buffer[set_time];
                        }

                        tiles_data[t].wgt_row = this->weight_buffer[filter_set + t][set_time];
                        tiles_data[t].wgt_addresses = this->wgt_address_buffer[filter_set + t][set_time];
                        tiles_data[t].wgt_banks = this->wgt_bank_buffer[filter_set + t][set_time];

                        tiles_data[t].windows = this->windows;
                        tiles_data[t].filters = this->filters[t];
                        tiles_data[t].time = set_time;
                        tiles_data[t].lanes = this->EF_LANES;
                        tiles_data[t].valid = true;

                        still_work = true;
                        this->time[t]++;
                        break;

                    } // Buffer time

                } // Tile

                if (still_work) return true;

                this->time = std::vector<int>(this->arch->getTiles(), 0);
                this->skip = std::vector<int>(this->arch->getTiles(), 0);
                this->filter_buffer_filled = false;
                this->filters.clear();
                this->filter_set_it += this->arch->getTiles();
            } // Filter set

            this->filter_set_it = 0;
            this->window_buffer_filled = false;
            this->windows.clear();
            this->window_set_it++;
        } // Window set

        this->window_set_it = 0;
        this->filter_set_it = 0;
        this->time = std::vector<int>(this->arch->getTiles(), 0);
        this->skip = std::vector<int>(this->arch->getTiles(), 0);
        this->window_buffer_filled = false;
        this->filter_buffer_filled = false;
        return false;

    }

    template <typename T>
    bool WindowFirstOutS<T>::still_on_chip_data_linear_layer(std::vector<core::TileData<T>> &tiles_data) {

        // Select values from current node
        const auto &current_node = std::static_pointer_cast<typename OutputStationary<T>::NodeOutS>
                (this->on_chip_graph.front());
        const auto &recurrence = current_node->recurrence;
        const auto &filter_tile_sets = current_node->filter_sets;
        const auto &time_step = current_node->time_step;
        const auto &max_time = current_node->max_time;

        // Fill window buffer
        if (!this->window_buffer_filled) { // Avoid filling the buffer for subset of channels
            this->windows = {std::make_tuple(0, 0)};
            this->fill_window_buffer();
            this->window_buffer_filled = true;
        }

        while (this->filter_set_it < filter_tile_sets.size()) {

            auto filter_set = filter_tile_sets[this->filter_set_it];

            // Filter set
            if (!this->filter_buffer_filled) {

                this->filters = std::vector<std::vector<int>>(this->arch->getTiles(), std::vector<int>());

                // Select filter for each tile
                for (int t = 0; t < this->arch->getTiles(); ++t) {

                    auto filter_idx = (filter_set + t) * this->EF_ROWS;

                    auto num_filters = this->wgt->getShape()[0];
                    for (int r = 0; r < this->EF_ROWS; ++r) {
                        auto filter = filter_idx + r;
                        if (filter >= num_filters)
                            continue;
                        this->filters[t].push_back(filter);
                    }

                }

                this->filter_buffer_filled = true;
            }

            bool still_work = false;
            for (int t = 0; t < this->arch->getTiles(); ++t) {

                tiles_data[t].valid = false;

                if (this->filters[t].empty()) break;

                while (this->time[t] < max_time) {
                    auto set_time = time_step * max_time + this->time[t];

                    if (set_time > this->max_buffer_time)
                        continue;

                    if (this->arch->schedule()) {

                        // Skip lines of zeroes
                        bool zero_line = this->scheduler->check_zero_line(this->weight_buffer[filter_set + t][set_time]);
                        if (this->skip[t] < this->scheduler->getLookaheadH() && zero_line) {
                            this->skip[t]++;
                            this->time[t]++;
                            continue;
                        }
                        this->skip[t] = 0;

                    }

                    auto num_act_rows = 1;
                    if (this->arch->schedule()) num_act_rows += this->scheduler->getLookaheadH();
                    tiles_data[t].act_row = BufferSet<T>(this->window_buffer.begin() + set_time,
                            std::min(this->window_buffer.begin() + set_time +
                            num_act_rows, this->window_buffer.end()));
                    if (t == 0) {
                        if (!this->layer_act_on_chip) {
                            tiles_data[t].act_addresses =
                                    AddressBufferSet(this->window_address_buffer.begin() + set_time,
                                    std::min(this->window_address_buffer.begin() + set_time +
                                    num_act_rows, this->window_address_buffer.end()));
                            }
                        tiles_data[t].act_banks = this->window_bank_buffer[set_time];
                    }

                    tiles_data[t].wgt_row = this->weight_buffer[filter_set + t][set_time];
                    tiles_data[t].wgt_addresses = this->wgt_address_buffer[filter_set + t][set_time];
                    tiles_data[t].wgt_banks = this->wgt_bank_buffer[filter_set + t][set_time];

                    tiles_data[t].windows = std::vector<WindowCoord>(this->EF_COLUMNS, std::make_tuple(0, 0));
                    tiles_data[t].filters = this->filters[t];
                    tiles_data[t].time = set_time;
                    tiles_data[t].lanes = this->EF_LANES;
                    tiles_data[t].valid = true;

                    still_work = true;
                    this->time[t]++;
                    break;

                } // Buffer time

            } // Tile

            if (still_work) return true;

            this->time = std::vector<int>(this->arch->getTiles(), 0);
            this->skip = std::vector<int>(this->arch->getTiles(), 0);
            this->filter_buffer_filled = false;
            this->filters.clear();
            this->filter_set_it += this->arch->getTiles();
        } // Filter set

        this->window_set_it = 0;
        this->filter_set_it = 0;
        this->time = std::vector<int>(this->arch->getTiles(), 0);
        this->skip = std::vector<int>(this->arch->getTiles(), 0);
        this->window_buffer_filled = false;
        this->filter_buffer_filled = false;
        return false;

    }

    template <typename T>
    bool WindowFirstOutS<T>::still_on_chip_data(std::vector<core::TileData<T>> &tiles_data) {
        if (this->linear) return still_on_chip_data_linear_layer(tiles_data);
        else return still_on_chip_data_conv_layer(tiles_data);
    }

    INITIALISE_DATA_TYPES(WindowFirstOutS);

}
