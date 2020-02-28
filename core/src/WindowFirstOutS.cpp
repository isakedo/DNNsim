
#include <core/WindowFirstOutS.h>

namespace core {

    /* CYCLES */

    template <typename T>
    std::string WindowFirstOutS<T>::name() {
        return "Window First Output Stationary";
    }

    int calculate_num_act_rows(std::vector<std::vector<int>> &window_sets,
            std::vector<std::tuple<int, int>> &read_act_rows, std::vector<std::tuple<int, int>> &clean_act_rows) {

        throw std::runtime_error("TODO");

        return window_sets.size();
    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph_conv_layer() {

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        uint64_t act_channels, Nx, Ny;
        if (this->lstm) {
            act_channels = act_shape[2];
            Nx = 1;
            Ny = 1;
        } else {
            act_channels = act_shape[1];
            Nx = act_shape[2];
            Ny = act_shape[3];
        }

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        auto last_act_index = (int)ceil(act_channels / (double)this->dram->getValuesPerBlock()) - 1;

        // Try to fit whole layer
        auto all_windows = act_channels * Nx * Ny * this->dram->getDataSize();
        auto all_filters = num_filters * wgt_channels * Kx * Ky * this->dram->getDataSize();
        auto all_output = num_filters * this->out_x * this->out_y * this->dram->getDataSize();
        auto extra_rows = ceil(this->EF_COLUMNS / (double)this->out_x) - 1;
        if (all_windows + all_filters + all_output < this->gbuffer->getSize()) {

            this->next_layer_act_on_chip = true;
            typename OutputStationary<T>::NodeOutS unique_node;

            // Fill parameters
            unique_node.start_channel = 0;
            unique_node.end_channel = act_channels;

            // Fill windows
            unique_node.window_sets = std::vector<int>(this->window_sets, 0);
            std::iota(unique_node.window_sets.begin(), unique_node.window_sets.end(), 0);

            // Fil filters
            unique_node.filter_sets = std::vector<int>(this->filter_sets, 0);
            std::iota(unique_node.filter_sets.begin(), unique_node.filter_sets.end(), 0);

            unique_node.read_addresses = std::vector<AddressRange>(3);

            // First read fist row of activations
            if (!this->layer_act_on_chip) {
                auto first_address = this->act_address_map[0][0][0];
                auto last_address = this->act_address_map[Ky + extra_rows - 1][Nx - 1][last_act_index];
                unique_node.read_addresses[0] = std::make_tuple(first_address, last_address);
            }

            // Then read all filters
            auto first_address = this->wgt_address_map.first_address;
            auto last_address = this->wgt_address_map.last_address;
            unique_node.read_addresses[1] = std::make_tuple(first_address, last_address);

            // Finally read remaining activations
            if (!this->layer_act_on_chip) {
                first_address = this->act_address_map[Ky + extra_rows][0][0];
                last_address = this->act_address_map[Ny - 1][Nx - 1][last_act_index];
                unique_node.read_addresses[2] = std::make_tuple(first_address, last_address);
            }

            this->on_chip_graph.emplace_back(std::make_shared<typename OutputStationary<T>::NodeOutS>(unique_node));

            return;
        }

        // Try to fit one row of activations and all filters
        auto act_row = act_channels * Nx * (Ky + extra_rows) * this->dram->getDataSize();
        if (act_row + all_filters < this->gbuffer->getSize()) {

            // Calculate number of activation rows
            std::vector<std::vector<int>> tmp_window_sets;
            std::vector<std::tuple<int, int>> tmp_read_act_rows;
            std::vector<std::tuple<int, int>> tmp_clean_act_rows;
            auto num_act_rows = calculate_num_act_rows(tmp_window_sets, tmp_read_act_rows, tmp_clean_act_rows);

            this->on_chip_graph = std::vector<std::shared_ptr<typename Control<T>::Node>>(num_act_rows);

            // Fill nodes
            for (int r = 0; r < num_act_rows; ++r) {

                // Fill parameters
                //this->on_chip_graph[r].start_channel = 0;
                //this->on_chip_graph[r].end_channel = act_channels;

                // Fil windows
                //this->on_chip_graph[r].window_sets = tmp_window_sets[r];

                // Fil filters
                //this->on_chip_graph[r].filter_sets = std::vector<int>(this->filter_sets, 0);
                //std::iota(this->on_chip_graph[r].filter_sets.begin(), this->on_chip_graph[r].filter_sets.end(), 0);

                // Read row of activations
                auto start_act_row = std::get<0>(tmp_read_act_rows[r]);
                auto end_act_row = std::min(std::get<1>(tmp_read_act_rows[r]), (int)Ny);

                if (start_act_row != -1) {
                    auto first_address = this->act_address_map[start_act_row][0][0];
                    auto last_address = this->act_address_map[end_act_row - 1][Nx - 1][last_act_index];
                    //this->on_chip_graph[r].read_addresses.emplace_back(std::make_tuple(first_address, last_address));
                }

                // Read all filters for first node
                if (r == 0) {
                    //auto first_address = this->wgt_address_map[0][0][0][0];
                    //auto last_address = this->wgt_address_map[num_filters - 1][Ky - 1][Kx - 1][last_wgt_index];
                    //this->on_chip_graph[r].read_addresses.emplace_back(std::make_tuple(first_address, last_address));
                }

                // Clean old row of activations
                start_act_row = std::get<0>(tmp_clean_act_rows[r]);
                end_act_row = std::get<1>(tmp_clean_act_rows[r]);

                if (start_act_row != -1) {
                    auto first_address = this->act_address_map[start_act_row][0][0];
                    auto last_address = this->act_address_map[end_act_row - 1][Nx - 1][last_act_index];
                    //this->on_chip_graph[r].clean_addresses.emplace_back(std::make_tuple(first_address, last_address));
                }

            }

            return;
        }

        // Try to fit one row of activations and one set of filters
        auto working_set_filters = std::min((uint32_t)num_filters, this->EF_ROWS * this->arch->getNTiles());
        auto filter_set = wgt_channels * Kx * Ky * working_set_filters * this->dram->getDataSize();
        if (act_row + filter_set < this->gbuffer->getSize()) {

            auto tile_filter_sets = (uint64_t)ceil(this->filter_sets / (double)this->arch->getNTiles());

            // Calculate number of activation rows
            std::vector<std::vector<int>> tmp_window_sets;
            std::vector<std::tuple<int, int>> tmp_read_act_rows;
            std::vector<std::tuple<int, int>> tmp_clean_act_rows;
            auto num_act_rows = calculate_num_act_rows(tmp_window_sets, tmp_read_act_rows, tmp_clean_act_rows);

            auto num_nodes = num_act_rows * tile_filter_sets;
            //this->on_chip_graph = std::vector<typename OutputStationary<T>::NodeOutS>(num_nodes);

            // Fill nodes
            int idx = 0, prev_f = - 1;
            for (int r = 0; r < num_act_rows; ++r) {

                // Read row of activations
                auto start_act_row = std::get<0>(tmp_read_act_rows[r]);
                auto end_act_row = std::min(std::get<1>(tmp_read_act_rows[r]), (int)Ny);

                if (start_act_row != -1) {
                    auto first_address = this->act_address_map[start_act_row][0][0];
                    auto last_address = this->act_address_map[end_act_row - 1][Nx - 1][last_act_index];
                    //this->on_chip_graph[idx].read_addresses.emplace_back(std::make_tuple(first_address, last_address));
                }

                // Clean old row of activations
                start_act_row = std::get<0>(tmp_clean_act_rows[r]);
                end_act_row = std::get<1>(tmp_clean_act_rows[r]);

                if (start_act_row != -1) {
                    auto first_address = this->act_address_map[start_act_row][0][0];
                    auto last_address = this->act_address_map[end_act_row - 1][Nx - 1][last_act_index];
                    //this->on_chip_graph[idx].clean_addresses.emplace_back(std::make_tuple(first_address, last_address));
                }

                for (int f = 0; f < tile_filter_sets; ++f) {

                    // Fill parameters
                    //this->on_chip_graph[idx].start_channel = 0;
                    //this->on_chip_graph[idx].end_channel = act_channels;

                    // Fil windows
                    //this->on_chip_graph[idx].window_sets = tmp_window_sets[r];

                    // Fill filters
                    //this->on_chip_graph[idx].filter_sets.push_back(f);

                    // Read new filters
                    auto last_filter = std::min((f + 1) * this->EF_ROWS, (uint32_t)num_filters);
                    //auto first_address = this->wgt_address_map[f * this->N_ROWS][0][0][0];
                    //auto last_address = this->wgt_address_map[last_filter - 1][Ky - 1][Kx - 1][last_wgt_index];
                    //this->on_chip_graph[idx].read_addresses.emplace_back(std::make_tuple(first_address, last_address));

                    // Clean old filters
                    if (prev_f != -1) {
                        last_filter = std::min((prev_f + 1) * this->EF_ROWS, (uint32_t)num_filters);
                        //first_address = this->wgt_address_map[prev_f * this->N_ROWS][0][0][0];
                        //last_address = this->wgt_address_map[last_filter - 1][Ky - 1][Kx - 1][last_wgt_index];
                        //this->on_chip_graph[idx].clean_addresses.emplace_back(std::make_tuple(first_address, last_address));
                    }

                    prev_f = f;
                    idx++;
                }

            }

            return;
        }

        // Try to fit one set of window and one set of filters
        auto working_set_windows = std::min(this->out_x * this->out_y, (int)this->EF_COLUMNS);
        auto worst_case_window_set = act_channels * Kx * Ky * working_set_windows * this->dram->getDataSize();
        if (worst_case_window_set + filter_set < this->gbuffer->getSize()) {

            int idx = 0, prev_w = -1, prev_f = - 1;
            for (int w = 0; w < this->window_sets; ++w) {

                throw std::runtime_error("TODO");

                /*for (int f = 0; f < filter_tile_sets; ++f) {

                    // Fill parameters
                    this->on_chip_graph[idx].start_channel = 0;
                    this->on_chip_graph[idx].end_channel = act_channels;

                    // Fil windows
                    this->on_chip_graph[idx].window_sets.push_back(w);

                    // Fill filters
                    this->on_chip_graph[idx].filter_tile_sets.push_back(f);

                    // Read new filters
                    auto last_filter = std::min((f + 1) * this->N_ROWS, (uint32_t)num_filters);
                    auto first_address = this->wgt_address_map[f * this->N_ROWS][0][0][0];
                    auto last_address = this->wgt_address_map[last_filter - 1][Ky - 1][Kx - 1][wgt_channels - 1];
                    this->on_chip_graph[idx].read_addresses.emplace_back(std::make_tuple(first_address, last_address));

                    // Clean old filters
                    if (prev_f != -1) {
                        last_filter = std::min((prev_f + 1) * this->N_ROWS, (uint32_t)num_filters);
                        first_address = this->wgt_address_map[prev_f * this->N_ROWS][0][0][0];
                        last_address = this->wgt_address_map[last_filter - 1][Ky - 1][Kx - 1][wgt_channels - 1];
                        this->on_chip_graph[idx].clean_addresses.emplace_back(std::make_tuple(first_address, last_address));
                    }

                    prev_f = f;
                    idx++;
                }

                prev_w = w;*/
            }


            return;
        }

        // Try to fit one channel of filters and windows
        auto one_window_channel_set = act_channels * working_set_windows * this->dram->getDataSize();
        auto one_filter_channel_set = wgt_channels * working_set_filters * this->dram->getDataSize();
        if (one_window_channel_set + one_filter_channel_set < this->gbuffer->getSize()) {

            throw std::runtime_error("TODO");

            return;
        }

        throw std::runtime_error("Error allocating layer on-chip: Global buffer is too small");

    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph_linear_layer() {

        throw std::runtime_error("TODO");

        // Try to fit some values on chip
        /*auto act_channels_on_chip = act_channels;
        auto wgt_channels_on_chip = wgt_channels;
        while (true) {

            act_channels_on_chip = ceil(act_channels_on_chip / 2.);
            wgt_channels_on_chip = ceil(wgt_channels_on_chip / 2.);

            if (act_channels_on_chip < this->N_LANES)
            throw std::runtime_error("Error allocating layer on-chip: Global buffer is too small");

            one_window_channel_set = act_channels_on_chip * working_set_windows * this->data_size;
            one_filter_channel_set = wgt_channels_on_chip * working_set_filters * this->data_size;

            if ((one_window_channel_set + one_filter_channel_set) < this->global_buffer_size) {
            this->on_chip_policy = OutputStationary<T>::SUBSET_VALUES;

                throw std::runtime_error("TODO");

                return;
            }

        }*/

    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph() {
        if (this->linear) generate_execution_graph_linear_layer();
        else generate_execution_graph_conv_layer();
    }

    template <typename T>
    void WindowFirstOutS<T>::configure_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, bool _linear, bool _lstm, int _stride, uint32_t _EF_COLUMNS,
            uint32_t _EF_ROWS) {

        OutputStationary<T>::configure_layer(_act, _wgt, _linear, _lstm, _stride, _EF_COLUMNS, _EF_ROWS);

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

                    this->filters = std::vector<std::vector<int>>(this->arch->getNTiles(), std::vector<int>());

                    // Select filter for each tile
                    for (int t = 0; t < this->arch->getNTiles(); ++t) {

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
                for (int t = 0; t < this->arch->getNTiles(); ++t) {

                    tiles_data[t].valid = false;

                    if (this->filters[t].empty()) break;

                    while (this->time[t] < this->max_buffer_time) {

                        if (this->arch->schedule()) {

                            // Skip lines of zeroes
                            bool zero_line = this->scheduler->check_zero_line(this->weight_buffer
                                    [filter_set + t][this->time[t]]);
                            if (this->skip[t] < this->scheduler->getLookaheadH() && zero_line) {
                                this->skip[t]++;
                                this->time[t]++;
                                continue;
                            }
                            this->skip[t] = 0;

                        }

                        auto num_act_rows = 1;
                        if (this->arch->schedule()) num_act_rows += this->scheduler->getLookaheadH();
                        tiles_data[t].act_row = BufferSet<T>(this->window_buffer.begin() + this->time[t],
                                std::min(this->window_buffer.begin() + this->time[t] +
                                num_act_rows, this->window_buffer.end()));
                        if (t == 0) {
                            tiles_data[t].act_addresses =
                                    AddressBufferSet(this->window_address_buffer.begin() + this->time[t],
                                    std::min(this->window_address_buffer.begin() + this->time[t] +
                                    num_act_rows, this->window_address_buffer.end()));
                            tiles_data[t].act_banks = this->window_bank_buffer[this->time[t]];
                        }

                        tiles_data[t].wgt_row = this->weight_buffer[filter_set + t][this->time[t]];
                        tiles_data[t].wgt_addresses = this->wgt_address_buffer[filter_set + t][this->time[t]];
                        tiles_data[t].wgt_banks = this->wgt_bank_buffer[filter_set + t][this->time[t]];

                        tiles_data[t].windows = this->windows;
                        tiles_data[t].filters = this->filters[t];
                        tiles_data[t].time = this->time[t];
                        tiles_data[t].lanes = this->EF_LANES;
                        tiles_data[t].valid = true;

                        still_work = true;
                        this->time[t]++;
                        break;

                    } // Buffer time

                } // Tile

                if (still_work) return true;

                this->time = std::vector<int>(this->arch->getNTiles(), 0);
                this->skip = std::vector<int>(this->arch->getNTiles(), 0);
                this->filter_buffer_filled = false;
                this->filters.clear();
                this->filter_set_it += this->arch->getNTiles();
            } // Filter set

            this->filter_set_it = 0;
            this->window_buffer_filled = false;
            this->windows.clear();
            this->window_set_it++;
        } // Window set

        return false;

    }

    template <typename T>
    bool WindowFirstOutS<T>::still_on_chip_data_linear_layer(std::vector<core::TileData<T>> &tiles_data) {
        throw std::runtime_error("TODO");
    }

    template <typename T>
    bool WindowFirstOutS<T>::still_on_chip_data(std::vector<core::TileData<T>> &tiles_data) {
        if (this->linear) return still_on_chip_data_linear_layer(tiles_data);
        else return still_on_chip_data_conv_layer(tiles_data);
    }

    INITIALISE_DATA_TYPES(WindowFirstOutS);

}
