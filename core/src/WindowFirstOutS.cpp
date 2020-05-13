
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

        auto num_windows = this->out_x * this->out_y;

        auto all_input_size = (uint32_t)ceil(act_channels * Nx * Ny * this->dram->getActDataSize() / 8.);
        auto subset_windows = std::min((uint32_t)num_windows, this->EF_COLUMNS);
        auto subset_windows_size = (uint32_t)(subset_windows * act_channels * Kx * Ky //TODO reduce worst case size
                * this->dram->getActDataSize() / 8.);
        auto row_input_size = (uint32_t)ceil(subset_windows_size / (double)this->max_buffer_time);

        auto all_filters_size = (uint32_t)ceil(num_filters * wgt_channels * Kx * Ky
                * this->dram->getWgtDataSize() / 8.);
        auto subset_filters = std::min((uint32_t)num_filters, this->EF_ROWS * this->arch->getTiles());
        auto subset_filters_size = (uint32_t)ceil(subset_filters * wgt_channels * Kx * Ky
                * this->dram->getWgtDataSize() / 8.);
        auto row_filter_size = (uint32_t)ceil(subset_filters_size / (double)this->max_buffer_time);

        auto all_output_size = (uint32_t)ceil(num_filters * num_windows * this->dram->getBaseDataSize() / 8.);
        auto subset_filters_output_size = (uint32_t)ceil(subset_filters * num_windows
                * this->dram->getBaseDataSize() / 8.);
        auto subset_windows_output_size = (uint32_t)ceil(num_filters * subset_windows
                * this->dram->getBaseDataSize() / 8.);
        auto subsets_output_size = (uint32_t)ceil(subset_filters * subset_windows * this->dram->getBaseDataSize() / 8.);

        MemPolicy act_policy;
        if (all_input_size + all_output_size <= this->gbuffer->getActSize()) act_policy = ALL;
        else if (all_input_size + subset_filters_output_size <= this->gbuffer->getActSize()) act_policy = INPUTS;
        else if (subset_windows_size + subset_windows_output_size <= this->gbuffer->getActSize()) act_policy = SET;
        else if (subset_windows_size + subsets_output_size <= this->gbuffer->getActSize()) act_policy = SUBSET;
        else if (row_input_size + subsets_output_size <= this->gbuffer->getActSize()) act_policy = CHANNELS;
        else throw std::runtime_error("Activation memory size too small to allocate one buffer row and one set outputs.");

        MemPolicy wgt_policy;
        if (all_filters_size <= this->gbuffer->getWgtSize()) wgt_policy = ALL;
        else if (subset_filters_size <= this->gbuffer->getWgtSize()) wgt_policy = SET;
        else if (row_filter_size <= this->gbuffer->getWgtSize()) wgt_policy = CHANNELS;
        else throw std::runtime_error("Weight memory size too small to allocate one buffer row.");

        uint32_t time_steps_act = 1, time_steps_wgt = 1;
        auto total_filter_sets = (uint32_t)ceil(num_filters / (double)subset_filters);
        auto filter_sets_per_step_act = total_filter_sets, filter_sets_per_step_wgt = total_filter_sets;

        if (act_policy == ALL) {
            this->next_layer_act_on_chip = true;
        } else if (act_policy == INPUTS) {
            auto output_left_size = this->gbuffer->getActSize() - all_input_size;
            assert(output_left_size < this->gbuffer->getActSize());

            filter_sets_per_step_act = output_left_size / subset_filters_output_size;
            assert(filter_sets_per_step_act < total_filter_sets);
        } else if (act_policy == SET) {
            // None
        } else if (act_policy == SUBSET) {
            auto output_left_size = this->gbuffer->getActSize() - subset_windows_size;
            assert(output_left_size < this->gbuffer->getActSize());

            filter_sets_per_step_act = output_left_size / subsets_output_size;
            assert(filter_sets_per_step_act < total_filter_sets);
        } else {
            auto input_left_size = this->gbuffer->getActSize() - subsets_output_size;
            assert(input_left_size < this->gbuffer->getActSize());

            filter_sets_per_step_act = 1;
            time_steps_act = (uint32_t)ceil(subset_windows_size / (double)input_left_size);
            assert(time_steps_act != 1);

            assert(!this->layer_act_on_chip);
        }

        if (wgt_policy == ALL) {
            // None
        } else if (wgt_policy == SET) {
            filter_sets_per_step_wgt = this->gbuffer->getWgtSize() / subset_filters_size;
            assert(filter_sets_per_step_wgt < this->filter_sets);
        } else {
            filter_sets_per_step_wgt = 1;
            time_steps_wgt = (uint32_t)ceil(subset_filters_size / (double)this->gbuffer->getWgtSize());
            assert(time_steps_wgt != 1);
        }

        auto filter_sets_per_step = std::min(filter_sets_per_step_act, filter_sets_per_step_wgt);
        auto filter_steps = (uint32_t)ceil(total_filter_sets / (double)filter_sets_per_step);

        std::vector<std::vector<int>> window_steps;
        if (wgt_policy == CHANNELS || act_policy == CHANNELS) {
            for (int w = 0; w < this->window_sets; ++w) {
                window_steps.emplace_back(std::vector<int>(1, w));
            }
        } else if (act_policy == ALL || act_policy == INPUTS){
            window_steps.emplace_back(std::vector<int>(this->window_sets, 0));
            std::iota(window_steps.front().begin(), window_steps.front().end(), 0);
        } else {
            auto window_out_size = (uint32_t)ceil(subset_filters * filter_sets_per_step
                    * this->dram->getBaseDataSize() / 8.);
            this->fill_window_steps(window_steps, window_out_size, act_channels);
            assert(window_steps.size() > 1);
        }

        auto time_steps = std::max(time_steps_act, time_steps_wgt);
        auto max_time_per_step = (uint32_t)ceil(this->max_buffer_time /(double)time_steps);
        if ((time_steps - 1) * max_time_per_step >= this->max_buffer_time) {
            assert(max_time_per_step == 2);
            time_steps = this->max_buffer_time;
            max_time_per_step = 1;
        }

        assert(this->max_buffer_time >= time_steps);

        auto last_act_blk = (uint32_t)ceil(act_channels / (double)this->dram->getActValuesPerBlock());
        auto blks_per_window = (uint32_t)ceil(Ky * Kx * act_channels / (double)this->dram->getActValuesPerBlock());
        auto blocks_per_time = (uint32_t)ceil(blks_per_window / (double)this->max_buffer_time);
        auto blocks_per_step = max_time_per_step * blocks_per_time;

        auto next_out_address = this->next_act_address;

        this->on_chip_graph = std::vector<std::shared_ptr<typename Control<T>::Node>>();

        for (int wstep = 0; wstep < window_steps.size(); ++wstep) {

            auto start_window = window_steps[wstep].front() * this->EF_COLUMNS;
            auto total_windows = std::min(window_steps[wstep].size() * this->EF_COLUMNS,
                    (uint64_t)(num_windows - start_window));
            auto end_window = start_window + total_windows;

            for(int fstep = 0; fstep < filter_steps; ++fstep) {

                auto start_filter_set = fstep * filter_sets_per_step;
                auto filter_per_set = std::min(filter_sets_per_step, total_filter_sets - start_filter_set);
                auto end_filter_set = start_filter_set + filter_per_set;

                auto start_filter_subset = fstep * filter_sets_per_step * this->arch->getTiles();
                auto filter_per_subset = std::min(filter_sets_per_step * this->arch->getTiles(),
                        this->filter_sets - start_filter_subset);
                auto end_filter_subset = start_filter_subset + filter_per_subset;

                auto start_filter = fstep * filter_sets_per_step * this->arch->getTiles() * this->EF_ROWS;
                auto total_filters = std::min(filter_sets_per_step * this->arch->getTiles() * this->EF_ROWS,
                        (uint32_t)(num_filters - start_filter));

                for (int tstep = 0; tstep < time_steps; ++tstep) {

                    auto node = std::make_shared<typename OutputStationary<T>::NodeOutS>();

                    node->time_step = tstep;
                    node->max_time = max_time_per_step;
                    node->layer_act_on_chip = this->layer_act_on_chip;
                    node->groups = {0};

                    // Fil activations
                    node->window_sets = window_steps[wstep];

                    if (act_policy == ALL || act_policy == INPUTS) {
                        if (wstep == 0 && fstep == 0 && tstep == 0 && !this->layer_act_on_chip) {
                            auto first_address = this->act_address_map[0][0][0];
                            auto last_address = this->act_address_map[Ny - 1][Nx - 1][last_act_blk - 1];
                            node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        }

                    } else if (act_policy == SET || act_policy == SUBSET) {
                        if (fstep == 0 && tstep == 0 && ((!this->layer_act_on_chip && wstep == 0) || wstep != 0)) {
                            node->read_act_addresses = this->generate_addresses(0, blks_per_window, last_act_blk,
                                    start_window, end_window, 0);
                            node->evict_act = true;
                        }

                        node->layer_act_on_chip = wstep == 0 ? this->layer_act_on_chip : false;

                    } else {
                        auto start_act_blk = tstep * blocks_per_step;
                        auto end_act_blk = std::min((tstep + 1) * blocks_per_step, blks_per_window);

                        if (this->arch->schedule()) {
                            end_act_blk += this->scheduler->getLookaheadH() * blocks_per_time;
                            end_act_blk = std::min(end_act_blk, blks_per_window);
                        }

                        node->read_act_addresses = this->generate_addresses(start_act_blk, end_act_blk, last_act_blk,
                                start_window, end_window, 0);
                        node->evict_act = true;

                        if (fstep != 0 || tstep != 0)
                            node->use_prev_buffer = true;

                    }

                    // Fil filters
                    node->filter_sets = std::vector<int>(filter_per_subset, 0);
                    std::iota(node->filter_sets.begin(), node->filter_sets.end(), start_filter_subset);

                    if (wgt_policy == ALL) {
                        if (wstep == 0 && fstep == 0 && tstep == 0) {
                            auto first_address = std::get<0>(this->wgt_address_map[0]);
                            auto last_address = std::get<1>(this->wgt_address_map[total_filter_sets - 1]);
                            node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        }

                    } else if (wgt_policy == SET) {
                        if (tstep == 0) {
                            auto first_address = std::get<0>(this->wgt_address_map[start_filter_set]);
                            auto last_address = std::get<1>(this->wgt_address_map[end_filter_set - 1]);
                            node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                            node->evict_wgt = true;
                        }

                    } else {
                        auto start_time = tstep * max_time_per_step;
                        auto end_time = std::min((tstep + 1) * max_time_per_step, (uint32_t)this->max_buffer_time) - 1;

                        uint64_t first_address = NULL_ADDR, last_address = NULL_ADDR;
                        if (this->arch->schedule()) {
                            for (int tmp = start_time; tmp <= end_time; ++tmp) {

                                for (int subset = start_filter_subset; subset < end_filter_subset; ++subset) {
                                    if (subset > this->filter_sets) continue;
                                    if (this->wgt_address_buffer[subset][tmp].front() != NULL_ADDR) {
                                        first_address = this->wgt_address_buffer[subset][tmp].front();
                                        break;
                                    }
                                }

                                if (first_address != NULL_ADDR)
                                    break;
                            }

                            for (int tmp = end_time; tmp >= start_time; --tmp) {
                                for (int subset = end_filter_subset; subset > start_filter_subset; --subset) {
                                    if (subset > this->filter_sets) continue;
                                    if (this->wgt_address_buffer[subset - 1][tmp].back() != NULL_ADDR) {
                                        last_address = this->wgt_address_buffer[subset - 1][tmp].back();
                                        break;
                                    }
                                }

                                if (last_address != NULL_ADDR)
                                    break;

                            }

                            if (first_address == NULL_ADDR || last_address == NULL_ADDR)
                                continue;

                        } else {
                            first_address = this->wgt_address_buffer[start_filter_subset][start_time].front();
                            last_address = this->wgt_address_buffer[end_filter_subset - 1][end_time].back();
                        }

                        assert(first_address != NULL_ADDR);
                        assert(last_address != NULL_ADDR);

                        node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        node->evict_wgt = true;

                    }

                    // Fil write addresses
                    if (!this->next_layer_act_on_chip && tstep == time_steps - 1) {
                        auto first_address = this->dram->getStartActAddress() + next_out_address;
                        auto out_blks = ceil(total_filters * total_windows /
                                (double)this->dram->getBaseValuesPerBlock());
                        next_out_address += out_blks * BLOCK_SIZE;
                        auto last_address = this->dram->getStartActAddress() + next_out_address - BLOCK_SIZE;
                        node->write_addresses.emplace_back(first_address, last_address);
                    }

                    this->on_chip_graph.emplace_back(node);

                } // Time step
            } // Filter step
        } // Window step

    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph_grouped_layer() {

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        auto num_windows = this->out_x * this->out_y;

        auto all_input_size = (uint32_t)ceil(act_channels * Nx * Ny * this->dram->getActDataSize() / 8.);
        auto subset_windows = std::min((uint32_t)num_windows, this->EF_COLUMNS);
        auto subset_windows_size = (uint32_t)(subset_windows * act_channels * Kx * Ky //TODO reduce worst case size
                * this->dram->getActDataSize() / 8.);
        auto group_subset_windows_size = (uint32_t)(subset_windows * this->filters_per_group * Kx * Ky //TODO reduce worst case size
                * this->dram->getActDataSize() / 8.);

        auto all_filters_size = (uint32_t)ceil(num_filters * wgt_channels * Kx * Ky
                * this->dram->getWgtDataSize() / 8.);
        auto subset_filters_size = (uint32_t)ceil(this->filters_per_group * wgt_channels * Kx * Ky
                * this->dram->getWgtDataSize() / 8.);

        auto all_output_size = (uint32_t)ceil(num_filters * num_windows * this->dram->getBaseDataSize() / 8.);
        auto subset_filters_output_size = (uint32_t)ceil(this->filters_per_group * num_windows
                * this->dram->getBaseDataSize() / 8.);
        auto subset_windows_output_size = (uint32_t)ceil(num_filters * subset_windows
                * this->dram->getBaseDataSize() / 8.);
        auto subsets_output_size = (uint32_t)ceil(this->filters_per_group * subset_windows
                * this->dram->getBaseDataSize() / 8.);

        MemPolicy act_policy;
        if (all_input_size + all_output_size <= this->gbuffer->getActSize()) act_policy = ALL;
        else if (all_input_size + subset_filters_output_size <= this->gbuffer->getActSize()) act_policy = INPUTS;
        else if (subset_windows_size + subset_windows_output_size <= this->gbuffer->getActSize()) act_policy = SET;
        else if (group_subset_windows_size + subsets_output_size <= this->gbuffer->getActSize()) act_policy = GROUPS;
        else throw std::runtime_error("Activation memory size too small to allocate one window set and group outputs.");

        MemPolicy wgt_policy;
        if (all_filters_size <= this->gbuffer->getWgtSize()) wgt_policy = ALL;
        else if (subset_filters_size <= this->gbuffer->getWgtSize()) wgt_policy = GROUPS;
        else throw std::runtime_error("Weight memory size too small to allocate one group of filters.");

        auto groups_per_step_act = this->groups, groups_per_step_wgt = this->groups;

        if (act_policy == ALL) {
            this->next_layer_act_on_chip = true;
        } else if (act_policy == INPUTS) {
            auto output_left_size = this->gbuffer->getActSize() - all_input_size;
            assert(output_left_size < this->gbuffer->getActSize());

            groups_per_step_act = output_left_size / subset_filters_output_size;
            assert(groups_per_step_act < (this->filter_sets * this->groups));
        } else if (act_policy == SET) {
            // None
        } else {
            groups_per_step_act = this->gbuffer->getActSize() / (group_subset_windows_size + subsets_output_size);
            assert(groups_per_step_act < (this->filter_sets * this->groups));
        }

        if (wgt_policy == ALL) {
            // None
        } else {
            groups_per_step_wgt = this->gbuffer->getWgtSize() / subset_filters_size;
            assert(groups_per_step_wgt < (this->filter_sets * this->groups));
        }

        auto groups_per_step = std::min(groups_per_step_act, groups_per_step_wgt);
        auto group_steps = (uint32_t)ceil(this->groups / (double)groups_per_step);

        std::vector<std::vector<int>> window_steps;
        if (act_policy == ALL || act_policy == INPUTS) {
            window_steps.emplace_back(std::vector<int>(this->window_sets, 0));
            std::iota(window_steps.front().begin(), window_steps.front().end(), 0);
        } else {
            auto channels = std::min(this->filters_per_group * groups_per_step, (uint32_t)act_channels);
            auto window_out_size = (uint32_t)ceil(this->filters_per_group * groups_per_step
                    * this->dram->getBaseDataSize() / 8.);
            this->fill_window_steps(window_steps, window_out_size, channels);
        }

        auto filter_sets_per_group = (uint32_t)ceil(this->filters_per_group /
                (double)(this->EF_ROWS * this->arch->getTiles()));
        auto total_filter_sets = this->groups * filter_sets_per_group;

        auto last_act_blk = (uint32_t)ceil(act_channels / (double)this->dram->getActValuesPerBlock());
        auto last_grp_act_blk = (uint32_t)ceil(this->filters_per_group / (double)this->dram->getActValuesPerBlock());

        auto blks_per_window_group = (uint32_t)ceil(Ky * Kx * this->filters_per_group /
                (double)this->dram->getActValuesPerBlock());
        auto blocks_per_step = blks_per_window_group * groups_per_step;

        auto next_out_address = this->next_act_address;

        this->on_chip_graph = std::vector<std::shared_ptr<typename Control<T>::Node>>();

        for (int gstep = 0; gstep < group_steps; ++gstep) {

            auto start_group = gstep * groups_per_step;
            auto total_groups = std::min(groups_per_step, this->groups - start_group);
            auto start_act_ch = start_group * last_grp_act_blk;

            auto start_filter_set = gstep * groups_per_step * filter_sets_per_group;
            auto filter_per_set = std::min(groups_per_step * filter_sets_per_group, total_filter_sets - start_filter_set);
            auto end_filter_set = start_filter_set + filter_per_set;

            auto start_filter = gstep * this->filters_per_group;
            auto total_filters = std::min(total_groups * this->filters_per_group,
                    (uint32_t)(num_filters - start_filter));

            for (int wstep = 0; wstep < window_steps.size(); ++wstep) {

                auto start_window = window_steps[wstep].front() * this->EF_COLUMNS;
                auto total_windows = std::min(window_steps[wstep].size() * this->EF_COLUMNS,
                        (uint64_t)(num_windows - start_window));
                auto end_window = start_window + total_windows;

                auto node = std::make_shared<typename OutputStationary<T>::NodeOutS>();

                node->time_step = 0;
                node->max_time = this->max_buffer_time;
                node->layer_act_on_chip = this->layer_act_on_chip;

                // Fil groups
                node->groups = std::vector<int>(total_groups, 0);
                std::iota(node->groups.begin(), node->groups.end(), start_group);

                // Fil activations
                node->window_sets = window_steps[wstep];

                if (act_policy == ALL || act_policy == INPUTS) {
                    if (gstep == 0 && wstep == 0 && !this->layer_act_on_chip) {
                        auto first_address = this->act_address_map[0][0][0];
                        auto last_address = this->act_address_map[Ny - 1][Nx - 1][last_act_blk - 1];
                        node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                    }

                } else {
                    if ((!this->layer_act_on_chip && wstep == 0) || wstep != 0) {
                        node->read_act_addresses = this->generate_addresses(0, blocks_per_step,
                                last_grp_act_blk * total_groups, start_window, end_window, start_act_ch);
                        node->evict_act = true;
                    }

                    node->layer_act_on_chip = gstep == 0 && wstep == 0 ? this->layer_act_on_chip : false;

                }

                // Fil filters
                node->filter_sets = std::vector<int>(this->filter_sets, 0);
                std::iota(node->filter_sets.begin(), node->filter_sets.end(), 0);

                if (wgt_policy == ALL) {
                    if (gstep == 0 && wstep == 0) {
                        auto first_address = std::get<0>(this->wgt_address_map[0]);
                        auto last_address = std::get<1>(this->wgt_address_map[this->filter_sets * this->groups - 1]);
                        node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                    }

                } else {
                    if (wstep == 0) {
                        auto first_address = std::get<0>(this->wgt_address_map[start_filter_set]);
                        auto last_address = std::get<1>(this->wgt_address_map[end_filter_set - 1]);
                        node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        node->evict_wgt = true;
                    }

                }

                // Fil write addresses
                if (!this->next_layer_act_on_chip) {
                    auto first_address = this->dram->getStartActAddress() + next_out_address;
                    auto out_blks = ceil(total_filters * total_windows / (double)this->dram->getBaseValuesPerBlock());
                    next_out_address += out_blks * BLOCK_SIZE;
                    auto last_address = this->dram->getStartActAddress() + next_out_address - BLOCK_SIZE;
                    node->write_addresses.emplace_back(first_address, last_address);
                }

                this->on_chip_graph.emplace_back(node);

            } // Window step
        } // Groups

    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph_linear_layer() {

        const std::vector<size_t> &act_shape = this->act->getShape();
        const std::vector<size_t> &wgt_shape = this->wgt->getShape();

        auto recurrences = this->_3dim ? act_shape[1] : 1;
        auto act_channels = this->_3dim ? act_shape[2] : act_shape[1];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];

        auto all_input_size = (uint32_t)ceil(act_channels * this->dram->getActDataSize() / 8.);
        auto row_input_size = (uint32_t)ceil(all_input_size / (double)this->max_buffer_time);

        auto all_filters_size = (uint32_t)ceil(num_filters * wgt_channels * this->dram->getWgtDataSize() / 8.);
        auto subset_filters = std::min((uint32_t)num_filters, this->EF_ROWS * this->arch->getTiles());
        auto subset_filters_size = (uint32_t)ceil(subset_filters * wgt_channels * this->dram->getWgtDataSize() / 8.);
        auto row_filter_size = (uint32_t)ceil(subset_filters_size / (double)this->max_buffer_time);

        auto all_output_size = (uint32_t)ceil(num_filters * this->dram->getBaseDataSize() / 8.);
        auto subset_filters_output_size = (uint32_t)ceil(subset_filters * this->dram->getBaseDataSize() / 8.);

        MemPolicy act_policy;
        if (all_input_size + all_output_size <= this->gbuffer->getActSize()) act_policy = ALL;
        else if (all_input_size + subset_filters_output_size <= this->gbuffer->getActSize()) act_policy = INPUTS;
        else if (row_input_size + subset_filters_output_size <= this->gbuffer->getActSize()) act_policy = CHANNELS;
        else throw std::runtime_error("Activation memory size too small to allocate one buffer row and one set outputs.");

        MemPolicy wgt_policy;
        if (all_filters_size <= this->gbuffer->getWgtSize()) wgt_policy = ALL;
        else if (subset_filters_size <= this->gbuffer->getWgtSize()) wgt_policy = SET;
        else if (row_filter_size <= this->gbuffer->getWgtSize()) wgt_policy = CHANNELS;
        else throw std::runtime_error("Weight memory size too small to allocate one buffer row.");

        uint32_t time_steps_act = 1, time_steps_wgt = 1;
        auto total_filter_sets = (uint32_t)ceil(num_filters / (double)subset_filters);
        auto filter_sets_per_step_act = total_filter_sets, filter_sets_per_step_wgt = total_filter_sets;

        if (act_policy == ALL) {
            this->next_layer_act_on_chip = true;
        } else if (act_policy == INPUTS) {
            auto output_left_size = this->gbuffer->getActSize() - all_input_size;
            assert(output_left_size < this->gbuffer->getActSize());

            filter_sets_per_step_act = output_left_size / subset_filters_output_size;
            assert(filter_sets_per_step_act < total_filter_sets);
        } else {
            auto input_left_size = this->gbuffer->getActSize() - subset_filters_output_size;
            assert(input_left_size < this->gbuffer->getActSize());

            filter_sets_per_step_act = 1;
            time_steps_act = (uint32_t)ceil(all_input_size / (double)input_left_size);
            assert(time_steps_act != 1);

            assert(!this->layer_act_on_chip);
        }

        if (wgt_policy == ALL) {
            // None
        } else if (wgt_policy == SET) {
            filter_sets_per_step_wgt = this->gbuffer->getWgtSize() / subset_filters_size;
            assert(filter_sets_per_step_wgt < this->filter_sets);
        } else {
            filter_sets_per_step_wgt = 1;
            time_steps_wgt = (uint32_t)ceil(subset_filters_size / (double)this->gbuffer->getWgtSize());
            assert(time_steps_wgt != 1);
        }

        uint32_t filter_sets_per_step = std::min(filter_sets_per_step_act, filter_sets_per_step_wgt);
        uint32_t filter_steps = ceil(total_filter_sets / (double)filter_sets_per_step);

        auto time_steps = std::max(time_steps_act, time_steps_wgt);
        auto max_time_per_step = (uint32_t)ceil(this->max_buffer_time / (double)time_steps);
        if ((time_steps - 1) * max_time_per_step >= this->max_buffer_time) {
            assert(max_time_per_step == 2);
            time_steps = this->max_buffer_time;
            max_time_per_step = 1;
        }

        assert(this->max_buffer_time >= time_steps);

        auto last_act_blk = (uint32_t)ceil(act_channels / (double)this->dram->getActValuesPerBlock());
        auto blocks_per_time = (uint32_t)ceil(last_act_blk / (double)this->max_buffer_time);
        auto blocks_per_step = max_time_per_step * blocks_per_time;

        this->on_chip_graph = std::vector<std::shared_ptr<typename Control<T>::Node>>();

        for (int r = 0; r < recurrences; ++r) {

            auto next_out_address = this->next_act_address;

            for (int fstep = 0; fstep < filter_steps; ++fstep) {

                auto start_filter_set = fstep * filter_sets_per_step;
                auto filter_per_set = std::min(filter_sets_per_step, total_filter_sets - start_filter_set);
                auto end_filter_set = start_filter_set + filter_per_set;

                auto start_filter_subset = fstep * filter_sets_per_step * this->arch->getTiles();
                auto filter_per_subset = std::min(filter_sets_per_step * this->arch->getTiles(),
                        this->filter_sets - start_filter_subset);
                auto end_filter_subset = start_filter_subset + filter_per_subset;

                auto start_filter = fstep * filter_sets_per_step * this->arch->getTiles() * this->EF_ROWS;
                auto total_filters = std::min(filter_sets_per_step * this->arch->getTiles() * this->EF_ROWS,
                        (uint32_t)(num_filters - start_filter));

                for (int tstep = 0; tstep < time_steps; ++tstep) {

                    auto node = std::make_shared<typename OutputStationary<T>::NodeOutS>();

                    node->time_step = tstep;
                    node->max_time = max_time_per_step;
                    node->layer_act_on_chip = this->layer_act_on_chip;
                    node->recurrence = r;

                    // Fil activations
                    if (act_policy == ALL) {
                        if (r == 0 && fstep == 0 && tstep == 0 && !this->layer_act_on_chip) {
                            auto first_address = this->act_address_map[0][0][0];
                            auto last_address = this->act_address_map[0][0][last_act_blk - 1];
                            node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        }

                        if (fstep != 0 || tstep != 0)
                            node->use_prev_buffer = true;

                    } else if (act_policy == INPUTS) {
                        if (fstep == 0 && tstep == 0 && ((!this->layer_act_on_chip && r == 0) || r != 0)) {
                            auto first_address = this->act_address_map[0][0][0];
                            auto last_address = this->act_address_map[0][0][last_act_blk - 1];
                            node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                            node->evict_act = true;
                        }

                        node->layer_act_on_chip = r == 0 ? this->layer_act_on_chip : false;

                        if (fstep != 0)
                            node->use_prev_buffer = true;

                    } else {
                        auto start_act_blk = tstep * blocks_per_step;
                        auto end_act_blk = std::min((tstep + 1) * blocks_per_step, last_act_blk);

                        if (this->arch->schedule()) {
                            end_act_blk += this->scheduler->getLookaheadH() * blocks_per_time;
                            end_act_blk = std::min(end_act_blk, last_act_blk);
                        }

                        auto first_address = this->act_address_map[0][0][start_act_blk];
                        auto last_address = this->act_address_map[0][0][end_act_blk - 1];
                        node->read_act_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        node->evict_act = true;

                        if (fstep != 0 || tstep != 0)
                            node->use_prev_buffer = true;

                    }

                    // Fil filters
                    node->filter_sets = std::vector<int>(filter_per_subset, 0);
                    std::iota(node->filter_sets.begin(), node->filter_sets.end(), start_filter_subset);

                    if (wgt_policy == ALL) {
                        if (r == 0 && fstep == 0 && tstep == 0) {
                            auto first_address = std::get<0>(this->wgt_address_map[0]);
                            auto last_address = std::get<1>(this->wgt_address_map[total_filter_sets - 1]);
                            node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        }

                    } else if (wgt_policy == SET) {
                        if (tstep == 0) {
                            auto first_address = std::get<0>(this->wgt_address_map[start_filter_set]);
                            auto last_address = std::get<1>(this->wgt_address_map[end_filter_set - 1]);
                            node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                            node->evict_wgt = true;
                        }

                    } else {
                        auto start_time = tstep * max_time_per_step;
                        auto end_time = std::min((tstep + 1) * max_time_per_step, (uint32_t)this->max_buffer_time) - 1;

                        uint64_t first_address = NULL_ADDR, last_address = NULL_ADDR;
                        if (this->arch->schedule()) {
                            for (int tmp = start_time; tmp <= end_time; ++tmp) {

                                for (int subset = start_filter_subset; subset < end_filter_subset; ++subset) {
                                    if (subset > this->filter_sets) continue;
                                    if (this->wgt_address_buffer[subset][tmp].front() != NULL_ADDR) {
                                        first_address = this->wgt_address_buffer[subset][tmp].front();
                                        break;
                                    }
                                }

                                if (first_address != NULL_ADDR)
                                    break;
                            }

                            for (int tmp = end_time; tmp >= start_time ; --tmp) {
                                for (int subset = end_filter_subset; subset > start_filter_subset; --subset) {
                                    if (subset > this->filter_sets) continue;
                                    if (this->wgt_address_buffer[subset - 1][tmp].back() != NULL_ADDR) {
                                        last_address = this->wgt_address_buffer[subset - 1][tmp].back();
                                        break;
                                    }
                                }

                                if (last_address != NULL_ADDR)
                                    break;

                            }

                            if (first_address == NULL_ADDR || last_address == NULL_ADDR)
                                continue;

                        } else {
                            first_address = this->wgt_address_buffer[start_filter_subset][start_time].front();
                            last_address = this->wgt_address_buffer[end_filter_subset - 1][end_time].back();
                        }

                        assert(first_address != NULL_ADDR);
                        assert(last_address != NULL_ADDR);

                        node->read_wgt_addresses.emplace_back(std::make_tuple(first_address, last_address));
                        node->evict_wgt = true;

                    }

                    // Fil write addresses
                    if (!this->next_layer_act_on_chip && tstep == time_steps - 1) {
                        auto first_address = this->dram->getStartActAddress() + next_out_address;
                        auto out_blks = ceil(total_filters / (double)this->dram->getBaseValuesPerBlock());
                        next_out_address += out_blks * BLOCK_SIZE;
                        auto last_address = this->dram->getStartActAddress() + next_out_address - BLOCK_SIZE;
                        node->write_addresses.emplace_back(first_address, last_address);
                    }

                    this->on_chip_graph.emplace_back(node);

                } // Time step
            } // Filter step
        } // Recurrence

    }

    template <typename T>
    void WindowFirstOutS<T>::generate_execution_graph() {
        if (this->linear) generate_execution_graph_linear_layer();
        else if (this->groups > 1) generate_execution_graph_grouped_layer();
        else generate_execution_graph_conv_layer();
    }

    template <typename T>
    void WindowFirstOutS<T>::configure_layer(const std::shared_ptr<base::Array<T>> &_act,
            const std::shared_ptr<base::Array<T>> &_wgt, uint32_t act_prec, uint32_t wgt_prec, bool _linear,
            bool __3dim, int _stride) {

        OutputStationary<T>::configure_layer(_act, _wgt, act_prec, wgt_prec, _linear, __3dim, _stride);

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
        const auto &groups = current_node->groups;
        const auto &window_sets = current_node->window_sets;
        const auto &filter_tile_sets = current_node->filter_sets;
        const auto &time_step = current_node->time_step;
        const auto &max_time = current_node->max_time;
        const auto &use_prev_buffer = current_node->use_prev_buffer;
        const auto &layer_act_on_chip = current_node->layer_act_on_chip;

        while (this->group_it < groups.size()) {
            auto group_idx = groups[this->group_it];

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

                    if (!use_prev_buffer)
                        this->fill_window_buffer(group_idx);
                    this->window_buffer_filled = true;
                }

                while (this->filter_set_it < filter_tile_sets.size()) {

                    auto filter_set = filter_tile_sets[this->filter_set_it];

                    // Filter set
                    if (!this->filter_buffer_filled) {

                        this->filters = std::vector<std::vector<int>>(this->arch->getTiles(), std::vector<int>());

                        // Select filter for each tile
                        for (int t = 0; t < this->arch->getTiles(); ++t) {

                            auto filter_idx = group_idx * this->filters_per_group + (filter_set + t) * this->EF_ROWS;

                            auto num_filters = this->wgt->getShape()[0];
                            for (int r = 0; r < this->EF_ROWS; ++r) {
                                auto filter = filter_idx + r;
                                if (filter >= ((group_idx + 1)) * this->filters_per_group || filter >= num_filters)
                                    continue;
                                this->filters[t].push_back(filter);
                            }

                        }

                        this->filter_buffer_filled = true;
                        if (this->prev_filter_set != filter_set)
                            this->skip = std::vector<int>(this->arch->getTiles(), 0);
                        this->prev_filter_set = filter_set;
                    }

                    bool first = true;
                    bool still_work = false;
                    for (int t = 0; t < this->arch->getTiles(); ++t) {

                        tiles_data[t].write = false;
                        tiles_data[t].valid = false;

                        if (this->filters[t].empty()) break;

                        while (this->time[t] < max_time) {
                            auto set_time = time_step * max_time + this->time[t];

                            if (set_time >= this->max_buffer_time)
                                break;

                            if (this->arch->schedule()) {

                                // Skip lines of zeroes
                                bool zero_line = this->scheduler->check_zero_line(this->weight_buffer
                                        [group_idx * this->filter_sets + filter_set + t][set_time]);
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
                            if (first) {
                                if (!layer_act_on_chip) {
                                    tiles_data[t].act_addresses =
                                            AddressBufferSet(this->window_address_buffer.begin() + set_time,
                                            std::min(this->window_address_buffer.begin() + set_time + num_act_rows,
                                            this->window_address_buffer.end()));
                                }
                                tiles_data[t].act_banks = this->window_bank_buffer[set_time];
                                first = false;
                            } else {
                                tiles_data[t].act_addresses.clear();
                                tiles_data[t].act_banks.clear();
                            }

                            tiles_data[t].wgt_row =
                                    this->weight_buffer[group_idx * this->filter_sets + filter_set + t][set_time];
                            tiles_data[t].wgt_addresses =
                                    this->wgt_address_buffer[group_idx * this->filter_sets + filter_set + t][set_time];
                            tiles_data[t].wgt_banks =
                                    this->wgt_bank_buffer[group_idx * this->filter_sets + filter_set + t][set_time];

                            tiles_data[t].windows = this->windows;
                            tiles_data[t].filters = this->filters[t];
                            tiles_data[t].time = set_time;
                            tiles_data[t].lanes = this->EF_LANES;
                            tiles_data[t].valid = true;

                            still_work = true;
                            this->write[t] = true;
                            this->time[t]++;
                            break;

                        } // Buffer time

                    } // Tile

                    if (still_work) {

                        // Check if all tiles are done
                        bool tiles_done = true;
                        for (int t = 0; t < this->arch->getTiles() && tiles_done; ++t) {
                            if (!tiles_data[t].valid) continue;
                            auto set_time = time_step * max_time + this->time[t];
                            if (set_time <= this->wgt_end_time[filter_set + t])
                                tiles_done = false;
                        }

                        if (tiles_done) {
                            auto out_bank_idx = 0;
                            for (int t = 0; t < this->arch->getTiles(); ++t) {
                                if (!this->write[t]) continue;
                                auto outputs = this->windows.size() * this->filters[t].size();
                                tiles_data[t].out_banks = BankBufferRow(outputs, 0);
                                tiles_data[t].write = true;

                                for (int ob = 0; ob < outputs; ++ob) {
                                    tiles_data[t].out_banks[ob] = out_bank_idx;
                                    out_bank_idx = (out_bank_idx + 1) % this->gbuffer->getOutBanks();
                                }
                            }
                        }

                        return true;
                    }

                    this->write = std::vector<bool>(this->arch->getTiles(), false);
                    this->time = std::vector<int>(this->arch->getTiles(), 0);
                    this->filter_buffer_filled = false;
                    this->prev_filter_set = 0;
                    this->filters.clear();
                    this->filter_set_it += this->arch->getTiles();
                } // Filter set

                this->filter_set_it = 0;
                this->window_buffer_filled = false;
                this->windows.clear();
                this->window_set_it++;
            } // Window set

            this->window_set_it = 0;
            this->group_it++;
        } // Groups

        this->group_it = 0;
        return false;

    }

    template <typename T>
    bool WindowFirstOutS<T>::still_on_chip_data_linear_layer(std::vector<core::TileData<T>> &tiles_data) {

        // Select values from current node
        const auto &current_node = std::static_pointer_cast<typename OutputStationary<T>::NodeOutS>
                (this->on_chip_graph.front());
        const auto &filter_tile_sets = current_node->filter_sets;
        const auto &time_step = current_node->time_step;
        const auto &max_time = current_node->max_time;
        const auto &use_prev_buffer = current_node->use_prev_buffer;
        const auto &layer_act_on_chip = current_node->layer_act_on_chip;

        // Fill window buffer
        if (!use_prev_buffer && !this->window_buffer_filled) {
            this->windows = {{0, 0}};
            this->fill_window_buffer(0);
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
                if (this->prev_filter_set != filter_set)
                    this->skip = std::vector<int>(this->arch->getTiles(), 0);
                this->prev_filter_set = filter_set;
            }

            bool first = true;
            bool still_work = false;
            for (int t = 0; t < this->arch->getTiles(); ++t) {

                tiles_data[t].write = false;
                tiles_data[t].valid = false;

                if (this->filters[t].empty()) break;

                while (this->time[t] < max_time) {
                    auto set_time = time_step * max_time + this->time[t];

                    if (set_time >= this->max_buffer_time)
                        break;

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
                    if (first) {
                        if (!layer_act_on_chip) {
                            tiles_data[t].act_addresses =
                                    AddressBufferSet(this->window_address_buffer.begin() + set_time,
                                    std::min(this->window_address_buffer.begin() + set_time + num_act_rows,
                                    this->window_address_buffer.end()));
                        }
                        tiles_data[t].act_banks = this->window_bank_buffer[set_time];
                        first = false;
                    } else {
                        tiles_data[t].act_addresses.clear();
                        tiles_data[t].act_banks.clear();
                    }

                    tiles_data[t].wgt_row = this->weight_buffer[filter_set + t][set_time];
                    tiles_data[t].wgt_addresses = this->wgt_address_buffer[filter_set + t][set_time];
                    tiles_data[t].wgt_banks = this->wgt_bank_buffer[filter_set + t][set_time];

                    tiles_data[t].windows = std::vector<WindowCoord>(this->EF_COLUMNS, {0, 0});
                    tiles_data[t].filters = this->filters[t];
                    tiles_data[t].time = set_time;
                    tiles_data[t].lanes = this->EF_LANES;
                    tiles_data[t].valid = true;

                    still_work = true;
                    this->write[t] = true;
                    this->time[t]++;
                    break;

                } // Buffer time

            } // Tile

            if (still_work) {

                // Check if all tiles are done
                bool tiles_done = true;
                for (int t = 0; t < this->arch->getTiles() && tiles_done; ++t) {
                    if (!tiles_data[t].valid) continue;
                    auto set_time = time_step * max_time + this->time[t];
                    if (set_time <= this->wgt_end_time[filter_set + t])
                        tiles_done = false;
                }

                if (tiles_done) {
                    auto out_bank_idx = 0;
                    for (int t = 0; t < this->arch->getTiles(); ++t) {
                        if (!this->write[t]) continue;
                        tiles_data[t].out_banks = BankBufferRow(this->filters.size(), 0);
                        tiles_data[t].write = true;

                        for (int ob = 0; ob < this->filters[t].size(); ++ob) {
                            tiles_data[t].out_banks[ob] = out_bank_idx;
                            out_bank_idx = (out_bank_idx + 1) % this->gbuffer->getOutBanks();
                        }
                    }
                }

                return true;
            }

            this->write = std::vector<bool>(this->arch->getTiles(), false);
            this->time = std::vector<int>(this->arch->getTiles(), 0);
            this->filter_buffer_filled = false;
            this->prev_filter_set = 0;
            this->filters.clear();
            this->filter_set_it += this->arch->getTiles();
        } // Filter set

        this->window_set_it = 0;
        this->filter_set_it = 0;
        this->window_buffer_filled = false;
        return false;

    }

    template <typename T>
    bool WindowFirstOutS<T>::still_on_chip_data(std::vector<core::TileData<T>> &tiles_data) {
        if (this->linear) return still_on_chip_data_linear_layer(tiles_data);
        else return still_on_chip_data_conv_layer(tiles_data);
    }

    INITIALISE_DATA_TYPES(WindowFirstOutS);

}
