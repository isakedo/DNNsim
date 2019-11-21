
#include <core/DynamicStripes.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t DynamicStripes<T>::computeDynamicStripesBitsPE(uint8_t layer_prec, const int network_bits) {
        return layer_prec * (uint8_t)network_bits;
    }

    template <typename T>
    uint8_t DynamicStripes<T>::computeDynamicStripesColumn(int batch, int recursion, int act_x, int act_y, int kernel_x,
            int kernel_y, int init_channel, int stride, const base::Array<T> &padded_act, uint16_t act_mask,
            int max_channel, bool lstm) {

        uint8_t max_bit = 0, min_bit = 16;
        for (int channel = init_channel; channel < std::min(init_channel + (int)N_LANES, max_channel); channel++) {

            // Computation cycles
            uint16_t act_bits;
            if(lstm)
                act_bits = padded_act.get(recursion, batch, channel);
            else
                act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x, stride * act_y + kernel_y);

            bool neg = false;
            if((act_bits & act_mask) != 0) {
                act_bits = act_bits & ~act_mask;
                neg = true;
            }

            const auto &min_max_act_bits = this->minMax(act_bits);

            auto min_act_bit = std::get<0>(min_max_act_bits);
            auto max_act_bit = std::get<1>(min_max_act_bits);

            if(neg) max_act_bit += 1;

            if(min_act_bit < min_bit) min_bit = min_act_bit;
            if(max_act_bit > max_bit) max_bit = max_act_bit;

        }

        int cycles;
        if(!LEADING_BIT) cycles = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
        else cycles = max_bit + 1;
        return (uint8_t)cycles;

    }

    template <typename T>
    void DynamicStripes<T>::computeDynamicStripesTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int stride,
            const base::Array<T> &padded_act, uint16_t act_mask, int max_channel,
            std::vector<uint32_t> &cycles_per_group, std::vector<uint32_t> &end_previous_pallet,
            uint64_t &stall_cycles) {

        uint32_t N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        uint32_t WINDOWS_PER_GROUP = N_COLUMNS / N_GROUPS;

        std::vector<uint8_t> per_group_cycles (N_GROUPS, 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_bit = 0, min_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {

            if(group_counter == WINDOWS_PER_GROUP)  {
                max_bit = 0, min_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for (int channel = init_channel; channel < std::min(init_channel + (int)N_LANES, max_channel); channel++) {

                // Computation cycles
                uint16_t act_bits;
                if(DIFFY) {
                    short raw_act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                            stride * list_act_y[window] + kernel_y);
                    short prev_act_bits = (stride * list_act_x[window] - stride < 0) ? 0 :
                            padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x - stride,
                                stride * list_act_y[window] + kernel_y);

                    raw_act_bits = raw_act_bits - prev_act_bits;

                    act_bits = this->sign_magnitude(raw_act_bits,act_mask);
                } else {
                    act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                            stride * list_act_y[window] + kernel_y);
                }

                bool neg = false;
                if((act_bits & act_mask) != 0) {
                    act_bits = act_bits & ~act_mask;
                    neg = true;
                }

                const auto &min_max_act_bits = this->minMax(act_bits);

                auto min_act_bit = std::get<0>(min_max_act_bits);
                auto max_act_bit = std::get<1>(min_max_act_bits);

                if(neg) max_act_bit += 1;

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }

            group_counter++;
            if(group_counter == WINDOWS_PER_GROUP) {
                if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 : max_bit - min_bit + 1);
                else per_group_cycles[group_index] = (uint8_t)(max_bit + 1);
            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 : max_bit - min_bit + 1);
            else per_group_cycles[group_index] = (uint8_t)(max_bit + 1);
        }

        for(int group = 0; group < N_GROUPS; group++) {
            cycles_per_group[group] += per_group_cycles[group];
        }

        if(COLUMN_REGISTERS > 0) {
            auto fastest_column = end_previous_pallet[0] + 1;
            for(auto &column_cycles : cycles_per_group) {
                if(column_cycles <= end_previous_pallet[0]) {
                    if(column_cycles < fastest_column) fastest_column = column_cycles;
                    column_cycles = end_previous_pallet[0] + 1;
                }
            }
            stall_cycles += (end_previous_pallet[0] + 1) - fastest_column;

            //Update end_previous_pallet
            for(int i = 0; i < COLUMN_REGISTERS - 1; i++) {
                end_previous_pallet[i] = end_previous_pallet[i + 1];
            }
            end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_group.begin(),
                    cycles_per_group.end());
        } else {
            auto slowest_group = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
            auto fastest_group = *std::min_element(cycles_per_group.begin(), cycles_per_group.end());
            cycles_per_group = std::vector<uint32_t>(N_GROUPS, slowest_group);
            stall_cycles += slowest_group - fastest_group;
        }

    }

    template <typename T>
    void DynamicStripes<T>::computeDynamicStripes2DTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_filter, int stride,
            const base::Array<T> &padded_act, const base::Array<T> &wgt, uint16_t act_mask, int max_filter,
            std::vector<uint32_t> &cycles_per_group, std::vector<uint32_t> &end_previous_pallet,
            uint64_t &stall_cycles) {

        //Get the slowest column
        uint32_t N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        uint32_t WINDOWS_PER_GROUP = N_COLUMNS / N_GROUPS;

        std::vector<uint8_t> per_group_cycles (N_GROUPS, 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_bit = 0, min_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {

            if(group_counter == WINDOWS_PER_GROUP)  {
                max_bit = 0, min_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for (int filter = init_filter; filter < std::min(init_filter + (int)(N_ROWS * N_TILES), max_filter); filter++) {

                uint16_t act_bits = padded_act.get(batch, filter, stride * list_act_x[window] + kernel_x,
                        stride * list_act_y[window] + kernel_y);

                bool neg = false;
                if((act_bits & act_mask) != 0) {
                    act_bits = act_bits & ~act_mask;
                    neg = true;
                }

                const auto &min_max_act_bits = this->minMax(act_bits);

                auto min_act_bit = std::get<0>(min_max_act_bits);
                auto max_act_bit = std::get<1>(min_max_act_bits);

                if(neg) max_act_bit += 1;

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }

            group_counter++;
            if(group_counter == WINDOWS_PER_GROUP) {
                if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 :
                        max_bit - min_bit + 1);
                else per_group_cycles[group_index] = (uint8_t)(max_bit + 1);

            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 : max_bit - min_bit + 1);
            else per_group_cycles[group_index] = (uint8_t)(max_bit + 1);

        }

        for(int group = 0; group < N_GROUPS; group++) {
            cycles_per_group[group] += per_group_cycles[group];
        }

        if(COLUMN_REGISTERS > 0) {
            auto fastest_column = end_previous_pallet[0] + 1;
            for(auto &column_cycles : cycles_per_group) {
                if(column_cycles <= end_previous_pallet[0]) {
                    if(column_cycles < fastest_column) fastest_column = column_cycles;
                    column_cycles = end_previous_pallet[0] + 1;
                }
            }
            stall_cycles += (end_previous_pallet[0] + 1) - fastest_column;

            //Update end_previous_pallet
            for(int i = 0; i < COLUMN_REGISTERS - 1; i++) {
                end_previous_pallet[i] = end_previous_pallet[i + 1];
            }
            end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_group.begin(),
                    cycles_per_group.end());
        } else {
            auto slowest_group = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
            auto fastest_group = *std::min_element(cycles_per_group.begin(), cycles_per_group.end());
            cycles_per_group = std::vector<uint32_t>(N_GROUPS, slowest_group);
            stall_cycles += slowest_group - fastest_group;
        }

    }

    /* CYCLES */

    template <typename T>
    void DynamicStripes<T>::run(const base::Network<T> &network) {

        // Initialize statistics
        std::string arch = "DynamicStripes";
        arch += (DIFFY ? "_Diffy" : "");
        std::string filename = arch + "_L" + std::to_string(N_LANES) + "_C" + std::to_string(N_COLUMNS) + "_R" +
                std::to_string(N_ROWS) + "_T" + std::to_string(N_TILES) + "_PG" + std::to_string(PRECISION_GRANULARITY)
                + "_CR" + std::to_string(COLUMN_REGISTERS) + "_BP" + std::to_string(BITS_PE) +
                (LEADING_BIT ? "_LB" : "") + "_cycles";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto baseline_cycles = stats.register_uint_t("baseline_cycles", 0, sys::AverageTotal);
        auto speedup = stats.register_double_t("speedup", 0, sys::Special);
        auto stall_cycles = stats.register_uint_t("stall_cycles", 0, sys::AverageTotal);
        auto idle_rows = stats.register_uint_t("idle_rows", 0, sys::AverageTotal);
        auto rows_per_wgt = stats.register_uint_t("rows_per_wgt", 0, sys::AverageTotal);
        auto weight_buff_reads = stats.register_uint_t("weight_buff_reads", 0, sys::AverageTotal);
        auto act_buff_reads = stats.register_uint_t("act_buff_reads", 0, sys::AverageTotal);
        auto accumulator_updates = stats.register_uint_t("accumulator_updates", 0, sys::AverageTotal);
        auto scheduled_pe = stats.register_uint_t("scheduled_pe", 0, sys::AverageTotal);
        auto idle_pe = stats.register_uint_t("idle_pe", 0, sys::AverageTotal);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto TOTAL_ROWS = N_ROWS * N_TILES;

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            if(!DIFFY) act.sign_magnitude_representation(layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();
            if(fc) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            if(conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            if(act.getShape()[1] == 3 && stride > 1) {
                act.reshape_first_layer_act((uint16_t)stride);
                wgt.reshape_first_layer_wgt((uint16_t)stride);
                stride = 1;
            }

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
            if(this->FAST_MODE) batch_size = 1;

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            auto act_layer_prec = (uint16_t)layer.getActPrecision();
            auto act_mask = (uint16_t)(1u << (act_layer_prec - 1u));

            auto wgt_layer_prec = layer.getWgtPrecision();
            auto layer_rows_per_wgt = (int)ceil(wgt_layer_prec / (double)BITS_PE);
            auto filters_per_tile = TOTAL_ROWS/layer_rows_per_wgt;

            auto groups = act_channels / wgt_channels;
            auto num_filters_sets = (uint32_t)ceil(num_filters/(double)filters_per_tile/groups);
            auto baseline_filters_sets = (uint32_t)ceil(num_filters/(double)TOTAL_ROWS/groups);

            auto base_cycles = (uint64_t)(conv ? out_x * out_y * ceil(act_channels/(double)N_LANES) * Kx * Ky *
                    baseline_filters_sets : ceil(act_channels/(double)N_LANES) * baseline_filters_sets * R);

            for(int n = 0; n < batch_size; n++) {

                uint64_t batch_cycles = 0;
                uint64_t batch_stall_cycles = 0;
                uint64_t batch_weight_buff_reads = 0;
                uint64_t batch_act_buff_reads = 0;
                uint64_t batch_accumulator_updates = 0;
                uint64_t batch_scheduled_pe = 0;
                uint64_t batch_idle_pe = 0;

                if (conv && wgt_shape[1] == 1 && act_shape[1] != 1) {

                    std::vector<int> list_x, list_y;
                    int x_counter = 0, y_counter = 0;
                    std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(COLUMN_REGISTERS, 0);
                    std::vector<uint32_t> cycles_per_group = std::vector<uint32_t>(N_COLUMNS * 16 / PRECISION_GRANULARITY, 0);

                    for(int m = 0; m < num_filters; m += filters_per_tile) {
                        while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                            for (int i = 0; i < Kx; i++) {
                                for (int j = 0; j < Ky; j++) {
                                    computeDynamicStripes2DTile(n,list_x, list_y, i, j, m, stride, act, wgt, act_mask,
                                            (int)num_filters, cycles_per_group, end_previous_pallet, batch_stall_cycles);

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                    batch_scheduled_pe += list_x.size() * TOTAL_ROWS;
                                    batch_idle_pe += (N_COLUMNS - list_x.size()) * TOTAL_ROWS;
                                }
                            }
                            batch_accumulator_updates++;
                        }
                    }

                    cycles->value[layer_it][n] = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
                    stall_cycles->value[layer_it][n] = batch_stall_cycles;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates;
                    scheduled_pe->value[layer_it][n] = batch_scheduled_pe;
                    idle_pe->value[layer_it][n] = batch_idle_pe;
                    speedup->value[layer_it][n] = base_cycles / (double)cycles->value[layer_it][n];

                } else if (conv) {

                    std::vector<int> list_x, list_y;
                    int x_counter = 0, y_counter = 0;
                    std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(COLUMN_REGISTERS, 0);
                    std::vector<uint32_t> cycles_per_group = std::vector<uint32_t>(N_COLUMNS * 16 / PRECISION_GRANULARITY, 0);

                    while (this->iterateWindows(out_x, out_y, list_x, list_y, x_counter, y_counter, N_COLUMNS)) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = 0; k < act_channels; k += N_LANES) {
                                    computeDynamicStripesTile(n, list_x, list_y, i, j, k, stride, act, act_mask,
                                            (int)act_channels, cycles_per_group, end_previous_pallet, batch_stall_cycles);

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                    batch_scheduled_pe += list_x.size() * TOTAL_ROWS;
                                    batch_idle_pe += (N_COLUMNS - list_x.size()) * TOTAL_ROWS;
                                }
                            }
                        }
                        batch_accumulator_updates++;
                    }

                    batch_cycles = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
                    cycles->value[layer_it][n] = batch_cycles * num_filters_sets;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles * num_filters_sets;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads * num_filters_sets * N_TILES;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads * num_filters_sets * N_TILES;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates * num_filters_sets * N_TILES;
                    scheduled_pe->value[layer_it][n] = batch_scheduled_pe * num_filters_sets;
                    idle_pe->value[layer_it][n] = batch_idle_pe * num_filters_sets;
                    baseline_cycles->value[layer_it][n] = base_cycles;
                    speedup->value[layer_it][n] = base_cycles / (double)cycles->value[layer_it][n];

                } else {

                    int column_index = 0;
                    std::vector<uint64_t> column_end = std::vector<uint64_t>(N_COLUMNS, 0);

                    for (int r = 0; r < R; r++) {
                        for (int k = 0; k < act_channels; k += N_LANES) {
                            if(batch_cycles < column_end[column_index]) {
                                batch_stall_cycles += column_end[column_index] - batch_cycles;
                                batch_cycles = column_end[column_index];
                            }
                            auto column_cycles = computeDynamicStripesColumn(n,r,0,0,0,0,k,0,act,act_mask,(int)act_channels,lstm);
                            column_end[column_index] = batch_cycles + column_cycles;
                            batch_cycles++;
                            column_index++;
                            if(column_index >= N_COLUMNS) column_index = 0;

                            batch_act_buff_reads++;
                            batch_weight_buff_reads++;
                        }
                        batch_accumulator_updates++;
                    }

                    uint64_t last_column_end = *std::max_element(column_end.begin(), column_end.end());
                    uint64_t last_column_rem_cycles = last_column_end - batch_cycles;
                    cycles->value[layer_it][n] = batch_cycles * num_filters_sets;
                    cycles->value[layer_it][n] += last_column_rem_cycles;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles * num_filters_sets;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads * num_filters_sets * N_TILES;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads * num_filters_sets * N_TILES;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates * num_filters_sets * N_TILES;
                    scheduled_pe->value[layer_it][n] = (uint64_t)(num_filters * TOTAL_ROWS *
                            ceil(act_channels/(double)N_LANES));
                    auto batch_idle_rows = TOTAL_ROWS - (num_filters % TOTAL_ROWS);
                    batch_idle_rows = batch_idle_rows == 16 ? 0 : batch_idle_rows;
                    idle_pe->value[layer_it][n] = (uint64_t)(batch_idle_rows * ceil(act_channels/(double)N_LANES));
                    baseline_cycles->value[layer_it][n] = base_cycles;
                    speedup->value[layer_it][n] = base_cycles / (double)cycles->value[layer_it][n];

                }

                idle_rows->value[layer_it][n] = TOTAL_ROWS - filters_per_tile * layer_rows_per_wgt;
                rows_per_wgt->value[layer_it][n] = layer_rows_per_wgt;
                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = wgt_layer_prec;

            }

        }

        speedup->special_value = sys::get_total(baseline_cycles->value) / (double)sys::get_total(cycles->value);

        //Dump statistics
        std::string header = "DynamicStripes Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(N_TILES) + "\n";
        header += "Number of values per group: " + std::to_string(PRECISION_GRANULARITY) + "\n";
        std::string ldn_bit = LEADING_BIT ? "True" : "False";
        header += "Calculate only leading bit: " + ldn_bit + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(COLUMN_REGISTERS) + "\n";
        header += "Size of the PE in bits: " + std::to_string(BITS_PE) + "\n";
        std::string dffy = DIFFY ? "True" : "False";
        header += "Diffy: " + dffy + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);


    }

    /* POTENTIALS */

    template <typename T>
    void DynamicStripes<T>::potentials(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "DynamicStripes_potentials";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto work_reduction = stats.register_double_t("work_reduction", 0, sys::Average);
        auto speedup = stats.register_double_t("speedup", 0, sys::Average);
        auto par_mult = stats.register_double_t("parallel_multiplication", 0, sys::AverageTotal);
        auto bit_multiplications = stats.register_uint_t("bit_multiplications", 0, sys::AverageTotal);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();

            base::Array<T> wgt = layer.getWeights();
            if(conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

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

            long out_x = (Nx - Kx + 2*padding)/stride + 1;
            long out_y = (Ny - Ky + 2*padding)/stride + 1;

            // Get layer precision
            auto act_layer_prec = layer.getActPrecision();

            auto network_bits = network.getNetwork_bits();

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                bit_counter = (uint64_t)computeDynamicStripesBitsPE((uint8_t)act_layer_prec,network_bits);
                bit_counter *= conv ? out_x * out_y * Kx * Ky * wgt_channels * num_filters:
                               wgt_channels * num_filters * R;

                bit_multiplications->value[layer_it][n] = bit_counter;
                work_reduction->value[layer_it][n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
                speedup->value[layer_it][n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
                par_mult->value[layer_it][n] = parallel_mult;
                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = layer.getWgtPrecision();
            }

        }

        //Dump statistics
        std::string header = "DynamicStripes Potentials/Work Reduction for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);
    }

    /* AVERAGE WIDTH */

    template <typename T>
    std::vector<double> DynamicStripes<T>::computeAvgWidthDynamicStripesActTile(int batch, int recursion,
            const std::vector<int> &list_act_x, const std::vector<int> &list_act_y, int kernel_x, int kernel_y,
            int init_channel, int stride, const base::Array<T> &padded_act, int max_channel, uint16_t act_mask,
            bool lstm) {

        int N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        int WINDOWS_PER_GROUP = N_COLUMNS / N_GROUPS;

        // Activations
        std::vector<double> act_width = std::vector<double>(std::min(N_GROUPS,(int)list_act_x.size()), 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_bit = 0, min_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {
            auto act_x = list_act_x[window];
            auto act_y = list_act_y[window];

            if(group_counter == WINDOWS_PER_GROUP)  {
                max_bit = 0, min_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for(int channel = init_channel; channel < std::min(init_channel + (int)N_LANES,max_channel); channel++) {

                uint16_t act_bits;
                if(lstm)
                    act_bits = padded_act.get(recursion, batch, channel);
                else
                    act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x, stride * act_y + kernel_y);


                if((act_bits & act_mask) != 0) {
                    act_bits = act_bits & ~act_mask;
                }

                const auto &min_max_act_bits = this->minMax(act_bits);

                auto min_act_bit = std::get<0>(min_max_act_bits);
                auto max_act_bit = std::get<1>(min_max_act_bits);

                max_act_bit += 1;

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }

            group_counter++;
            if(group_counter == WINDOWS_PER_GROUP) {
                if(!LEADING_BIT) act_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                else act_width[group_index] = max_bit + 1;
            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(!LEADING_BIT) act_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
            else act_width[group_index] = max_bit + 1;
        }

        return act_width;

    }

    template <typename T>
    std::vector<double> DynamicStripes<T>::computeAvgWidthDynamicStripesWgtTile(int kernel_x, int kernel_y,
            int init_channel, int init_filter, const base::Array<T> &wgt, int max_channel, int max_filter,
            uint16_t wgt_mask) {

        int N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        int WINDOWS_PER_GROUP = N_COLUMNS / N_GROUPS;

        // Weights
        std::vector<double> wgt_width = std::vector<double>(std::min(N_GROUPS,max_filter-init_filter), 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_bit = 0, min_bit = 16;
        for (int filter = init_filter; filter < std::min(init_filter + (int)N_ROWS, max_filter); filter++) {

            if(group_counter == WINDOWS_PER_GROUP)  {
                max_bit = 0, min_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for(int channel = init_channel; channel < std::min(init_channel + (int)N_LANES,max_channel); channel++) {

                uint16_t wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

                if((wgt_bits & wgt_mask) != 0) {
                    wgt_bits = wgt_bits & ~wgt_mask;
                }

                const auto &min_max_wgt_bits = this->minMax(wgt_bits);

                auto min_wgt_bit = std::get<0>(min_max_wgt_bits);
                auto max_wgt_bit = std::get<1>(min_max_wgt_bits);

                max_wgt_bit += 1;

                if(min_wgt_bit < min_bit) min_bit = min_wgt_bit;
                if(max_wgt_bit > max_bit) max_bit = max_wgt_bit;

            }

            group_counter++;
            if(group_counter == WINDOWS_PER_GROUP) {
                if(!LEADING_BIT) wgt_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                else wgt_width[group_index] = max_bit + 1;
            }
        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(!LEADING_BIT) wgt_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
            else wgt_width[group_index] = max_bit + 1;
        }

        return wgt_width;

    }

    template <typename T>
    void DynamicStripes<T>::average_width(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "DynamicStripes_L" + std::to_string(N_LANES) + "_C" + std::to_string(N_COLUMNS) + "_R" +
                std::to_string(N_ROWS) + "_PG" + std::to_string(PRECISION_GRANULARITY) + (LEADING_BIT ? "_LB" : "") +
                "_average_width";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto act_avg_width = stats.register_double_t("act_avg_width", 0, sys::Average);
        auto act_width_reduction = stats.register_double_t("act_width_reduction", 0, sys::Average);
        auto act_bits_baseline = stats.register_uint_t("act_bits_baseline", 0, sys::AverageTotal);
        auto act_bits_profiled = stats.register_uint_t("act_bits_profiled", 0, sys::AverageTotal);
        auto act_bits_datawidth = stats.register_uint_t("act_bits_datawidth", 0, sys::AverageTotal);
        auto act_bits_scnn = stats.register_uint_t("act_bits_scnn", 0, sys::AverageTotal);
        auto act_bits_fpc = stats.register_uint_t("act_bits_fpc", 0, sys::AverageTotal);
        auto act_bits_bdi = stats.register_uint_t("act_bits_bdi", 0, sys::AverageTotal);
        auto act_bits_bdi_opt = stats.register_uint_t("act_bits_bdi_opt", 0, sys::AverageTotal);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_avg_width = stats.register_double_t("wgt_avg_width", 0, sys::Average);
        auto wgt_width_reduction = stats.register_double_t("wgt_width_reduction", 0, sys::Average);
        auto wgt_bits_baseline = stats.register_uint_t("wgt_bits_baseline", 0, sys::AverageTotal);
        auto wgt_bits_profiled = stats.register_uint_t("wgt_bits_profiled", 0, sys::AverageTotal);
        auto wgt_bits_datawidth = stats.register_uint_t("wgt_bits_datawidth", 0, sys::AverageTotal);
        auto wgt_bits_scnn = stats.register_uint_t("wgt_bits_scnn", 0, sys::AverageTotal);
        auto wgt_bits_fpc = stats.register_uint_t("wgt_bits_fpc", 0, sys::AverageTotal);
        auto wgt_bits_bdi = stats.register_uint_t("wgt_bits_bdi", 0, sys::AverageTotal);
        auto wgt_bits_bdi_opt = stats.register_uint_t("act_bits_bdi_opt", 0, sys::AverageTotal);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto network_bits = network.getNetwork_bits();
        auto act_width_need = stats.register_double_dist_t("act_width_need",0,network_bits,0.0,sys::Average);
        auto wgt_width_need = stats.register_double_dist_t("wgt_width_need",0,network_bits,0.0,sys::Average);
        auto act_distr = stats.register_double_dist_t("act_width_need",0,pow(2,network_bits),0.0,sys::Total);
        auto wgt_distr = stats.register_double_dist_t("wgt_width_need",0,pow(2,network_bits),0.0,sys::Total);
        auto signed_activations = !network.isUnsignedAct();
        auto signed_weights = !network.isUnsignedWgt();

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            //if (layer_it != 0) signed_activations = false;

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            act.sign_magnitude_representation(layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();
            if(act.getDimensions() == 2) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            wgt.sign_magnitude_representation(layer.getWgtPrecision());
            if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if(conv) act.zero_pad(padding);

            if(act.getShape()[1] == 3 && stride > 1) {
                act.reshape_first_layer_act((uint16_t)stride);
                wgt.reshape_first_layer_wgt((uint16_t)stride);
                stride = 1;
            }

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            uint64_t batch_size, act_channels, Nx, Ny, R;
            if(lstm) {
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
            if(this->FAST_MODE) batch_size = 1;

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            auto act_layer_prec = layer.getActPrecision();
            auto act_mask = (uint16_t)(1u << (act_layer_prec - 1));

            auto wgt_layer_prec = layer.getWgtPrecision();
            auto wgt_mask = (uint16_t)(1u << (wgt_layer_prec - 1));

            // Activations
            for(int n=0; n<batch_size; n++) {

                std::vector<double> act_width;

                /*std::vector<int> list_x, list_y;
                int x_counter = 0, y_counter = 0;
                std::vector<double> act_width;

                for(int r = 0; r < R; r++) {
                    while (this->iterateWindows(out_x, out_y, list_x, list_y, x_counter, y_counter, N_COLUMNS)) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = 0; k < act_channels; k += N_LANES) {
                                    auto tile_act_width = computeAvgWidthDynamicStripesActTile(n, r, list_x, list_y, i,
                                            j, k, stride, act, (int)act_channels, act_mask, lstm);
                                    act_width.insert(act_width.end(), tile_act_width.begin(), tile_act_width.end());
                                }
                            }
                        }
                    }
                }*/

                for(int r = 0; r < R; r++) {
                    for (int ch = 0; ch < act_channels; ++ch) {
                        for (int x = 0; x < Nx; ++x) {
                            for (int y = 0; y < Ny; ++y) {
                                auto act_bits = lstm ? act.get(r, n, ch) : act.get(n, ch, x, y);

                                if (act_bits == 0) continue;

                                if (signed_activations) {
                                    if ((act_bits & act_mask) != 0) {
                                        act_bits = act_bits & ~act_mask;
                                        //int pos = 128 - act_bits;
                                        //act_distr->value[pos][layer_it][n]++;
                                    } else {
                                        //int pos = act_bits + 127;
                                        //act_distr->value[pos][layer_it][n]++;
                                    }
                                } else {
                                    act_distr->value[act_bits][layer_it][n]++;
                                }


                                const auto &min_max_act_bits = this->minMax(act_bits);
                                auto max_act_bit = std::get<1>(min_max_act_bits);
                                if (signed_activations) max_act_bit += 1;

                                act_width.push_back((max_act_bit + 1));
                            }
                        }
                    }
                }

                double batch_act_avg_width = sys::get_average(act_width);

                // Calculate bits needed
                std::vector<uint64_t> batch_act_width_need (network_bits + 1, 0);
                std::vector<double> act_width_need_per (network_bits + 1 ,0);
                for(auto act_group : act_width)
                    for(auto a = (int)act_group; a <= network_bits; a++)
                        batch_act_width_need[a]++;
                for(int a = 0; a < batch_act_width_need.size(); a++)
                    act_width_need_per[a] = batch_act_width_need[a] / (double)act_width.size() * 100.;

                act_avg_width->value[layer_it][n] = batch_act_avg_width;
                act_width_reduction->value[layer_it][n] = (act_layer_prec - batch_act_avg_width) * 100. / act_layer_prec;

                for(int i = 0; i <= network_bits; i++)
                    act_width_need->value[i][layer_it][n] = act_width_need_per[i];

            }

            for(int n = 0; n < batch_size; n++) {

                uint64_t batch_act_bits_datawidth = 0;
                uint64_t batch_act_bits_datawidth_non_zeros = 0;
                for(int r = 0; r < R; r++) {
                    for (int k = 0; k < act_channels; k += N_LANES) {
                        for (int j = 0; j < Ny; j++) {
                            for (int i = 0; i < Nx; i++) {
                                uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                                for(int channel = k; channel < std::min(k + (int)N_LANES,(int)act_channels); channel++) {
                                    uint16_t act_bits;
                                    if(lstm)
                                        act_bits = act.get(r, n, channel);
                                    else
                                        act_bits = act.get(n, channel, i, j);

                                    if(act_bits != 0) non_zeroes++;

                                    bool neg = false;
                                    if((act_bits & act_mask) != 0) {
                                        act_bits = act_bits & ~act_mask;
                                        neg = true;
                                    }

                                    const auto &min_max_act_bits = this->minMax(act_bits);

                                    auto min_act_bit = std::get<0>(min_max_act_bits);
                                    auto max_act_bit = std::get<1>(min_max_act_bits);

                                    if(neg) max_act_bit += 1;

                                    if(min_act_bit < min_bit) min_bit = min_act_bit;
                                    if(max_act_bit > max_bit) max_bit = max_act_bit;

                                }
                                int width;
                                if(!LEADING_BIT) width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                                else width = max_bit + 1;
                                batch_act_bits_datawidth_non_zeros += (width * non_zeroes);
                                batch_act_bits_datawidth += (width * N_LANES);
                            }
                        }
                    }
                }

                // Calculate data from off-chip
                auto num_act = R * Nx * Ny * act_channels;
                act_bits_baseline->value[layer_it][n] = num_act * network_bits;
                act_bits_profiled->value[layer_it][n] = 4 + num_act * act_layer_prec;
                auto overhead_non_zeros = (uint64_t)((16 + log2(network_bits)) * ceil(num_act / 16.));
                auto overhead = (uint64_t)(log2(network_bits) * ceil(num_act / 16.));
                act_bits_datawidth->value[layer_it][n] = std::min(overhead + batch_act_bits_datawidth,
                        overhead_non_zeros + batch_act_bits_datawidth_non_zeros);

            }

            // Weights
            std::vector<double> wgt_width;
            for(int m = 0; m < num_filters; m += N_ROWS) {

                /*for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = 0; k < wgt_channels; k += N_LANES) {
                            auto tile_wgt_width = computeAvgWidthDynamicStripesWgtTile(i,j,k,m,wgt,(int)wgt_channels,
                                    (int)num_filters, wgt_mask);
                            wgt_width.insert(wgt_width.end(),tile_wgt_width.begin(),tile_wgt_width.end());

                        }
                    }

                }*/

                 for (int ch = 0; ch < wgt_channels; ++ch) {
                     for (int x = 0; x < Kx; ++x) {
                         for (int y = 0; y < Ky; ++y) {
                             auto wgt_bits = wgt.get(m, ch, x, y);

                             if (wgt_bits == 0) continue;

                             if (signed_weights) {
                                 if ((wgt_bits & wgt_mask) != 0) {
                                     wgt_bits = wgt_bits & ~wgt_mask;
                                     int pos = 128 - wgt_bits;
                                     wgt_distr->value[pos][layer_it][0]++;
                                 } else {
                                     int pos = wgt_bits + 127;
                                     wgt_distr->value[pos][layer_it][0]++;
                                 }
                             } else {
                                 wgt_distr->value[wgt_bits][layer_it][0]++;
                             }


                             const auto &min_max_wgt_bits = this->minMax(wgt_bits);
                             auto max_wgt_bit = std::get<1>(min_max_wgt_bits);
                             if (signed_weights) max_wgt_bit += 1;

                             wgt_width.push_back((max_wgt_bit + 1));
                         }
                     }
                 }
            }

            uint64_t batch_wgt_bits_datawidth = 0;
            uint64_t batch_wgt_bits_datawidth_non_zeros = 0;
            for(int m = 0; m < num_filters; m++) {
                for (int k = 0; k < wgt_channels; k += N_LANES) {
                    for (int j = 0; j < Ky; j++) {
                        for (int i = 0; i < Kx; i++) {
                            uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                            for(int channel = k; channel < std::min(k + (int)N_LANES,(int)wgt_channels); channel++) {

                                uint16_t wgt_bits = wgt.get(m, channel, i, j);

                                if(wgt_bits != 0) non_zeroes++;

                                bool neg = false;
                                if((wgt_bits & wgt_mask) != 0) {
                                    wgt_bits = wgt_bits & ~wgt_mask;
                                    neg = true;
                                }

                                const auto &min_max_wgt_bits = this->minMax(wgt_bits);

                                auto min_wgt_bit = std::get<0>(min_max_wgt_bits);
                                auto max_wgt_bit = std::get<1>(min_max_wgt_bits);

                                if(neg) max_wgt_bit += 1;

                                if(min_wgt_bit < min_bit) min_bit = min_wgt_bit;
                                if(max_wgt_bit > max_bit) max_bit = max_wgt_bit;

                            }
                            int width;
                            if(!LEADING_BIT) width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                            else width = max_bit + 1;
                            batch_wgt_bits_datawidth_non_zeros += (width * non_zeroes);
                            batch_wgt_bits_datawidth += (width * N_LANES);
                        }
                    }

                }
            }

            double batch_wgt_avg_width = sys::get_average(wgt_width);

            // Calculate bits needed
            std::vector<uint64_t> batch_wgt_width_need (network_bits + 1, 0);
            std::vector<double> wgt_width_need_per (network_bits + 1, 0);
            for(auto wgt_group : wgt_width)
                for(auto w = (int)wgt_group; w <= network_bits; w++)
                    batch_wgt_width_need[w]++;
            for(int w = 0; w < batch_wgt_width_need.size(); w++)
                wgt_width_need_per[w] = batch_wgt_width_need[w] / (double)wgt_width.size() * 100.;

            for(int n = 0; n < batch_size; n++) {

                // Calculate data from off-chip
                auto num_wgt = wgt.getMax_index();
                wgt_bits_baseline->value[layer_it][n] = num_wgt * network_bits;
                wgt_bits_profiled->value[layer_it][n] = 4 + num_wgt * wgt_layer_prec;
                auto overhead_non_zeros = (uint64_t)((16 + log2(network_bits)) * ceil(num_wgt / 16.));
                auto overhead = (uint64_t)(log2(network_bits) * ceil(num_wgt / 16.));
                wgt_bits_datawidth->value[layer_it][n] = std::min(overhead + batch_wgt_bits_datawidth,
                        overhead_non_zeros + batch_wgt_bits_datawidth_non_zeros);

                wgt_avg_width->value[layer_it][n] = batch_wgt_avg_width;
                wgt_width_reduction->value[layer_it][n] = (wgt_layer_prec - batch_wgt_avg_width) * 100. / wgt_layer_prec;

                for (int i = 0; i <= network_bits; i++) {
                    wgt_width_need->value[i][layer_it][n] = wgt_width_need_per[i];
                }
            }

            // SCNN
            for(int n = 0; n < batch_size; n++) {

                int skips = 0;
                uint64_t batch_act_bits_scnn = 0;
                for(int r = 0; r < R; r++) {
                    for (int k = 0; k < act_channels; k++) {
                        for (int y = 0; y < Ny; y++) {
                            for (int x = 0; x < Nx; x++) {
                                T act_bits;
                                if(lstm) act_bits = act.get(r, n, k);
                                else act_bits = act.get(n, k, x, y);
                                if (act_bits != 0) {
                                    batch_act_bits_scnn += network_bits + 4;
                                    skips = 0;
                                } else {
                                    skips++;
                                    if (skips == 16) {
                                        batch_act_bits_scnn += network_bits + 4;
                                        skips = 0;
                                    }
                                }
                            }
                        }
                    }
                }

                act_bits_scnn->value[layer_it][n] = batch_act_bits_scnn;
            }

            int skips = 0;
            uint64_t batch_wgt_bits_scnn = 0;
            for(int m = 0; m < num_filters; m++) {

                for (int k = 0; k < wgt_channels; k++) {
                    for (int y = 0; y < Ky; y++) {
                        for (int x = 0; x < Kx; x++) {
                            auto act_bits = wgt.get(m, k, x, y);
                            if(act_bits != 0) {
                                batch_wgt_bits_scnn += network_bits + 4;
                                skips = 0;
                            } else {
                                skips++;
                                if(skips == 16) {
                                    batch_wgt_bits_scnn += network_bits + 4;
                                    skips = 0;
                                }
                            }
                        }
                    }
                }
            }

            for(int n = 0; n < batch_size; n++)
                wgt_bits_scnn->value[layer_it][n] = batch_wgt_bits_scnn;

            // FPC
            auto value_per_block = 4;
            for(int n = 0; n < batch_size; n++) {

                uint64_t batch_act_bits_fpc = 0;
                for(int r = 0; r < R; r++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int i = 0; i < Nx; i++) {
                            for (int k = 0; k < act_channels; k += value_per_block) {

                                int count = 0;
                                int bits = 0;
                                for(int channel = k; channel < (k + (int)value_per_block); channel++) {

                                    if (channel >= act_channels) {
                                        continue;
                                    }

                                    uint16_t act_bits;
                                    if(lstm)
                                        act_bits = act.get(r, n, channel);
                                    else
                                        act_bits = act.get(n, channel, i, j);

                                    if ((count == 0 || count ==1) && act_bits != 0) {
                                        bits = 32;
                                        break;
                                    } else if (count == 2 && act_bits != 0) {
                                        bits = 16;
                                        break;
                                    } else if (count == 3 && act_bits != 0) {
                                        auto is_4bit = act_bits & 0xF0u;
                                        if (is_4bit == 0) bits = 4;
                                        else bits = 8;
                                    } else if (count == 3) {
                                        bits = 3;
                                    }

                                    count ++;
                                }

                                batch_act_bits_fpc += bits + 3;


                            }
                        }
                    }
                }

                act_bits_fpc->value[layer_it][n] = batch_act_bits_fpc;

            }

            uint64_t batch_wgt_bits_fpc = 32 / network_bits;
            for(int m = 0; m < num_filters; m++) {
                for (int k = 0; k < wgt_channels; k += value_per_block) {
                    for (int j = 0; j < Ky; j++) {
                        for (int i = 0; i < Kx; i++) {

                            int count = 0;
                            int bits = 0;
                            for(int channel = k; channel < (k + (int)value_per_block); channel++) {

                                if (channel >= wgt_channels) {
                                    continue;
                                }

                                uint16_t wgt_bits = wgt.get(m, channel, i, j);

                                if ((count == 0 || count == 1) && wgt_bits != 0) {
                                    bits = 32;
                                    break;
                                } else if (count == 2 && wgt_bits != 0) {
                                    bits = 16;
                                    break;
                                } else if (count == 3 && wgt_bits != 0) {
                                    auto is_4bit = wgt_bits & 0xF0u;
                                    if (is_4bit == 0) bits = 4;
                                    else bits = 8;
                                } else if (count == 3) {
                                    bits = 3;
                                }
                                count++;

                            }

                            batch_wgt_bits_fpc += bits + 3;

                        }
                    }

                }
            }

            for(int n = 0; n < batch_size; n++)
                wgt_bits_fpc->value[layer_it][n] = batch_wgt_bits_fpc;

            // BDI
            auto value_per_block_bdi = 64 * 8 / network_bits;
            for(int n = 0; n < batch_size; n++) {

                uint64_t batch_act_bits_bdi = 0;
                for(int r = 0; r < R; r++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int i = 0; i < Nx; i++) {
                            for (int k = 0; k < act_channels; k += value_per_block_bdi) {


                                bool all_zero = true;
                                for(int channel = k; channel < (k + (int)value_per_block_bdi); channel++) {

                                    if (channel >= act_channels) {
                                        continue;
                                    }

                                    uint16_t act_bits;
                                    if(lstm)
                                        act_bits = act.get(r, n, channel);
                                    else
                                        act_bits = act.get(n, channel, i, j);

                                    if (act_bits != 0) all_zero = false;

                                }

                                if (all_zero) {
                                    batch_act_bits_bdi += value_per_block_bdi + 4;
                                } else {
                                    batch_act_bits_bdi += value_per_block_bdi * network_bits + 4;
                                }

                            }
                        }
                    }
                }

                act_bits_bdi->value[layer_it][n] = batch_act_bits_bdi;

            }

            uint64_t batch_wgt_bits_bdi = 0;
            for(int m = 0; m < num_filters; m++) {
                for (int k = 0; k < wgt_channels; k += value_per_block_bdi) {
                    for (int j = 0; j < Ky; j++) {
                        for (int i = 0; i < Kx; i++) {

                            bool all_zero = true;
                            for(int channel = k; channel < (k + (int)value_per_block_bdi); channel++) {

                                if (channel >= wgt_channels) {
                                    continue;
                                }

                                uint16_t wgt_bits = wgt.get(m, channel, i, j);

                                if (wgt_bits != 0) all_zero = false;

                            }

                            if (all_zero) {
                                batch_wgt_bits_bdi += value_per_block_bdi + 4;
                            } else {
                                batch_wgt_bits_bdi += value_per_block_bdi * network_bits + 4;
                            }
                        }
                    }

                }
            }

            for(int n = 0; n < batch_size; n++)
                wgt_bits_bdi->value[layer_it][n] = batch_wgt_bits_bdi;


            // BDI OPT
            uint8_t act_data_pt = 0u;
            for(int n = 0; n < batch_size; n++) {

                uint64_t batch_act_bits_bdi_opt = 0;
                for(int r = 0; r < R; r++) {
                    for (int j = 0; j < Ny; j++) {
                        for (int i = 0; i < Nx; i++) {
                            for (int k = 0; k < act_channels; k += PRECISION_GRANULARITY) {

                                if (act_data_pt >= network_bits) {
                                    act_data_pt %= network_bits;
                                }

                                uint64_t delta_0 = 0, delta_1 = 0;

                                for(int channel = k; channel < (k + (int)PRECISION_GRANULARITY); channel++) {

                                    if (channel >= act_channels) {
                                        continue;
                                    }

                                    uint16_t act_bits;
                                    if (lstm)
                                        act_bits = act.get(r, n, channel);
                                    else
                                        act_bits = act.get(n, channel, i, j);

                                    if (act_bits > delta_1)
                                        delta_1 = act_bits;
                                }

                                bool all_zero = true;
                                uint64_t max_width = 0;
                                for(int channel = k; channel < (k + (int)PRECISION_GRANULARITY); channel++) {

                                    if (channel >= act_channels) {
                                        continue;
                                    }

                                    uint16_t act_bits;
                                    if(lstm)
                                        act_bits = act.get(r, n, channel);
                                    else
                                        act_bits = act.get(n, channel, i, j);

                                    if (act_bits != 0) all_zero = false;

                                    auto value_0 = act_bits - delta_0;
                                    auto value_1 = delta_1 - act_bits;

                                    const auto &min_max_val_0_bits = this->minMax(value_0);
                                    const auto &min_max_val_1_bits = this->minMax(value_1);

                                    auto max_val_0_bit = std::get<1>(min_max_val_0_bits);
                                    auto max_val_1_bit = std::get<1>(min_max_val_1_bits);

                                    auto max_val_bit = std::min(max_val_0_bit, max_val_1_bit);

                                    if (max_val_bit <= 2 && max_width < 2) {
                                        max_width = 2;
                                    } else if (max_val_bit >= 2 && max_val_bit <= 4 && max_width < 4) {
                                        max_width = 4;
                                    } else if (max_val_bit >= 4) {
                                        max_width = 8;
                                    }

                                }

                                if (all_zero) {
                                    batch_act_bits_bdi_opt += PRECISION_GRANULARITY + 4;
                                    act_data_pt += 1;
                                } else {

                                    uint64_t compressed_size = max_width * PRECISION_GRANULARITY + network_bits + 4;
                                    uint64_t baseline_size = network_bits * PRECISION_GRANULARITY + 4;

                                    if (compressed_size < baseline_size) {
                                        batch_act_bits_bdi_opt += compressed_size;
                                        act_data_pt += max_width;
                                    }
                                    else batch_act_bits_bdi_opt += baseline_size;
                                }


                            }

                            if (act_data_pt >= network_bits) {
                                act_data_pt %= network_bits;
                            }

                            // Padding overhead
                            if (act_data_pt != 0) {

                                if (act_data_pt >= network_bits) {
                                    act_data_pt %= network_bits;
                                }

                                batch_act_bits_bdi_opt += (network_bits - act_data_pt) * PRECISION_GRANULARITY;
                                act_data_pt = 0;
                            }
                        }
                    }
                }

                batch_act_bits_bdi_opt += Ny * 32;
                act_bits_bdi_opt->value[layer_it][n] = batch_act_bits_bdi_opt;

            }

            uint8_t wgt_data_pt = 0u;
            uint64_t batch_wgt_bits_bdi_opt = 0;
            for(int m = 0; m < num_filters; m++) {
                for (int k = 0; k < wgt_channels; k += PRECISION_GRANULARITY) {
                    for (int j = 0; j < Ky; j++) {
                        for (int i = 0; i < Kx; i++) {

                            if (wgt_data_pt >= network_bits) {
                                wgt_data_pt %= network_bits;
                            }

                            uint64_t delta_0 = 0, delta_1 = 0;

                            for (int channel = k; channel < (k + (int) PRECISION_GRANULARITY); channel++) {

                                if (channel >= wgt_channels) {
                                    continue;
                                }

                                uint16_t wgt_bits = wgt.get(m, channel, i, j);

                                if (wgt_bits > delta_1)
                                    delta_1 = wgt_bits;
                            }

                            bool all_zero = true;
                            uint64_t max_width = 0;
                            for (int channel = k; channel < (k + (int) PRECISION_GRANULARITY); channel++) {

                                if (channel >= wgt_channels) {
                                    continue;
                                }

                                uint16_t wgt_bits = wgt.get(m, channel, i, j);


                                if (wgt_bits != 0) all_zero = false;

                                auto value_0 = wgt_bits - delta_0;
                                auto value_1 = delta_1 - wgt_bits;

                                const auto &min_max_val_0_bits = this->minMax(value_0);
                                const auto &min_max_val_1_bits = this->minMax(value_1);

                                auto max_val_0_bit = std::get<1>(min_max_val_0_bits);
                                auto max_val_1_bit = std::get<1>(min_max_val_1_bits);

                                auto max_val_bit = std::min(max_val_0_bit, max_val_1_bit);

                                if (max_val_bit <= 2 && max_width < 2) {
                                    max_width = 2;
                                } else if (max_val_bit >= 2 && max_val_bit <= 4 && max_width < 4) {
                                    max_width = 4;
                                } else if (max_val_bit >= 4) {
                                    max_width = 8;
                                }

                            }

                            if (all_zero) {
                                batch_wgt_bits_bdi_opt += PRECISION_GRANULARITY + 4;
                            } else {

                                uint64_t compressed_size = max_width * PRECISION_GRANULARITY + network_bits + 4;
                                uint64_t baseline_size = network_bits * PRECISION_GRANULARITY + 4;

                                if (compressed_size < baseline_size) {
                                    batch_wgt_bits_bdi_opt += compressed_size;
                                    wgt_data_pt += max_width;
                                } else batch_wgt_bits_bdi_opt += baseline_size;
                            }

                        }
                    }

                }

                // Padding overhead
                if (wgt_data_pt != 0) {

                    if (wgt_data_pt >= network_bits) {
                        wgt_data_pt %= network_bits;
                    }

                    batch_wgt_bits_bdi_opt += (network_bits - wgt_data_pt) * PRECISION_GRANULARITY;
                    wgt_data_pt = 0;
                }

            }

            batch_wgt_bits_bdi_opt += num_filters * 32;

            for(int n = 0; n < batch_size; n++)
                wgt_bits_bdi_opt->value[layer_it][n] = batch_wgt_bits_bdi_opt;

        }

        //Dump statistics
        std::string header = "DynamicStripes Average Width for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of values per group: " + std::to_string(PRECISION_GRANULARITY) + "\n";
        std::string ldn_bit = LEADING_BIT ? "True" : "False";
        header += "Calculate only leading bit: " + ldn_bit + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template <typename T>
    void DynamicStripes<T>::layer_fusion(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "DynamicStripes_layer_fusion";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto max_act_bits_baseline = stats.register_uint_t("act_bits_baseline", 0, sys::Max);
        auto max_act_bits_datawidth = stats.register_uint_t("act_bits_datawidth", 0, sys::Max);

        auto network_bits = network.getNetwork_bits();
        auto signed_activations0 = !network.isUnsignedAct();
        auto signed_activations1 = !network.isUnsignedAct();
        auto signed_activations2 = !network.isUnsignedAct();

        for(auto layer_it = 0; layer_it < network.getNumLayers(); layer_it += 2) {

            int layer_it_1 = layer_it + 1;
            int layer_it_2 = layer_it + 2;

            if (layer_it_1 >= network.getNumLayers() || layer_it_2 >= network.getNumLayers()) continue;

            if (layer_it != 0) signed_activations0 = false;
            if (layer_it_1 != 0) signed_activations1 = false;
            if (layer_it_2 != 0) signed_activations2 = false;

            const base::Layer<T> &layer0 = network.getLayers()[layer_it];
            bool conv0 = layer0.getType() == "Convolution";
            bool lstm0 = layer0.getType() == "LSTM";
            bool fc0 = layer0.getType() == "InnerProduct";

            if (!conv0) continue;

            base::Array<T> act0 = layer0.getActivations();
            act0.sign_magnitude_representation(layer0.getActPrecision());
            if(act0.getDimensions() == 2) act0.reshape_to_4D();

            base::Array<T> wgt0 = layer0.getWeights();
            wgt0.sign_magnitude_representation(layer0.getWgtPrecision());
            if(wgt0.getDimensions() == 2) wgt0.reshape_to_4D();

            int padding0 = layer0.getPadding();
            int stride0 = layer0.getStride();

            act0.zero_pad(padding0);

            if(act0.getShape()[1] == 3 && stride0 > 1) {
                act0.reshape_first_layer_act((uint16_t)stride0);
                wgt0.reshape_first_layer_wgt((uint16_t)stride0);
                stride0 = 1;
            }

            const std::vector<size_t> &act_shape0 = act0.getShape();
            const std::vector<size_t> &wgt_shape0 = wgt0.getShape();

            auto batch_size0 = act_shape0[0];
            auto act_channels0 = act_shape0[1];
            auto Nx0 = act_shape0[2];
            auto Ny0 = act_shape0[3];
            if(this->FAST_MODE) batch_size0 = 1;

            auto num_filters0 = wgt_shape0[0];
            auto wgt_channels0 = wgt_shape0[1];
            auto Kx0 = wgt_shape0[2];
            auto Ky0 = wgt_shape0[3];

            long out_x0 = (Nx0 - Kx0)/stride0 + 1;
            long out_y0 = (Ny0 - Ky0)/stride0 + 1;

            auto act_layer_prec0 = layer0.getActPrecision();
            auto act_mask0 = (uint16_t)(1u << (act_layer_prec0 - 1));

            auto wgt_layer_prec0 = layer0.getWgtPrecision();
            auto wgt_mask0 = (uint16_t)(1u << (wgt_layer_prec0 - 1));

            const base::Layer<T> &layer1 = network.getLayers()[layer_it_1];
            bool conv1 = layer1.getType() == "Convolution";
            bool lstm1 = layer1.getType() == "LSTM";
            bool fc1 = layer1.getType() == "InnerProduct";

            if (!conv1) continue;

            base::Array<T> act1 = layer1.getActivations();
            act1.sign_magnitude_representation(layer1.getActPrecision());
            if(act1.getDimensions() == 2) act1.reshape_to_4D();

            base::Array<T> wgt1 = layer1.getWeights();
            wgt1.sign_magnitude_representation(layer1.getWgtPrecision());
            if(wgt1.getDimensions() == 2) wgt1.reshape_to_4D();

            int padding1 = layer1.getPadding();
            int stride1 = layer1.getStride();

            act1.zero_pad(padding1);

            const std::vector<size_t> &act_shape1 = act1.getShape();
            const std::vector<size_t> &wgt_shape1 = wgt1.getShape();

            auto batch_size1 = act_shape1[0];
            auto act_channels1 = act_shape1[1];
            auto Nx1 = act_shape1[2];
            auto Ny1 = act_shape1[3];
            if(this->FAST_MODE) batch_size1 = 1;

            auto num_filters1 = wgt_shape1[0];
            auto wgt_channels1 = wgt_shape1[1];
            auto Kx1 = wgt_shape1[2];
            auto Ky1 = wgt_shape1[3];

            long out_x1 = (Nx1 - Kx1)/stride1 + 1;
            long out_y1 = (Ny1 - Ky1)/stride1 + 1;

            auto act_layer_prec1 = layer1.getActPrecision();
            auto act_mask1 = (uint16_t)(1u << (act_layer_prec1 - 1));

            auto wgt_layer_prec1 = layer1.getWgtPrecision();
            auto wgt_mask1 = (uint16_t)(1u << (wgt_layer_prec1 - 1));

            const base::Layer<T> &layer2 = network.getLayers()[layer_it_2];
            bool conv2 = layer2.getType() == "Convolution";
            bool lstm2 = layer2.getType() == "LSTM";
            bool fc2 = layer2.getType() == "InnerProduct";

            if (!conv2) continue;

            base::Array<T> act2 = layer2.getActivations();
            act2.sign_magnitude_representation(layer2.getActPrecision());
            if(act2.getDimensions() == 2) act2.reshape_to_4D();

            base::Array<T> wgt2 = layer2.getWeights();
            wgt2.sign_magnitude_representation(layer2.getWgtPrecision());
            if(wgt2.getDimensions() == 2) wgt2.reshape_to_4D();

            int padding2 = layer2.getPadding();
            int stride2 = layer2.getStride();

            const std::vector<size_t> &act_shape2 = act2.getShape();
            const std::vector<size_t> &wgt_shape2 = wgt2.getShape();

            auto batch_size2 = act_shape2[0];
            auto act_channels2 = act_shape2[1];
            auto Nx2 = act_shape2[2];
            auto Ny2 = act_shape2[3];
            if(this->FAST_MODE) batch_size2 = 1;

            auto num_filters2 = wgt_shape2[0];
            auto wgt_channels2 = wgt_shape2[1];
            auto Kx2 = wgt_shape2[2];
            auto Ky2 = wgt_shape2[3];

            long out_x2 = (Nx2 - Kx2)/stride2 + 1;
            long out_y2 = (Ny2 - Ky2)/stride2 + 1;

            auto act_layer_prec2 = layer2.getActPrecision();
            auto act_mask2 = (uint16_t)(1u << (act_layer_prec2 - 1));

            auto wgt_layer_prec2 = layer2.getWgtPrecision();
            auto wgt_mask2 = (uint16_t)(1u << (wgt_layer_prec2 - 1));


            auto pool_1 = out_x0 / (Nx1 - 2 * padding1);
            auto pool_2 = out_x1 / Nx2;

            for (int n = 0; n < batch_size0; ++n) {

                uint64_t max_baseline = 0;
                uint64_t max_datawidth = 0;
                for (int oy = 0; oy < out_y2; ++oy) {
                    for (int ox = 0; ox < out_x2; ++ox) {

                        // Sizes

                        std::set<std::tuple<int, int, int, int>> accesses; // Layer-Channel-X-Y

                        uint64_t max_x1 = 0, max_y1 = 0, max_x0 = 0, max_y0 = 0;

                        for (int y2 = 0; y2 < Ky2; ++y2) {
                            for (int x2 = 0; x2 < Kx2; ++x2) {

                                auto x2_pos = ox * stride2 + x2;
                                auto y2_pos = oy * stride2 + y2;

                                // Compress channels
                                for (int ch2 = 0; ch2 < act_channels2; ++ch2) {
                                    auto tuple = std::make_tuple(2, x2_pos, y2_pos, ch2);
                                    accesses.emplace(tuple);
                                }

                                for (int y1 = 0; y1 < (Ky1 + (pool_2 - 1) * stride1); ++y1) {
                                    for (int x1 = 0; x1 < (Kx1 + (pool_2 - 1) * stride1); ++x1) {

                                        auto x1_pos = x2_pos * pool_2 * stride1 + x1;
                                        auto y1_pos = y2_pos * pool_2 * stride1 + y1;

                                        // Compress values
                                        for (int ch1 = 0; ch1 < act_channels1; ++ch1) {
                                            auto tuple = std::make_tuple(1, x1_pos, y1_pos, ch1);
                                            if (x1_pos > max_x1) max_x1 = x1_pos;
                                            if (y1_pos > max_y1) max_y1 = y1_pos;

                                            accesses.emplace(tuple);
                                        }

                                        for (int y0 = 0; y0 < (Ky0 + (pool_1 - 1) * stride0); ++y0) {
                                            for (int x0 = 0; x0 < (Kx0 + (pool_1 - 1) * stride0); ++x0) {

                                                auto x0_pos = x1_pos * pool_1 * stride0 + x0;
                                                auto y0_pos = y1_pos * pool_1 * stride0 + y0;

                                                // Compress values
                                                for (int ch0 = 0; ch0 < act_channels0; ++ch0) {
                                                    auto tuple = std::make_tuple(0, x0_pos, y0_pos, ch0);

                                                    if (x0_pos > max_x0) max_x0 = x0_pos;
                                                    if (y0_pos > max_y0) max_y0 = y0_pos;

                                                    accesses.emplace(tuple);
                                                }
                                            }
                                        }

                                    }
                                }

                            }
                        }


                        // Calculate sizes
                        uint64_t baseline_size = 0;
                        uint64_t datawidth_size = 0;

                        uint64_t old_layer = 0;
                        uint64_t old_x = 0;
                        uint64_t old_y = 0;

                        int it = 0;
                        int max_bit = 0;
                        uint64_t act_data_pt = 0;
                        for (const auto &tuple : accesses) {

                            auto layer = std::get<0>(tuple);
                            auto x = std::get<1>(tuple);
                            auto y = std::get<2>(tuple);
                            auto ch = std::get<3>(tuple);

                            uint16_t act_bits = 0;
                            bool signed_activations;
                            uint16_t act_mask;
                            if (layer == 0) {
                                if (x >= Nx0 || y >= Ny0) continue;
                                act_bits = act0.get(n, ch, x, y);
                                signed_activations = signed_activations0;
                                act_mask = act_mask0;
                            } else if (layer == 1) {
                                if (x >= Nx1 || y >= Ny1) continue;
                                act_bits = act1.get(n, ch, x, y);
                                signed_activations = signed_activations1;
                                act_mask = act_mask1;
                            } else if (layer == 2) {
                                act_bits = act2.get(n, ch, x, y);
                                signed_activations = signed_activations2;
                                act_mask = act_mask2;
                            } else {
                                throw std::runtime_error("It should get in here");
                            }

                            if (signed_activations) {
                                if ((act_bits & act_mask) != 0) {
                                    act_bits = act_bits & ~act_mask;
                                }
                            }

                            const auto &min_max_act_bits = this->minMax(act_bits);
                            auto max_act_bit = std::get<1>(min_max_act_bits);
                            if (signed_activations) max_act_bit += 1;

                            if (it >= PRECISION_GRANULARITY || layer != old_layer || x != old_x || y != old_y) {

                                uint8_t width = max_bit + 1u;

                                baseline_size += PRECISION_GRANULARITY * network_bits;
                                datawidth_size += PRECISION_GRANULARITY * width;
                                datawidth_size += log2(network_bits);

                                act_data_pt = (act_data_pt + width) % network_bits;

                                if (act_data_pt != 0 && (layer != old_layer || x != old_x || y != old_y)) {
                                    datawidth_size += (network_bits - act_data_pt) * PRECISION_GRANULARITY;
                                    act_data_pt = 0;
                                }

                                it = 0;
                                max_bit = 0;
                            }


                            if (max_act_bit > max_bit) max_bit = max_act_bit;

                            old_layer = layer;
                            old_x = x;
                            old_y = y;
                            it++;
                        }

                        if (max_bit != 0) {
                            uint8_t width = max_bit + 1u;

                            baseline_size += PRECISION_GRANULARITY * network_bits;
                            datawidth_size += PRECISION_GRANULARITY * width;
                            datawidth_size += log2(network_bits);

                            act_data_pt = (act_data_pt + width) % network_bits;

                            if (it != 0) {
                                datawidth_size += (network_bits - act_data_pt) * PRECISION_GRANULARITY;
                                act_data_pt = 0;
                            }
                        }

                        if (baseline_size > max_baseline) max_baseline = baseline_size;
                        if (datawidth_size > max_datawidth) max_datawidth = datawidth_size;

                    }
                }

                max_act_bits_baseline->value[layer_it][n] = max_baseline;
                max_act_bits_datawidth->value[layer_it][n] = max_datawidth;

            }
        }

        //Dump statistics
        std::string header = "DynamicStripes Layer Fusion for " + network.getName() + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    /* ON CHIP */

    uint16_t get_value(std::map<uint64_t, uint16_t> &memory_map, uint64_t block_offset, uint64_t mem_pointer,
            uint16_t width, int network_bits, int GROUP_SIZE) {

        if ((width + mem_pointer) > (network_bits - 1)) {

            uint16_t block = memory_map[block_offset];
            uint16_t next_block = memory_map[block_offset + GROUP_SIZE];

            uint16_t width_msb = (width + mem_pointer) % network_bits;
            uint16_t width_lsb = width - width_msb;
            uint16_t value = 0;

            uint16_t read_mask_lsb = ((1u << width_lsb) - 1u) << mem_pointer;
            uint16_t value_lsb = block & read_mask_lsb;
            value_lsb >>= mem_pointer;

            uint16_t read_mask_msb = (1u << width_msb) - 1u;
            uint16_t value_msb = next_block & read_mask_msb;
            value_msb <<= width_lsb;

            value = value_msb | value_lsb;
            return value;

        } else {

            uint16_t block = memory_map[block_offset];

            uint16_t read_mask = ((1u << width) - 1u) << mem_pointer;
            uint16_t value = block & read_mask;
            value >>= mem_pointer;
            return value;

        }

    }

    template <typename T>
    void DynamicStripes<T>::on_chip(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "DynamicStripes_on_chip_PG" + std::to_string(PRECISION_GRANULARITY);
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto act_baseline_size = stats.register_uint_t("act_baseline_size", 0, sys::AverageTotal);
        auto act_profiled_size = stats.register_uint_t("act_profiled_size", 0, sys::AverageTotal);
        auto act_profiled_padding = stats.register_uint_t("act_profiled_padding", 0, sys::AverageTotal);
        auto act_datawidth_size = stats.register_uint_t("act_datawidth_size", 0, sys::AverageTotal);
        auto act_datawidth_groups = stats.register_uint_t("act_datawidth_groups", 0, sys::AverageTotal);
        auto act_datawidth_groups_diff = stats.register_uint_t("act_datawidth_groups_diff", 0, sys::AverageTotal);
        auto act_datawidth_padding = stats.register_uint_t("act_datawidth_padding", 0, sys::AverageTotal);
        auto act_datawidth_overhead = stats.register_uint_t("act_datawidth_overhead", 0, sys::AverageTotal);
        auto act_datawidth_row_overhead = stats.register_uint_t("act_datawidth_row_overhead", 0, sys::AverageTotal);
        auto act_datawidth_max_overhead = stats.register_uint_t("act_datawidth_max_overhead", 0, sys::AverageTotal);
        auto act_max_rel_pointer = stats.register_uint_t("act_max_rel_pointer", 0, sys::Max);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);

        auto wgt_baseline_size = stats.register_uint_t("wgt_baseline_size", 0, sys::AverageTotal);
        auto wgt_profiled_size = stats.register_uint_t("wgt_profiled_size", 0, sys::AverageTotal);
        auto wgt_profiled_padding = stats.register_uint_t("wgt_profiled_padding", 0, sys::AverageTotal);
        auto wgt_datawidth_size = stats.register_uint_t("wgt_datawidth_size", 0, sys::AverageTotal);
        auto wgt_datawidth_groups = stats.register_uint_t("wgt_datawidth_groups", 0, sys::AverageTotal);
        auto wgt_datawidth_groups_diff = stats.register_uint_t("wgt_datawidth_groups_diff", 0, sys::AverageTotal);
        auto wgt_datawidth_padding = stats.register_uint_t("wgt_datawidth_padding", 0, sys::AverageTotal);
        auto wgt_datawidth_overhead = stats.register_uint_t("wgt_datawidth_overhead", 0, sys::AverageTotal);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto network_bits = network.getNetwork_bits();
        auto signed_activations = !network.isUnsignedAct();
        auto signed_weights = !network.isUnsignedWgt();

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            if (layer_it != 0) signed_activations = false;

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            act.sign_magnitude_representation(layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();
            if(act.getDimensions() == 2) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            wgt.sign_magnitude_representation(layer.getWgtPrecision());
            if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if(conv) act.zero_pad(padding);

            if(act.getShape()[1] == 3 && stride > 1) {
                act.reshape_first_layer_act((uint16_t)stride);
                wgt.reshape_first_layer_wgt((uint16_t)stride);
                stride = 1;
            }

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            uint64_t batch_size, act_channels, Nx, Ny, R;
            if(lstm) {
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
            if(this->FAST_MODE) batch_size = 1;

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            auto channels_per_column = (uint64_t)ceil(act_channels / (double)N_COLUMNS);

            auto act_layer_prec = layer.getActPrecision();
            auto act_mask = (uint16_t)(1u << (act_layer_prec - 1));

            auto wgt_layer_prec = layer.getWgtPrecision();
            auto wgt_mask = (uint16_t)(1u << (wgt_layer_prec - 1));

            std::map<uint64_t, uint16_t> wgt_memory_map;
            std::map<uint64_t, std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint8_t, bool>>> metadata;

            // Weights compressed
            uint64_t wgt_data_start = 0xA0000000;
            uint64_t wgt_group_start = 0xF0000000;

            uint64_t proteus_wgt_size = 0;
            uint64_t proteus_wgt_padding = 0;
            uint8_t proteus_wgt_data_pt = 0;

            uint64_t batch_wgt_group_size = 0;
            uint64_t batch_wgt_group_diff_size = 0;
            uint64_t batch_wgt_padding_size = 0;
            uint64_t batch_wgt_datawidth_size = 0;
            uint64_t wgt_data_offset = 0;
            uint64_t wgt_group_offset = 0;
            uint8_t wgt_data_pt = 0u;
            uint8_t wgt_group_pt = 0u;

            std::cout << layer.getName() << std::endl;

            auto wgt_filter_position = std::vector<uint64_t>(num_filters);

            for(int m = 0; m < num_filters; m++) {

                uint8_t prev_group = 0;

                // Generated statically
                wgt_filter_position[m] = wgt_data_offset + wgt_data_start;

                for (int y = 0; y < Ky; ++y) {

                    for (int x = 0; x < Kx; ++x) {

                        for (int k = 0; k < wgt_channels; k += PRECISION_GRANULARITY) {

                            uint8_t max_bit = 0;
                            for (int ss = k; ss < std::min((uint64_t) (k + PRECISION_GRANULARITY), wgt_channels); ++ss) {

                                uint16_t wgt_bits = wgt.get(m, ss, x, y);

                                if (signed_weights) {
                                    if ((wgt_bits & wgt_mask) != 0) {
                                        wgt_bits = wgt_bits & ~wgt_mask;
                                    }
                                }

                                const auto &min_max_wgt_bits = this->minMax(wgt_bits);
                                auto max_wgt_bit = std::get<1>(min_max_wgt_bits);
                                if (signed_weights) max_wgt_bit += 1;

                                if (max_wgt_bit > max_bit) max_bit = max_wgt_bit;
                            }

                            uint8_t width = max_bit + 1u;
                            auto width_mask = (uint16_t)(1u << (width - 1u));

                            if (width > network_bits) {
                            //    throw std::runtime_error("OnChip weights dynamic size bigger than network size");
                            }

                            // Store group
                            auto metadata_grp = std::make_tuple(m, k, x, y, width, 4, false);
                            uint16_t shifted_group = (width - 1u) << wgt_group_pt;
                            wgt_memory_map[wgt_group_start + wgt_group_offset] |= shifted_group;
                            metadata[wgt_group_start + wgt_group_offset].emplace_back(metadata_grp);
                            wgt_group_pt += 4;
                            if (wgt_group_pt == network_bits) {
                                wgt_group_pt = 0;
                                wgt_group_offset += 1;
                            }

                            // Store data
                            bool split = width + wgt_data_pt > (network_bits - 1);
                            for (int ss = 0; ss < PRECISION_GRANULARITY; ++ss) {

                                if ((ss + k) < wgt_channels) {
                                    uint16_t weight = wgt.get(m, ss + k, x, y);
                                    auto metadata_tuple = std::make_tuple(m, ss + k, x, y, weight, width, false);

                                    if ((weight & wgt_mask) != 0) {
                                        weight &= ~wgt_mask;
                                        weight |= width_mask;
                                    }

                                    uint16_t shifted_weight = weight << wgt_data_pt;
                                    wgt_memory_map[wgt_data_start + wgt_data_offset + ss] |= shifted_weight;
                                    metadata[wgt_data_start + wgt_data_offset + ss].emplace_back(metadata_tuple);

                                    if (split) {
                                        uint16_t rem_weight = weight >> (network_bits - wgt_data_pt);
                                        wgt_memory_map[wgt_data_start + wgt_data_offset + PRECISION_GRANULARITY + ss] = rem_weight;
                                        metadata[wgt_data_start + wgt_data_offset + PRECISION_GRANULARITY + ss].emplace_back(metadata_tuple);
                                    }
                                } else {
                                    auto metadata_tuple = std::make_tuple(m, ss + k, x, y, 0, width, false);
                                    wgt_memory_map[wgt_data_start + wgt_data_offset + ss] |= 0;
                                    metadata[wgt_data_start + wgt_data_offset + ss].emplace_back(metadata_tuple);
                                    if (split) metadata[wgt_data_start + wgt_data_offset + PRECISION_GRANULARITY + ss].emplace_back(metadata_tuple);
                                }
                            }

                            batch_wgt_group_size += log2(network_bits);
                            batch_wgt_group_diff_size += (prev_group == width ? 1 : 1 + log2(network_bits));
                            batch_wgt_datawidth_size += PRECISION_GRANULARITY * width;
                            wgt_data_pt = (wgt_data_pt + width) % network_bits;
                            if (split || wgt_data_pt == 0)
                                wgt_data_offset += PRECISION_GRANULARITY;

                            // Proteus
                            proteus_wgt_size += PRECISION_GRANULARITY * wgt_layer_prec;
                            proteus_wgt_data_pt = (proteus_wgt_data_pt + wgt_layer_prec) % network_bits;

                            prev_group = width;

                        }
                    }
                }

                if (wgt_data_pt != 0) {
                    batch_wgt_padding_size += (network_bits - wgt_data_pt) * PRECISION_GRANULARITY;
                    wgt_data_offset += PRECISION_GRANULARITY;
                    wgt_data_pt = 0;
                }

                // Proteus
                if (proteus_wgt_data_pt != 0) {
                    proteus_wgt_padding += (network_bits - proteus_wgt_data_pt) * PRECISION_GRANULARITY;
                    proteus_wgt_data_pt = 0;
                }

            }

            for(int n = 0; n < batch_size; n++) {

                std::map<uint64_t, uint16_t> memory_map = wgt_memory_map;

                // Activations compressed
                uint64_t act_data_start = 0x20000000;
                uint64_t act_group_start = 0x40000000;

                uint64_t proteus_act_size = 0;
                uint64_t proteus_act_padding = 0;
                uint8_t proteus_act_data_pt = 0;

                uint64_t batch_act_group_size = 0;
                uint64_t batch_act_group_diff_size = 0;
                uint64_t batch_act_padding_size = 0;
                uint64_t batch_act_datawidth_size = 0;
                uint64_t act_data_offset = 0;
                uint64_t act_group_offset = 0;
                uint8_t act_data_pt = 0u;
                uint8_t act_group_pt = 0u;

                auto act_positions = std::vector<std::vector<uint64_t>>(Ny, std::vector<uint64_t>(Nx, 0));

                for (int r = 0; r < R; ++r) {

                    for (int y = 0; y < Ny; ++y) {

                        for (int x = 0; x < Nx; ++x) {

                            uint8_t prev_group = 0;

                            // Generated from "previous" layer
                            act_positions[y][x] = act_data_offset + act_data_start;

                            for (int k = 0; k < act_channels; k += PRECISION_GRANULARITY) {

                                uint8_t max_bit = 0;
                                for (int ss = k; ss < std::min((uint64_t) (k + PRECISION_GRANULARITY), act_channels); ++ss) {

                                    uint16_t act_bits = lstm ? act.get(r, n, ss) : act.get(n, ss, x, y);

                                    if (signed_activations) {
                                        if ((act_bits & act_mask) != 0) {
                                            act_bits = act_bits & ~act_mask;
                                        }
                                    }

                                    const auto &min_max_act_bits = this->minMax(act_bits);
                                    auto max_act_bit = std::get<1>(min_max_act_bits);
                                    if (signed_activations) max_act_bit += 1;

                                    if (max_act_bit > max_bit) max_bit = max_act_bit;
                                }

                                uint8_t width = max_bit + 1u;
                                auto width_mask = (uint16_t) (1u << (width - 1u));

                                if (width > network_bits) {
                                    //throw std::runtime_error("OnChip activations dynamic size bigger than network size");
                                }

                                // Store group
                                auto metadata_grp = std::make_tuple(n, k, x, y, width, 4, true);
                                uint16_t shifted_group = (width - 1u) << act_group_pt;
                                memory_map[act_group_start + act_group_offset] |= shifted_group;
                                metadata[act_group_start + act_group_offset].emplace_back(metadata_grp);
                                act_group_pt += 4;
                                if (act_group_pt == network_bits) {
                                    act_group_pt = 0;
                                    act_group_offset += 1;
                                }

                                // Store data
                                bool split = width + act_data_pt > (network_bits - 1);
                                for (int ss = 0; ss < PRECISION_GRANULARITY; ++ss) {

                                    if ((ss + k) < act_channels) {
                                        uint16_t activation = lstm ? act.get(r, n, ss + k) : act.get(n, ss + k, x, y);
                                        auto metadata_tuple = std::make_tuple(n, ss + k, x, y, activation, width, true);

                                        if ((activation & act_mask) != 0) {
                                            activation &= ~act_mask;
                                            activation |= width_mask;
                                        }

                                        uint16_t shifted_activation = activation << act_data_pt;
                                        memory_map[act_data_start + act_data_offset + ss] |= shifted_activation;
                                        metadata[act_data_start + act_data_offset + ss].emplace_back(metadata_tuple);

                                        if (split) {
                                            uint16_t rem_activation = activation >> (network_bits - act_data_pt);
                                            memory_map[act_data_start + act_data_offset + PRECISION_GRANULARITY +
                                                       ss] = rem_activation;
                                            metadata[act_data_start + act_data_offset + PRECISION_GRANULARITY + ss].emplace_back(
                                                    metadata_tuple);
                                        }
                                    } else {
                                        auto metadata_tuple = std::make_tuple(n, ss + k, x, y, 0, width, true);
                                        memory_map[act_data_start + act_data_offset + ss] |= 0;
                                        metadata[act_data_start + act_data_offset + ss].emplace_back(metadata_tuple);
                                        if (split)
                                            metadata[act_data_start + act_data_offset + PRECISION_GRANULARITY + ss].emplace_back(
                                                    metadata_tuple);
                                    }
                                }

                                batch_act_group_size += log2(network_bits);
                                batch_act_group_diff_size += (prev_group == width ? 1 : 1 + log2(network_bits));
                                batch_act_datawidth_size += PRECISION_GRANULARITY * width;
                                act_data_pt = (act_data_pt + width) % network_bits;
                                if (split || act_data_pt == 0)
                                    act_data_offset += PRECISION_GRANULARITY;

                                // Proteus
                                proteus_act_size += PRECISION_GRANULARITY * act_layer_prec;
                                proteus_act_data_pt = (proteus_wgt_data_pt + act_layer_prec) % network_bits;

                                prev_group = width;

                            }

                            if (act_data_pt != 0) {
                                batch_act_padding_size += (network_bits - act_data_pt) * PRECISION_GRANULARITY;
                                act_data_offset += PRECISION_GRANULARITY;
                                act_data_pt = 0;
                            }

                            // Proteus
                            if (proteus_act_data_pt != 0) {
                                proteus_act_padding += (network_bits - proteus_act_data_pt) * PRECISION_GRANULARITY;
                                proteus_act_data_pt = 0;
                            }

                        }

                    }

                }

                if (this->CHECK && conv) {

                    auto output_activations = std::vector<std::vector<std::vector<uint32_t>>>(num_filters,
                            std::vector<std::vector<uint32_t>>(out_x,std::vector<uint32_t>(out_y,0)));

                    // Actual convolution
                    for (int m = 0; m < num_filters; m++) {

                        for (int x = 0; x < out_x; x++) {
                            for (int y = 0; y < out_y; y++) {
                                uint32_t sum = 0;
                                for (int i = 0; i < Kx; i++) {
                                    for (int j = 0; j < Ky; j++) {
                                        for (int k = 0; k < wgt_channels; k++) {
                                            sum += act.get(n, k, stride * x + i, stride * y + j) * wgt.get(m, k, i, j);
                                        }
                                    }
                                }
                                output_activations[m][x][y] = sum;
                            }
                        }
                    }

                    // Compressed Memory Convolution
                    auto compressed_output_activations = std::vector<std::vector<std::vector<uint32_t>>>(num_filters,
                            std::vector<std::vector<uint32_t>>(out_x,std::vector<uint32_t>(out_y,0)));
                    uint32_t batch_act_max_rel_pointer = 0;

                    if (stride > 1) {

                        uint32_t wgt_next_blk = 0;

                        // Activations starting positions
                        auto channel_groups = (uint64_t) ceil(act_channels / (double) PRECISION_GRANULARITY);

                        auto num_windows = out_x * out_y;
                        auto windows_per_column = (uint16_t) ceil(num_windows / (double) N_COLUMNS);

                        for (int w = 0; w < windows_per_column; ++w) {

                            for (int m = 0; m < num_filters; m++) {

                                for (int C = 0; C < N_COLUMNS; C++) { // Windows in parallel

                                    // Weights starting positions
                                    wgt_next_blk = 0;
                                    uint8_t wgt_blk_index = 0;
                                    uint64_t wgt_base_addr = wgt_filter_position[m]; //[m];

                                    // Window starting positions
                                    auto flatten_pos = C * windows_per_column + w;

                                    // Last window may finish earlier
                                    if (flatten_pos >= (out_x * out_y))
                                        continue;

                                    int y = flatten_pos / out_x;
                                    int x = flatten_pos % out_x;

                                    for (int ky = 0; ky < Ky; ky++) {

                                        auto act_next_blk = act_positions[stride * y + ky][stride * x];

                                        for (int kx = 0; kx < Kx; kx++) {

                                            uint8_t act_blk_index = 0;

                                            for (int ch = 0; ch < channel_groups; ++ch) {

                                                // Activations width
                                                auto act_group_index = (stride * y + ky) * Nx * channel_groups +
                                                        (stride * x + kx) * channel_groups + ch;
                                                uint32_t act_block_offset = act_group_index * 4 / network_bits + act_group_start;
                                                uint32_t act_mem_pointer = act_group_index * 4 % network_bits;
                                                int act_width = get_value(memory_map, act_block_offset, act_mem_pointer,
                                                        4, network_bits,PRECISION_GRANULARITY) + 1;
                                                auto act_width_mask = (uint16_t) (1u << (act_width - 1u));


                                                // Weights width
                                                auto wgt_group_index = m * Kx * Ky * channel_groups +
                                                        ky * Kx * channel_groups + kx * channel_groups + ch;
                                                uint32_t wgt_block_offset = wgt_group_index * 4 / network_bits + wgt_group_start;
                                                uint32_t wgt_mem_pointer = wgt_group_index * 4 % network_bits;
                                                int wgt_width = get_value(memory_map, wgt_block_offset, wgt_mem_pointer,
                                                        4, network_bits,PRECISION_GRANULARITY) + 1;
                                                auto wgt_width_mask = (uint16_t) (1u << (wgt_width - 1u));

                                                for (int ss = 0; ss < PRECISION_GRANULARITY; ++ss) {

                                                    // Activations values
                                                    act_block_offset = ss + act_next_blk;
                                                    uint16_t ch_act = get_value(memory_map, act_block_offset,
                                                            act_blk_index, act_width, network_bits,PRECISION_GRANULARITY);

                                                    if ((ch_act & act_width_mask) != 0) {
                                                        ch_act &= ~act_width_mask;
                                                        ch_act |= act_mask;
                                                    }

                                                    // Weights values
                                                    wgt_block_offset = ss + wgt_next_blk + wgt_base_addr;
                                                    uint16_t ch_wgt = get_value(memory_map, wgt_block_offset,
                                                            wgt_blk_index, wgt_width, network_bits,PRECISION_GRANULARITY);

                                                    if ((ch_wgt & wgt_width_mask) != 0) {
                                                        ch_wgt &= ~wgt_width_mask;
                                                        ch_wgt |= wgt_mask;
                                                    }

                                                    // Multiply - Accumulate
                                                    compressed_output_activations[m][x][y] += ch_act * ch_wgt;

                                                }

                                                if ((act_width + act_blk_index) > (network_bits - 1)) {
                                                    act_next_blk += PRECISION_GRANULARITY;
                                                }
                                                act_blk_index = (act_blk_index + act_width) % network_bits;

                                                if ((wgt_width + wgt_blk_index) > (network_bits - 1)) {
                                                    wgt_next_blk += PRECISION_GRANULARITY;
                                                }
                                                wgt_blk_index = (wgt_blk_index + wgt_width) % network_bits;

                                            }

                                            if (act_blk_index != 0) act_next_blk += PRECISION_GRANULARITY;

                                        } // Kernel X

                                    } // Kernel Y

                                } // Parallel windows

                            } // Filters

                        } // Required window sets for convolution

                    } else {

                        uint32_t wgt_next_blk = 0;
                        std::vector<uint32_t> act_next_blk = std::vector<uint32_t>(N_COLUMNS);

                        // Activations starting positions
                        auto channel_groups = (uint64_t) ceil(act_channels / (double) PRECISION_GRANULARITY);
                        std::vector<std::vector<uint32_t>> act_column_offsets = std::vector<std::vector<uint32_t>>(
                                N_COLUMNS, std::vector<uint32_t>(Ky, 0));
                        std::vector<std::vector<uint32_t>> act_registers = std::vector<std::vector<uint32_t>>(N_COLUMNS,
                                std::vector<uint32_t>(Ky,0));

                        auto num_windows = out_x * out_y;
                        auto windows_per_column = (uint16_t) ceil(num_windows / (double) N_COLUMNS);
                        std::vector<uint64_t> act_base_addr = std::vector<uint64_t>(N_COLUMNS, 0);
                        for (int C = 0; C < N_COLUMNS; ++C) {

                            auto flatten_pos = C * windows_per_column;

                            // Not all windows required
                            if (flatten_pos >= (out_x * out_y))
                                continue;

                            int row = flatten_pos / out_x;
                            int column = flatten_pos % out_x;
                            act_base_addr[C] = act_positions[row][0];

                            for (int i = 0; i < Ky; ++i) {
                                act_column_offsets[C][i] = act_positions[row + i][0] - act_base_addr[C];
                                act_registers[C][i] = act_positions[row + i][column] - act_base_addr[C];
                                if (act_registers[C][i] > batch_act_max_rel_pointer)
                                    batch_act_max_rel_pointer = act_registers[C][i];
                            }
                        }

                        for (int w = 0; w < windows_per_column; ++w) {

                            for (int m = 0; m < num_filters; m++) {

                                for (int C = 0; C < N_COLUMNS; C++) { // Windows in parallel

                                    // Weights starting positions
                                    wgt_next_blk = 0;
                                    uint8_t wgt_blk_index = 0;
                                    uint64_t wgt_base_addr = wgt_filter_position[m]; //[m];

                                    // Window starting positions
                                    auto flatten_pos = C * windows_per_column + w;

                                    // Last window may finish earlier
                                    if (flatten_pos >= (out_x * out_y))
                                        continue;

                                    int y = flatten_pos / out_x;
                                    int x = flatten_pos % out_x;

                                    for (int ky = 0; ky < Ky; ky++) {

                                        act_next_blk[C] = act_registers[C][ky];

                                        for (int kx = 0; kx < Kx; kx++) {

                                            uint8_t act_blk_index = 0;

                                            for (int ch = 0; ch < channel_groups; ++ch) {

                                                // Activations width
                                                auto act_group_index = (y + ky) * Nx * channel_groups +
                                                        (x + kx) * channel_groups + ch;
                                                uint32_t act_block_offset = act_group_index * 4 / network_bits + act_group_start;
                                                uint32_t act_mem_pointer = act_group_index * 4 % network_bits;
                                                int act_width = get_value(memory_map, act_block_offset, act_mem_pointer,
                                                        4, network_bits,PRECISION_GRANULARITY) + 1;
                                                auto act_width_mask = (uint16_t) (1u << (act_width - 1u));

                                                // Weights width
                                                auto wgt_group_index = m * Kx * Ky * channel_groups +
                                                        ky * Kx * channel_groups + kx * channel_groups + ch;
                                                uint32_t wgt_block_offset = wgt_group_index * 4 / network_bits + wgt_group_start;
                                                uint32_t wgt_mem_pointer = wgt_group_index * 4 % network_bits;
                                                int wgt_width = get_value(memory_map, wgt_block_offset, wgt_mem_pointer,
                                                        4, network_bits,PRECISION_GRANULARITY) + 1;
                                                auto wgt_width_mask = (uint16_t) (1u << (wgt_width - 1u));

                                                /*int act_index = act_mem_pointer / 4;
                                                auto act_metadata = metadata[act_block_offset];
                                                if (std::get<1>(act_metadata[act_index]) != (ch * PRECISION_GRANULARITY))
                                                    exit(-1);
                                                if (std::get<2>(act_metadata[act_index]) != (x + kx))
                                                    exit(-1);
                                                if (std::get<3>(act_metadata[act_index]) != (y + ky))
                                                    exit(-1);
                                                if (std::get<4>(act_metadata[act_index]) != act_width)
                                                    exit(-1);

                                                int wgt_index = wgt_mem_pointer / 4;
                                                auto wgt_metadata = metadata[wgt_block_offset];
                                                if (std::get<0>(wgt_metadata[wgt_index]) != m)
                                                    exit(-1);
                                                if (std::get<1>(wgt_metadata[wgt_index]) != (ch * PRECISION_GRANULARITY))
                                                    exit(-1);
                                                if (std::get<2>(wgt_metadata[wgt_index]) != kx)
                                                    exit(-1);
                                                if (std::get<3>(wgt_metadata[wgt_index]) != ky)
                                                    exit(-1);
                                                if (std::get<4>(wgt_metadata[wgt_index]) != wgt_width)
                                                    exit(-1);*/

                                                for (int ss = 0; ss < PRECISION_GRANULARITY; ++ss) {

                                                    // Activations values
                                                    act_block_offset = ss + act_next_blk[C] + act_base_addr[C];
                                                    uint16_t ch_act = get_value(memory_map, act_block_offset,
                                                            act_blk_index, act_width, network_bits,PRECISION_GRANULARITY);

                                                    if ((ch_act & act_width_mask) != 0) {
                                                        ch_act &= ~act_width_mask;
                                                        ch_act |= act_mask;
                                                    }

                                                    // Weights values
                                                    wgt_block_offset = ss + wgt_next_blk + wgt_base_addr;
                                                    uint16_t ch_wgt = get_value(memory_map, wgt_block_offset,
                                                            wgt_blk_index, wgt_width, network_bits,PRECISION_GRANULARITY);

                                                    if ((ch_wgt & wgt_width_mask) != 0) {
                                                        ch_wgt &= ~wgt_width_mask;
                                                        ch_wgt |= wgt_mask;
                                                    }

                                                    /*act_metadata = metadata[act_block_offset];
                                                    if (((ch * PRECISION_GRANULARITY) + ss) < act_channels &&
                                                        act.get(n, (ch * PRECISION_GRANULARITY) + ss, x + kx, y + ky) != ch_act)
                                                        exit(-1);


                                                    wgt_metadata = metadata[wgt_block_offset];
                                                    if (((ch * PRECISION_GRANULARITY) + ss) < wgt_channels &&
                                                        wgt.get(m, (ch * PRECISION_GRANULARITY) + ss, kx, ky) != ch_wgt)
                                                        exit(-1);*/

                                                    // Multiply - Accumulate
                                                    compressed_output_activations[m][x][y] += ch_act * ch_wgt;

                                                }

                                                if ((act_width + act_blk_index) > (network_bits - 1)) {
                                                    act_next_blk[C] += PRECISION_GRANULARITY;
                                                }
                                                act_blk_index = (act_blk_index + act_width) % network_bits;

                                                if ((wgt_width + wgt_blk_index) > (network_bits - 1)) {
                                                    wgt_next_blk += PRECISION_GRANULARITY;
                                                }
                                                wgt_blk_index = (wgt_blk_index + wgt_width) % network_bits;

                                            }

                                            if (act_blk_index != 0) act_next_blk[C] += PRECISION_GRANULARITY;

                                            // Update pointer to next window after last filter
                                            if (m == (num_filters - 1) && (kx == (stride - 1) || Kx == 1)) {
                                                act_registers[C][ky] = act_next_blk[C];
                                            }

                                        } // Kernel X

                                    } // Kernel Y

                                    // Update column offsets
                                    if (x == (out_x - 1) && m == (num_filters - 1)) { // Last X window for last filter
                                        if (Kx != 1) {
                                            auto prev_offset = act_column_offsets[C][stride];
                                            act_base_addr[C] = act_base_addr[C] + act_column_offsets[C][stride];
                                            for (int ky = 0; ky < (Ky - 1); ++ky) {
                                                act_column_offsets[C][ky] = act_column_offsets[C][ky + 1] - prev_offset;
                                            }
                                            act_column_offsets[C][Ky - 1] = act_next_blk[C] - prev_offset;
                                        } else {
                                            act_base_addr[C] += act_next_blk[C];
                                            act_column_offsets[C][0] = 0;
                                        }

                                        // Update registers
                                        for (int i = 0; i < Ky; i++) {
                                            act_registers[C][i] = act_column_offsets[C][i];
                                            if (act_registers[C][i] > batch_act_max_rel_pointer)
                                                batch_act_max_rel_pointer = act_registers[C][i];
                                        }
                                    }

                                } // Parallel windows

                            } // Filters

                        } // Required window sets for convolution

                    }

                    // Check values
                    for (int ch = 0; ch < num_filters; ++ch) {
                        for (int x = 0; x < out_x; ++x) {
                            for (int y = 0; y < out_y; ++y) {
                                auto actual_value = output_activations[ch][x][y];
                                auto compressed_value = compressed_output_activations[ch][x][y];
                                if (actual_value != compressed_value)
                                    throw std::runtime_error("On-Chip compressed wrong value.");
                            }
                        }
                    }

                    act_max_rel_pointer->value[layer_it][n] = batch_act_max_rel_pointer;

                }

                // Act Bits
                auto num_act = (uint64_t)(R * Nx * Ny * ceil(act_channels / (double) PRECISION_GRANULARITY) *
                        PRECISION_GRANULARITY);
                act_baseline_size->value[layer_it][n] = num_act * network_bits;
                act_profiled_size->value[layer_it][n] = proteus_act_size;
                act_profiled_padding->value[layer_it][n] = proteus_act_padding;

                act_datawidth_size->value[layer_it][n] = batch_act_datawidth_size;
                act_datawidth_groups->value[layer_it][n] = batch_act_group_size;
                act_datawidth_groups_diff->value[layer_it][n] = batch_act_group_diff_size;
                act_datawidth_padding->value[layer_it][n] = batch_act_padding_size;

                uint64_t row_overhead = 0;
                if (N_COLUMNS > out_y) {
                    auto full_rows = N_COLUMNS / out_y;
                    auto partial_rows = N_COLUMNS % out_y + (Ky - 1);
                    row_overhead = Ny * full_rows * 32 + partial_rows * 32;
                } else {
                    row_overhead = Ny * 32;
                }

                if (stride > 1 && Kx <= stride) {
                    act_datawidth_overhead->value[layer_it][n] = out_y * out_x * 32;
                    act_datawidth_row_overhead->value[layer_it][n] = out_y * out_x * 32;
                    act_datawidth_max_overhead->value[layer_it][n] = out_y * out_x * 32;
                } else {
                    act_datawidth_overhead->value[layer_it][n] = N_COLUMNS * ((16 * Ky) + (16 * Ky) + 32);
                    act_datawidth_row_overhead->value[layer_it][n] = N_COLUMNS * (16 * Ky) + row_overhead;
                    act_datawidth_max_overhead->value[layer_it][n] = Ny * out_x * 32;
                }

                // Wgt Bits
                auto num_wgt = (uint64_t) (num_filters * Kx * Ky * ceil(wgt_channels/(double)PRECISION_GRANULARITY) *
                        PRECISION_GRANULARITY);
                wgt_baseline_size->value[layer_it][n] = num_wgt * network_bits;

                wgt_profiled_size->value[layer_it][n] = proteus_wgt_size;
                wgt_profiled_padding->value[layer_it][n] = proteus_wgt_padding;
                wgt_datawidth_size->value[layer_it][n] = batch_wgt_datawidth_size;
                wgt_datawidth_groups->value[layer_it][n] = batch_wgt_group_size;
                wgt_datawidth_groups_diff->value[layer_it][n] = batch_wgt_group_diff_size;
                wgt_datawidth_padding->value[layer_it][n] = batch_wgt_padding_size;
                wgt_datawidth_overhead->value[layer_it][n] = num_filters * 32;
                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = wgt_layer_prec;

            }
        }

        //Dump statistics
        std::string header = "DynamicStripes On Chip Compression for " + network.getName() + "\n";
        header += "Number of values per group: " + std::to_string(PRECISION_GRANULARITY) + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    void read_window_sets(std::list<int> &window_list, std::vector<int> &windows_on_chip,
            std::vector<std::list<uint64_t>> &window_requests, const address_map &act_address_map,
            const std::vector<std::vector<uint64_t>> &act_datawidth_size,
            const std::vector<std::vector<uint64_t>> &act_baseline_size, uint64_t on_chip_size, uint64_t n,
            uint64_t Kx, uint64_t Ky, uint64_t Nx, uint64_t Ny, uint64_t act_channels, uint64_t out_x, uint64_t out_y,
            uint64_t stride, uint64_t N_COLUMNS, uint64_t values_block, uint64_t &max_act) {

        std::vector<std::vector<bool>> requested_address = std::vector<std::vector<bool>>(Ny,
                std::vector<bool>(Nx, false));

        uint64_t act_size = 0;
        while (true) {

            if (window_list.empty())
                break;

            // Select up to N_COLUMNS windows
            std::list<int> tmp_windows_on_chip;
            auto list_size = window_list.size();
            for (int i = 0; i < std::min(N_COLUMNS, list_size); ++i) {
                int window_read = window_list.front();
                tmp_windows_on_chip.push_back(window_read);
                window_list.pop_front();
            }

            // Calculate window set size
            uint64_t window_set_size = 0;
            uint64_t window_set_bas_size = 0;
            std::vector<uint64_t> tmp_window_requests;
            for (auto window_read : tmp_windows_on_chip) {

                auto x_window = window_read % out_x;
                auto y_window = window_read / out_y;
                for (int y = 0; y < Ky; ++y) {
                    for (int x = 0; x < Kx; ++x) {

                        auto x_pos = x_window * stride + x;
                        auto y_pos = y_window * stride + y;
                        if (requested_address[y_pos][x_pos])
                            continue;

                        for (int k = 0; k < act_channels; k += values_block) {
                            auto activations_address = act_address_map[n][y_pos][x_pos][k / values_block];
                            tmp_window_requests.push_back(activations_address);
                        }

                        window_set_size += act_datawidth_size[y_pos][x_pos];
                        window_set_bas_size += act_baseline_size[y_pos][x_pos];
                        requested_address[y_pos][x_pos] = true;
                    }
                }

            }

            if (window_set_size > max_act)
                max_act = window_set_size;

            // If window set doesn't fit, return to list and exit
            if ((act_size + window_set_size) > on_chip_size) {
                window_list.insert(window_list.begin(), tmp_windows_on_chip.begin(), tmp_windows_on_chip.end());
                break;
            }

            windows_on_chip.insert(windows_on_chip.end(), tmp_windows_on_chip.begin(), tmp_windows_on_chip.end());

            uint64_t rel_addresses = tmp_window_requests.size(); //ceil(window_set_size / (double)window_set_bas_size * tmp_window_requests.size());

            std::list<uint64_t> tmp_tmp_window_requests;
            for (int addr = 0; addr < std::min(rel_addresses, tmp_window_requests.size()); ++addr) {
                tmp_tmp_window_requests.push_back(tmp_window_requests[addr]);
            }

            window_requests.emplace_back(tmp_tmp_window_requests);
            act_size += window_set_size;

        }

        assert(act_size);

    }

    void read_filter_sets(std::list<int> &filter_list, std::vector<int> &filters_on_chip,
            std::vector<std::list<uint64_t>> &filter_requests, const address_map &wgt_address_map,
            const std::vector<uint64_t> &wgt_datawidth_size, const std::vector<uint64_t> &wgt_baseline_size,
            uint64_t on_chip_size, uint64_t Kx, uint64_t Ky, uint64_t wgt_channels, uint64_t N_ROWS,
            uint64_t values_block, uint64_t &max_wgt) {

        uint64_t wgt_size = 0;
        while (true) {

            if (filter_list.empty())
                break;

            // Select up to N_ROWS filters
            std::list<int> tmp_filters_on_chip;
            auto list_size = filter_list.size();
            for (int i = 0; i < std::min(N_ROWS, list_size); ++i) {
                int filter_read = filter_list.front();
                tmp_filters_on_chip.push_back(filter_read);
                filter_list.pop_front();
            }

            // Calculate filter set size
            uint64_t filter_set_size = 0;
            uint64_t filter_set_bas_size = 0;
            std::vector<uint64_t> tmp_filter_requests;
            for (auto filter_read : tmp_filters_on_chip) {

                filter_set_size += wgt_datawidth_size[filter_read];
                filter_set_bas_size += wgt_baseline_size[filter_read];

                for (int y = 0; y < Ky; ++y) {
                    for (int x = 0; x < Kx; ++x) {
                        for (int k = 0; k < wgt_channels; k += values_block) {
                            auto weights_address = wgt_address_map[filter_read][y][x][k / values_block];
                            tmp_filter_requests.push_back(weights_address);
                        }
                    }
                }

            }

            if (filter_set_size > max_wgt)
                max_wgt = filter_set_size;

            // If filter set doesn't fit, return to list and exit
            if ((wgt_size + filter_set_size) > on_chip_size) {
                filter_list.insert(filter_list.begin(), tmp_filters_on_chip.begin(), tmp_filters_on_chip.end());
                break;
            }

            filters_on_chip.insert(filters_on_chip.end(), tmp_filters_on_chip.begin(), tmp_filters_on_chip.end());

            uint64_t rel_addresses = tmp_filter_requests.size(); //ceil(filter_set_size / (double)filter_set_bas_size * tmp_filter_requests.size());

            std::list<uint64_t> tmp_tmp_filter_requests;
            for (int addr = 0; addr < std::min(rel_addresses, tmp_filter_requests.size()); ++addr) {
                tmp_tmp_filter_requests.push_back(tmp_filter_requests[addr]);
            }

            filter_requests.emplace_back(tmp_tmp_filter_requests);

            wgt_size += filter_set_size;

        }

        if (wgt_size == 0) {
            read_filter_sets(filter_list, filters_on_chip, filter_requests, wgt_address_map, wgt_datawidth_size,
                    wgt_baseline_size, on_chip_size, Kx, Ky, wgt_channels, N_ROWS/2, values_block, max_wgt);
        }

    }

    template <typename T>
    void DynamicStripes<T>::on_chip_cycles(const base::Network<T> &network) {

        this->memory.initialise();
        std::string filename = "DynamicStripes_L" + std::to_string(N_LANES) + "_C" + std::to_string(N_COLUMNS) + "_R" +
                std::to_string(N_ROWS) + "_AS" + std::to_string(this->memory.getOnChipActSize()) + "_WS" +
                std::to_string(this->memory.getOnChipWgtSize()) + (BASELINE ? "_baseline" : "") + "_on_chip_cycles";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto compute_cycles = stats.register_uint_t("compute_cycles", 0, sys::AverageTotal);
        auto memory_cycles = stats.register_uint_t("memory_cycles", 0, sys::AverageTotal);
        auto act_on_chip = stats.register_uint_t("act_on_chip", 0 , sys::AverageTotal);
        auto act_off_chip = stats.register_uint_t("act_off_chip", 0 , sys::AverageTotal);
        auto act_off_chip_bytes = stats.register_uint_t("act_off_chip bytes", 0 , sys::AverageTotal);
        auto wgt_on_chip = stats.register_uint_t("wgt_on_chip", 0 , sys::AverageTotal);
        auto wgt_off_chip = stats.register_uint_t("wgt_off_chip", 0 , sys::AverageTotal);
        auto wgt_off_chip_bytes = stats.register_uint_t("wgt_off_chip bytes", 0 , sys::AverageTotal);
        auto prev_out_on_chip = stats.register_uint_t("prev_out_on_chip", 0 , sys::AverageTotal);
        auto act_max_set_bytes = stats.register_uint_t("act_max_set_bytes", 0 , sys::Max);
        auto wgt_max_set_bytes = stats.register_uint_t("wgt_max_set_bytes", 0 , sys::Max);

        uint64_t act_next_addr = 0;
        uint64_t act_base_addr = 0x40000000;

        uint64_t wgt_next_addr = 0;
        uint64_t wgt_base_addr = 0x00000000;

        auto network_bits_act = network.getNetwork_bits();
        auto network_bits_wgt = network.getNetwork_bits();
        if (network.isIntelINQ()) {
            network_bits_act = 16;
            network_bits_wgt = 8;
        }
        auto signed_activations = !network.isUnsignedAct();
        auto signed_weights = !network.isUnsignedWgt();
        auto values_block_act = 64 / network_bits_act;
        auto values_block_wgt = 64 / network_bits_wgt;

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            if (layer_it != 0) signed_activations = false;

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            if (!this->QUIET) std::cout << layer.getName() << std::endl;

            base::Array<T> act = layer.getActivations();
            act.sign_magnitude_representation(layer.getActPrecision());
            if (fc && act.getDimensions() == 4) act.reshape_to_2D();
            if (act.getDimensions() == 2) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            wgt.sign_magnitude_representation(layer.getWgtPrecision());
            if (wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            if (act.getShape()[1] == 3 && stride > 1) {
                act.reshape_first_layer_act((uint16_t) stride);
                wgt.reshape_first_layer_wgt((uint16_t) stride);
                stride = 1;
            }

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
            if (this->FAST_MODE) batch_size = 1;

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            auto out_x = (Nx - Kx) / stride + 1;
            auto out_y = (Ny - Ky) / stride + 1;

            // Off-chip memory layout
            address_map act_address_map = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(batch_size,
                    std::vector<std::vector<std::vector<uint64_t>>>(Ny, std::vector<std::vector<uint64_t>>(Nx,
                    std::vector<uint64_t>(ceil(act_channels / (double)values_block_act)))));

            // Image fourth
            for (int n = 0; n < batch_size; ++n) {

                // Column third
                for (int y = 0; y < Ny; ++y) {

                    // Row second
                    for (int x = 0; x < Nx; ++x) {

                        // Store channel-first
                        for (int k = 0; k < act_channels; k += values_block_act) {
                            act_address_map[n][y][x][k/values_block_act] = act_base_addr + act_next_addr;
                            act_next_addr += 0x40; // Align to 64 bits
                        }
                    }
                }
            }

            address_map wgt_address_map = std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(num_filters,
                    std::vector<std::vector<std::vector<uint64_t>>>(Ky, std::vector<std::vector<uint64_t>>(Kx,
                    std::vector<uint64_t>(ceil(wgt_channels / (double)values_block_wgt)))));

            // Filter fourth
            for (int m = 0; m < num_filters; ++m) {

                // Column third
                for (int y = 0; y < Ky; ++y) {

                    // Row second
                    for (int x = 0; x < Kx; ++x) {

                        // Store channel-first
                        for (int k = 0; k < wgt_channels; k += values_block_wgt) {
                            wgt_address_map[m][y][x][k/values_block_wgt] = wgt_base_addr + wgt_next_addr;
                            wgt_next_addr += 0x40; // Align to 64 bits
                        }
                    }
                }
            }

            auto act_layer_prec = layer.getActPrecision();
            auto act_mask = (uint16_t)(1u << (act_layer_prec - 1));

            auto wgt_layer_prec = layer.getWgtPrecision();
            auto wgt_mask = (uint16_t)(1u << (wgt_layer_prec - 1));

            // Iterate filter sets dataflow - Output stationary
            for (int n = 0; n < batch_size; ++n) {

                // Calculate activations Dynamic Width sizes per channel
                std::vector<std::vector<uint64_t>> act_datawidth_size = std::vector<std::vector<uint64_t>>(Ny,
                        std::vector<uint64_t>(Nx, 0));
                std::vector<std::vector<uint64_t>> act_baseline_size = std::vector<std::vector<uint64_t>>(Ny,
                        std::vector<uint64_t>(Nx, 0));

                std::vector<std::vector<uint64_t>> act_on_chip_accesses = std::vector<std::vector<uint64_t>>(Ny,
                        std::vector<uint64_t>(Nx, 0));

                uint64_t batch_out_on_chip = 0;
                uint8_t act_data_pt = 0u;
                for (int r = 0; r < R; ++r) {

                    for (int y = 0; y < Ny; ++y) {

                        for (int x = 0; x < Nx; ++x) {

                            uint64_t channel_size = 0;
                            uint64_t base_channel_size = 0;
                            act_on_chip_accesses[y][x]++;
                            batch_out_on_chip++;

                            for (int k = 0; k < act_channels; k += PRECISION_GRANULARITY) {

                                if (act_data_pt >= network_bits_act) {
                                    act_data_pt %= network_bits_act;
                                    act_on_chip_accesses[y][x]++;
                                    batch_out_on_chip++;
                                }

                                uint8_t max_bit = 0;
                                for (int ss = k; ss < std::min((uint64_t) (k + PRECISION_GRANULARITY), act_channels); ++ss) {

                                    uint16_t act_bits = lstm ? act.get(r, n, ss) : act.get(n, ss, x, y);

                                    if (signed_activations) {
                                        if ((act_bits & act_mask) != 0) {
                                            act_bits = act_bits & ~act_mask;
                                        }
                                    }

                                    const auto &min_max_act_bits = this->minMax(act_bits);
                                    auto max_act_bit = std::get<1>(min_max_act_bits);
                                    if (signed_activations) max_act_bit += 1;

                                    if (max_act_bit > max_bit) max_bit = max_act_bit;
                                }

                                uint8_t width = max_bit + 1u;
                                if (BASELINE || act_channels < PRECISION_GRANULARITY) {
                                    channel_size += PRECISION_GRANULARITY * network_bits_act; // Baseline values
                                    act_data_pt += network_bits_act;

                                    base_channel_size += PRECISION_GRANULARITY * network_bits_act;
                                } else {
                                    //channel_size += log2(network_bits); // Group overhead
                                    channel_size += log2(network_bits_act); // Group overhead
                                    channel_size += PRECISION_GRANULARITY * width; // Values
                                    act_data_pt += width;

                                    base_channel_size += PRECISION_GRANULARITY * network_bits_act;
                                }

                            }

                            if (act_data_pt >= network_bits_act) {
                                act_data_pt %= network_bits_act;
                            }

                            // Padding overhead
                            if (act_data_pt != 0) {

                                if (act_data_pt >= network_bits_act) {
                                    act_data_pt %= network_bits_act;
                                }

                                channel_size += (network_bits_act - act_data_pt) * PRECISION_GRANULARITY;
                                act_data_pt = 0;
                            }

                            act_datawidth_size[y][x] = ceil(channel_size / 8.);
                            act_baseline_size[y][x] = ceil(base_channel_size / 8.);

                        }

                    }

                }

                if (layer_it == 0) batch_out_on_chip = 0;
                prev_out_on_chip->value[layer_it][n] = batch_out_on_chip;

                // Calculate activations Dynamic Width sizes per channel
                std::vector<uint64_t> wgt_baseline_size = std::vector<uint64_t>(num_filters, 0);
                std::vector<uint64_t> wgt_datawidth_size = std::vector<uint64_t>(num_filters, 0);

                std::vector<uint64_t> wgt_on_chip_accesses = std::vector<uint64_t>(num_filters, 0);

                uint8_t wgt_data_pt = 0u;
                for (int m = 0; m < num_filters; ++m) {

                    uint64_t filter_size = 0;
                    uint64_t base_filter_size = 0;
                    wgt_on_chip_accesses[m]++;

                    for (int y = 0; y < Ky; ++y) {

                        for (int x = 0; x < Kx; ++x) {

                            for (int k = 0; k < wgt_channels; k += PRECISION_GRANULARITY) {

                                uint8_t max_bit = 0;
                                for (int ss = k; ss < std::min((uint64_t) (k + PRECISION_GRANULARITY), wgt_channels); ++ss) {

                                    if (wgt_data_pt >= network_bits_wgt) {
                                        wgt_data_pt %= network_bits_wgt;
                                        wgt_on_chip_accesses[m]++;
                                    }

                                    uint16_t wgt_bits = wgt.get(m, ss, x, y);

                                    if (signed_weights) {
                                        if ((wgt_bits & wgt_mask) != 0) {
                                            wgt_bits = wgt_bits & ~wgt_mask;
                                        }
                                    }

                                    const auto &min_max_wgt_bits = this->minMax(wgt_bits);
                                    auto max_wgt_bit = std::get<1>(min_max_wgt_bits);
                                    if (signed_weights) max_wgt_bit += 1;

                                    if (max_wgt_bit > max_bit) max_bit = max_wgt_bit;
                                }

                                uint8_t width = max_bit + 1u;
                                if (BASELINE) {
                                    filter_size += PRECISION_GRANULARITY * network_bits_wgt; // Baseline values
                                    wgt_data_pt += network_bits_wgt;

                                    base_filter_size += PRECISION_GRANULARITY * network_bits_wgt;
                                } else {
                                    //filter_size += log2(network_bits); // Group overhead
                                    filter_size += log2(network_bits_wgt); // Group overhead
                                    filter_size += PRECISION_GRANULARITY * width; // Values
                                    wgt_data_pt += width;

                                    base_filter_size += PRECISION_GRANULARITY * network_bits_wgt;
                                }

                            }

                        }

                    }

                    if (wgt_data_pt >= network_bits_wgt) {
                        wgt_data_pt %= network_bits_wgt;
                    }

                    // Padding overhead
                    if (wgt_data_pt != 0) {

                        if (wgt_data_pt >= network_bits_wgt) {
                            wgt_data_pt %= network_bits_wgt;
                        }

                        filter_size += (network_bits_wgt - wgt_data_pt) * PRECISION_GRANULARITY;
                        wgt_data_pt = 0;
                    }

                    wgt_datawidth_size[m] = ceil(filter_size / 8.);
                    wgt_baseline_size[m] = ceil(base_filter_size / 8.);

                }

                uint64_t batch_cycles = 0;
                uint64_t batch_compute_cycles = 0;
                this->memory.resetClockCycle();
                this->memory.resetMemCycle();

                // List all windows
                std::list<int> window_list(out_x * out_y);
                std::iota(std::begin(window_list), std::end(window_list), 0);

                // Select windows that fit on-chip
                while (!window_list.empty()) {

                    this->memory.requests.clear();

                    std::vector<int> windows_on_chip;
                    std::vector<std::list<uint64_t>> window_requests;

                    read_window_sets(window_list, windows_on_chip, window_requests, act_address_map, act_datawidth_size,
                            act_baseline_size, this->memory.getOnChipActSize(), n, Kx, Ky, Nx, Ny, act_channels, out_x,
                            out_y, stride, N_COLUMNS, values_block_act, act_max_set_bytes->value[layer_it][n]);

                    // List all filters
                    bool first = true;
                    std::list<int> filter_list(num_filters);
                    std::iota(std::begin(filter_list), std::end(filter_list), 0);

                    uint64_t max_act_address = 0;

                    // Select filters that fit on-chip
                    while (!filter_list.empty()) {

                        std::vector<int> filters_on_chip;
                        std::vector<std::list<uint64_t>> filter_requests;

                        read_filter_sets(filter_list, filters_on_chip, filter_requests, wgt_address_map,
                                wgt_datawidth_size, wgt_baseline_size, this->memory.getOnChipWgtSize(), Kx, Ky,
                                wgt_channels, N_ROWS, values_block_wgt, wgt_max_set_bytes->value[layer_it][n]);

                        auto num_window_sets = ceil(windows_on_chip.size() / (double)N_COLUMNS);
                        auto num_filter_sets = ceil(filters_on_chip.size() / (double)N_ROWS);
                        auto max_sets = (int)std::max(num_window_sets, num_filter_sets);

                        uint64_t max_wgt_address = 0;
                        for (int i = 0; i < max_sets; ++i) {

                            // Request the memory addresses
                            if (first && i < window_requests.size()) {
                                for (auto address : window_requests[i]) {
                                    this->memory.request_address(address, false);
                                    act_off_chip_bytes->value[layer_it][n] += 8;
                                    act_off_chip->value[layer_it][n]++;

                                    if (address > max_act_address)
                                        max_act_address = address;
                                }
                            }

                            // Request the memory addresses
                            if (i < filter_requests.size()) {
                                for (auto address : filter_requests[i]) {
                                    this->memory.request_address(address, false);
                                    wgt_off_chip_bytes->value[layer_it][n] += 8;
                                    wgt_off_chip->value[layer_it][n]++;

                                    if (address > max_wgt_address)
                                        max_wgt_address = address;
                                }
                            }

                        }

                        first = false;

                        // Convolute windows and filters on the on-chip memory
                        for (int w = 0; w < windows_on_chip.size(); w += N_COLUMNS) {

                            for (int m = 0; m < filters_on_chip.size(); m += N_ROWS) {

                                // Count on-chip weight accesses
                                for (int filter = m; filter < std::min((uint64_t)m + N_ROWS,
                                        filters_on_chip.size()); ++filter) {
                                    wgt_on_chip->value[layer_it][n] += wgt_on_chip_accesses[filter];
                                }

                                // Convolute windows
                                for (int y = 0; y < Ky; ++y) {
                                    for (int x = 0; x < Kx; ++x) {
                                        for (int k = 0; k < wgt_channels; k += N_LANES) {


                                            for (int window = w; window < std::min((uint64_t)w + N_COLUMNS,
                                                    windows_on_chip.size()); ++window) {

                                                auto x_window = windows_on_chip[window] % out_x;
                                                auto y_window = windows_on_chip[window] / out_y;

                                                // Count on-chip activations accesses
                                                if (k == 0) {
                                                    act_on_chip->value[layer_it][n] += act_on_chip_accesses
                                                            [y_window * stride + y][x_window * stride + x];
                                                }

                                                for (int filter = m; filter < std::min((uint64_t)m + N_ROWS,
                                                        filters_on_chip.size()); ++filter) {
                                                    for (int channel = k; channel < std::min((uint64_t)k + N_LANES,
                                                            wgt_channels); ++channel) {

                                                        auto act_address = max_act_address;
                                                        auto wgt_address = max_wgt_address;

                                                        this->memory.wait_for(act_address);
                                                        this->memory.wait_for(wgt_address);

                                                    }
                                                }
                                            }

                                            if (this->memory.getClockCycle() > batch_cycles)
                                                batch_cycles = this->memory.getClockCycle();
                                            else this->memory.wait_until(batch_cycles);

                                            batch_compute_cycles += 1;
                                            batch_cycles += 1;

                                        }
                                    }
                                }


                            }
                        }
                    }


                }

                cycles->value[layer_it][n] = batch_cycles;
                compute_cycles->value[layer_it][n] = batch_compute_cycles;
                memory_cycles->value[layer_it][n] = this->memory.getMemCycle();

            }
        }

        //Dump statistics
        std::string header = "DynamicStripes On-Chip Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "On-chip activations size: " + std::to_string(this->memory.getOnChipActSize()) + "\n";
        header += "On-chip weights size: " + std::to_string(this->memory.getOnChipWgtSize()) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template class DynamicStripes<uint16_t>;

}
