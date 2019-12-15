
#include <core/ShapeShifter.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t ShapeShifter<T>::computeDynamicStripesBitsPE(uint8_t layer_prec, const int network_bits) {
        return layer_prec * (uint8_t)network_bits;
    }

    template <typename T>
    uint8_t ShapeShifter<T>::computeDynamicStripesColumn(int batch, int recursion, int act_x, int act_y, int kernel_x,
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
    void ShapeShifter<T>::computeDynamicStripesTile(int batch, const std::vector<int> &list_act_x,
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
    void ShapeShifter<T>::computeDynamicStripes2DTile(int batch, const std::vector<int> &list_act_x,
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
    void ShapeShifter<T>::run(const base::Network<T> &network) {

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
    void ShapeShifter<T>::potentials(const base::Network<T> &network) {

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

    template class ShapeShifter<uint16_t>;

}
