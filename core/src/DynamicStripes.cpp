
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
                    short prev_act_bits = (stride * list_act_y[window] - stride < 0) ? 0 :
                            padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                                stride * list_act_y[window] + kernel_y - stride);

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

            for (int filter = init_filter; filter < std::min(init_filter + (int)N_ROWS, max_filter); filter++) {

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
                std::to_string(N_ROWS) + "_PG" + std::to_string(PRECISION_GRANULARITY) + "_CR" +
                std::to_string(COLUMN_REGISTERS) + "_BP" + std::to_string(BITS_PE) + (LEADING_BIT ? "_LB" : "") +
                "_cycles";
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

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

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
            auto filters_per_tile = N_ROWS/layer_rows_per_wgt;

            auto groups = act_channels / wgt_channels;
            auto num_filters_sets = (uint32_t)ceil(num_filters/(double)filters_per_tile/groups);
            auto baseline_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS/groups);

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
                                    batch_scheduled_pe += list_x.size() * N_ROWS;
                                    batch_idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
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
                                    batch_scheduled_pe += list_x.size() * N_ROWS;
                                    batch_idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                                }
                            }
                        }
                        batch_accumulator_updates++;
                    }

                    batch_cycles = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
                    cycles->value[layer_it][n] = batch_cycles * num_filters_sets;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles * num_filters_sets;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads * num_filters_sets;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads * num_filters_sets;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates * num_filters_sets;
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
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads * num_filters_sets;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads * num_filters_sets;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates * num_filters_sets;
                    scheduled_pe->value[layer_it][n] = (uint64_t)(num_filters * N_ROWS * ceil(act_channels/(double)N_LANES));
                    auto batch_idle_rows = N_ROWS - (num_filters % N_ROWS);
                    batch_idle_rows = batch_idle_rows == 16 ? 0 : batch_idle_rows;
                    idle_pe->value[layer_it][n] = (uint64_t)(batch_idle_rows * ceil(act_channels/(double)N_LANES));
                    baseline_cycles->value[layer_it][n] = base_cycles;
                    speedup->value[layer_it][n] = base_cycles / (double)cycles->value[layer_it][n];

                }

                idle_rows->value[layer_it][n] = N_ROWS - filters_per_tile * layer_rows_per_wgt;
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

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

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
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_avg_width = stats.register_double_t("wgt_avg_width", 0, sys::Average);
        auto wgt_width_reduction = stats.register_double_t("wgt_width_reduction", 0, sys::Average);
        auto wgt_bits_baseline = stats.register_uint_t("wgt_bits_baseline", 0, sys::AverageTotal);
        auto wgt_bits_profiled = stats.register_uint_t("wgt_bits_profiled", 0, sys::AverageTotal);
        auto wgt_bits_datawidth = stats.register_uint_t("wgt_bits_datawidth", 0, sys::AverageTotal);
        auto wgt_bits_scnn = stats.register_uint_t("wgt_bits_scnn", 0, sys::AverageTotal);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto network_bits = network.getNetwork_bits();
        auto act_width_need = stats.register_double_dist_t("act_width_need",0,network_bits,0.0,sys::AverageTotal);
        auto wgt_width_need = stats.register_double_dist_t("wgt_width_need",0,network_bits,0.0,sys::AverageTotal);

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            act.sign_magnitude_representation(layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();
            if(fc) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
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

                std::vector<int> list_x, list_y;
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
                                batch_act_bits_datawidth = batch_act_bits_datawidth + (width * non_zeroes);
                            }
                        }
                    }
                }

                // Calculate data from off-chip
                auto num_act = R * Nx * Ny * act_channels;
                act_bits_baseline->value[layer_it][n] = num_act * network_bits;
                act_bits_profiled->value[layer_it][n] = 4 + num_act * act_layer_prec;
                auto overhead = (uint64_t)((16 + log2(network_bits)) * ceil(num_act / 16.));
                act_bits_datawidth->value[layer_it][n] = overhead + batch_act_bits_datawidth;

            }

            // Weights
            std::vector<double> wgt_width;
            for(int m = 0; m < num_filters; m += N_ROWS) {

                for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = 0; k < wgt_channels; k += N_LANES) {
                            auto tile_wgt_width = computeAvgWidthDynamicStripesWgtTile(i,j,k,m,wgt,(int)wgt_channels,
                                    (int)num_filters, wgt_mask);
                            wgt_width.insert(wgt_width.end(),tile_wgt_width.begin(),tile_wgt_width.end());

                        }
                    }

                }
            }

            uint64_t batch_wgt_bits_datawidth = 0;
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
                            batch_wgt_bits_datawidth = batch_wgt_bits_datawidth + (width * non_zeroes);
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

            for(int n=0; n<batch_size; n++) {

                // Calculate data from off-chip
                auto num_wgt = wgt.getMax_index();
                wgt_bits_baseline->value[layer_it][n] = num_wgt * network_bits;
                wgt_bits_profiled->value[layer_it][n] = 4 + num_wgt * wgt_layer_prec;
                auto overhead = (uint64_t)((16 + log2(network_bits)) * ceil(num_wgt / 16.));
                wgt_bits_datawidth->value[layer_it][n] = overhead + batch_wgt_bits_datawidth;

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

    /* ON CHIP */

    const uint64_t GROUP_SIZE = 16;

    uint16_t get_value(std::map<uint64_t, uint16_t> &memory_map, uint64_t block_offset, uint64_t mem_pointer,
            uint16_t width) {

        if ((width + mem_pointer) > 15) {

            uint16_t block = memory_map[block_offset];
            uint16_t next_block = memory_map[block_offset + GROUP_SIZE];

            uint16_t width_msb = (width + mem_pointer) % 16;
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
        std::string filename = "DynamicStripes_on_chip";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto act_baseline_size = stats.register_uint_t("act_baseline_size", 0, sys::AverageTotal);
        auto act_profiled_size = stats.register_uint_t("act_profiled_size", 0, sys::AverageTotal);
        auto act_datawidth_size = stats.register_uint_t("act_datawidth_size", 0, sys::AverageTotal);
        auto act_datawidth_groups = stats.register_uint_t("act_datawidth_groups", 0, sys::AverageTotal);
        auto act_datawidth_padding = stats.register_uint_t("act_datawidth_padding", 0, sys::AverageTotal);
        auto act_datawidth_overhead = stats.register_uint_t("act_datawidth_overhead", 0, sys::AverageTotal);
        auto act_datawidth_row_overhead = stats.register_uint_t("act_datawidth_row_overhead", 0, sys::AverageTotal);
        auto act_datawidth_max_overhead = stats.register_uint_t("act_datawidth_max_overhead", 0, sys::AverageTotal);
        auto act_max_rel_pointer = stats.register_uint_t("act_max_rel_pointer", 0, sys::Max);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);

        auto wgt_baseline_size = stats.register_uint_t("wgt_baseline_size", 0, sys::AverageTotal);
        auto wgt_profiled_size = stats.register_uint_t("wgt_profiled_size", 0, sys::AverageTotal);
        auto wgt_datawidth_size = stats.register_uint_t("wgt_datawidth_size", 0, sys::AverageTotal);
        auto wgt_datawidth_groups = stats.register_uint_t("wgt_datawidth_groups", 0, sys::AverageTotal);
        auto wgt_datawidth_padding = stats.register_uint_t("wgt_datawidth_padding", 0, sys::AverageTotal);
        auto wgt_datawidth_overhead = stats.register_uint_t("wgt_datawidth_overhead", 0, sys::AverageTotal);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto network_bits = network.getNetwork_bits();

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            if (!conv) continue;

            base::Array<T> act = layer.getActivations();
            act.sign_magnitude_representation(layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();
            if(fc) act.reshape_to_4D();

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

            std::map<uint64_t, uint16_t> wgt_memory_map;
            std::map<uint64_t, std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint16_t, uint16_t, uint8_t, bool>>> metadata;

            // Weights compressed
            uint64_t wgt_data_start = 0xA0000000;
            uint64_t wgt_group_start = 0xF0000000;
            uint8_t wgt_data_pt = 0u;
            uint8_t wgt_group_pt = 0u;

            uint64_t batch_wgt_group_size = 0;
            uint64_t batch_wgt_padding_size = 0;
            uint64_t batch_wgt_datawidth_size = 0;
            uint64_t wgt_data_offset = 0;
            uint64_t wgt_group_offset = 0;

            std::cout << layer.getName() << std::endl;

            auto wgt_filter_position = std::vector<uint64_t>(num_filters);

            for(int m = 0; m < num_filters; m++) {

                // Generated statically
                wgt_filter_position[m] = wgt_data_offset + wgt_data_start;

                for (int y = 0; y < Ky; ++y) {

                    for (int x = 0; x < Kx; ++x) {

                        for (int k = 0; k < wgt_channels; k += GROUP_SIZE) {

                            uint8_t max_bit = 0;
                            for (int ss = k; ss < std::min((uint64_t) (k + GROUP_SIZE), wgt_channels); ++ss) {

                                uint16_t wgt_bits = wgt.get(m, ss, x, y);

                                if ((wgt_bits & wgt_mask) != 0) {
                                    wgt_bits = wgt_bits & ~wgt_mask;
                                }

                                const auto &min_max_wgt_bits = this->minMax(wgt_bits);
                                auto max_wgt_bit = std::get<1>(min_max_wgt_bits);
                                max_wgt_bit += 1;

                                if (max_wgt_bit > max_bit) max_bit = max_wgt_bit;
                            }

                            uint8_t width = max_bit + 1u;
                            auto width_mask = (uint16_t)(1u << (width - 1u));

                            // Store group
                            auto metadata_grp = std::make_tuple(m, k, x, y, width, 4, false);
                            uint16_t shifted_group = (width - 1u) << wgt_group_pt;
                            wgt_memory_map[wgt_group_start + wgt_group_offset] |= shifted_group;
                            metadata[wgt_group_start + wgt_group_offset].emplace_back(metadata_grp);
                            wgt_group_pt += 4;
                            if (wgt_group_pt == 16) {
                                wgt_group_pt = 0;
                                wgt_group_offset += 1;
                            }

                            // Store data
                            bool split = width + wgt_data_pt > 15;
                            for (int ss = 0; ss < GROUP_SIZE; ++ss) {

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
                                        uint16_t rem_weight = weight >> (16u - wgt_data_pt);
                                        wgt_memory_map[wgt_data_start + wgt_data_offset + GROUP_SIZE + ss] = rem_weight;
                                        metadata[wgt_data_start + wgt_data_offset + GROUP_SIZE + ss].emplace_back(metadata_tuple);
                                    }
                                } else {
                                    auto metadata_tuple = std::make_tuple(m, ss + k, x, y, 0, width, false);
                                    wgt_memory_map[wgt_data_start + wgt_data_offset + ss] |= 0;
                                    metadata[wgt_data_start + wgt_data_offset + ss].emplace_back(metadata_tuple);
                                    if (split) metadata[wgt_data_start + wgt_data_offset + GROUP_SIZE + ss].emplace_back(metadata_tuple);
                                }
                            }

                            batch_wgt_group_size += 4;
                            batch_wgt_datawidth_size += GROUP_SIZE * width;
                            wgt_data_pt = (wgt_data_pt + width) % 16;
                            if (split || wgt_data_pt == 0)
                                wgt_data_offset += GROUP_SIZE;

                        }
                    }
                }

                if (wgt_data_pt != 0) {
                    batch_wgt_padding_size += (16 - wgt_data_pt) * GROUP_SIZE;
                    wgt_data_offset += GROUP_SIZE;
                    wgt_data_pt = 0;
                }
            }

            for(int n = 0; n < batch_size; n++) {

                std::map<uint64_t, uint16_t> memory_map = wgt_memory_map;

                // Activations compressed
                uint64_t act_data_start = 0x20000000;
                uint64_t act_group_start = 0x40000000;
                uint8_t act_data_pt = 0u;
                uint8_t act_group_pt = 0u;

                uint64_t batch_act_group_size = 0;
                uint64_t batch_act_padding_size = 0;
                uint64_t batch_act_datawidth_size = 0;
                uint64_t act_data_offset = 0;
                uint64_t act_group_offset = 0;

                auto act_positions = std::vector<std::vector<uint64_t>>(Ny, std::vector<uint64_t>(Nx, 0));

                for (int y = 0; y < Ny; ++y) {

                    for (int x = 0; x < Nx; ++x) {

                        // Generated from "previous" layer
                        act_positions[y][x] = act_data_offset + act_data_start;

                        for (int k = 0; k < act_channels; k += GROUP_SIZE) {

                            uint8_t max_bit = 0;
                            for (int ss = k; ss < std::min((uint64_t) (k + GROUP_SIZE), act_channels); ++ss) {

                                uint16_t act_bits = act.get(n, ss, x, y);

                                if ((act_bits & act_mask) != 0) {
                                    act_bits = act_bits & ~act_mask;
                                }

                                const auto &min_max_act_bits = this->minMax(act_bits);
                                auto max_act_bit = std::get<1>(min_max_act_bits);
                                max_act_bit += 1;

                                if (max_act_bit > max_bit) max_bit = max_act_bit;
                            }

                            uint8_t width = max_bit + 1u;
                            auto width_mask = (uint16_t) (1u << (width - 1u));

                            // Store group
                            auto metadata_grp = std::make_tuple(n, k, x, y, width, 4, true);
                            uint16_t shifted_group = (width - 1u) << act_group_pt;
                            memory_map[act_group_start + act_group_offset] |= shifted_group;
                            metadata[act_group_start + act_group_offset].emplace_back(metadata_grp);
                            act_group_pt += 4;
                            if (act_group_pt == 16) {
                                act_group_pt = 0;
                                act_group_offset += 1;
                            }

                            // Store data
                            bool split = width + act_data_pt > 15;
                            for (int ss = 0; ss < GROUP_SIZE; ++ss) {

                                if ((ss + k) < act_channels) {
                                    uint16_t activation = act.get(n, ss + k, x, y);
                                    auto metadata_tuple = std::make_tuple(n, ss + k, x, y, activation, width, true);

                                    if ((activation & act_mask) != 0) {
                                        activation &= ~act_mask;
                                        activation |= width_mask;
                                    }

                                    uint16_t shifted_activation = activation << act_data_pt;
                                    memory_map[act_data_start + act_data_offset + ss] |= shifted_activation;
                                    metadata[act_data_start + act_data_offset + ss].emplace_back(metadata_tuple);

                                    if (split) {
                                        uint16_t rem_activation = activation >> (16u - act_data_pt);
                                        memory_map[act_data_start + act_data_offset + GROUP_SIZE + ss] = rem_activation;
                                        metadata[act_data_start + act_data_offset + GROUP_SIZE + ss].emplace_back(
                                                metadata_tuple);
                                    }
                                } else {
                                    auto metadata_tuple = std::make_tuple(n, ss + k, x, y, 0, width, true);
                                    memory_map[act_data_start + act_data_offset + ss] |= 0;
                                    metadata[act_data_start + act_data_offset + ss].emplace_back(metadata_tuple);
                                    if (split)
                                        metadata[act_data_start + act_data_offset + GROUP_SIZE + ss].emplace_back(
                                                metadata_tuple);
                                }
                            }

                            batch_act_group_size += 4;
                            batch_act_datawidth_size += GROUP_SIZE * width;
                            act_data_pt = (act_data_pt + width) % 16;
                            if (split || act_data_pt == 0)
                                act_data_offset += GROUP_SIZE;

                        }

                        if (act_data_pt != 0) {
                            batch_act_padding_size += (16 - act_data_pt) * GROUP_SIZE;
                            act_data_offset += GROUP_SIZE;
                            act_data_pt = 0;
                        }

                    }

                }

                if (!this->FAST_MODE) {

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
                        auto channel_groups = (uint64_t) ceil(act_channels / (double) GROUP_SIZE);

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
                                                uint32_t act_block_offset = act_group_index * 4 / 16 + act_group_start;
                                                uint32_t act_mem_pointer = act_group_index * 4 % 16;
                                                int act_width = get_value(memory_map, act_block_offset, act_mem_pointer, 4) + 1;
                                                auto act_width_mask = (uint16_t) (1u << (act_width - 1u));


                                                // Weights width
                                                auto wgt_group_index = m * Kx * Ky * channel_groups +
                                                        ky * Kx * channel_groups + kx * channel_groups + ch;
                                                uint32_t wgt_block_offset = wgt_group_index * 4 / 16 + wgt_group_start;
                                                uint32_t wgt_mem_pointer = wgt_group_index * 4 % 16;
                                                int wgt_width = get_value(memory_map, wgt_block_offset, wgt_mem_pointer, 4) + 1;
                                                auto wgt_width_mask = (uint16_t) (1u << (wgt_width - 1u));

                                                for (int ss = 0; ss < GROUP_SIZE; ++ss) {

                                                    // Activations values
                                                    act_block_offset = ss + act_next_blk;
                                                    uint16_t ch_act = get_value(memory_map, act_block_offset,
                                                            act_blk_index, act_width);

                                                    if ((ch_act & act_width_mask) != 0) {
                                                        ch_act &= ~act_width_mask;
                                                        ch_act |= act_mask;
                                                    }

                                                    // Weights values
                                                    wgt_block_offset = ss + wgt_next_blk + wgt_base_addr;
                                                    uint16_t ch_wgt = get_value(memory_map, wgt_block_offset,
                                                                                wgt_blk_index, wgt_width);

                                                    if ((ch_wgt & wgt_width_mask) != 0) {
                                                        ch_wgt &= ~wgt_width_mask;
                                                        ch_wgt |= wgt_mask;
                                                    }

                                                    // Multiply - Accumulate
                                                    compressed_output_activations[m][x][y] += ch_act * ch_wgt;

                                                }

                                                if ((act_width + act_blk_index) > 15) {
                                                    act_next_blk += GROUP_SIZE;
                                                }
                                                act_blk_index = (act_blk_index + act_width) % 16;

                                                if ((wgt_width + wgt_blk_index) > 15) {
                                                    wgt_next_blk += GROUP_SIZE;
                                                }
                                                wgt_blk_index = (wgt_blk_index + wgt_width) % 16;

                                            }

                                            if (act_blk_index != 0) act_next_blk += GROUP_SIZE;

                                        } // Kernel X

                                    } // Kernel Y

                                } // Parallel windows

                            } // Filters

                        } // Required window sets for convolution

                    } else {

                        uint32_t wgt_next_blk = 0;
                        std::vector<uint32_t> act_next_blk = std::vector<uint32_t>(N_COLUMNS);

                        // Activations starting positions
                        auto channel_groups = (uint64_t) ceil(act_channels / (double) GROUP_SIZE);
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
                                                uint32_t act_block_offset = act_group_index * 4 / 16 + act_group_start;
                                                uint32_t act_mem_pointer = act_group_index * 4 % 16;
                                                int act_width = get_value(memory_map, act_block_offset, act_mem_pointer, 4) + 1;
                                                auto act_width_mask = (uint16_t) (1u << (act_width - 1u));

                                                // Weights width
                                                auto wgt_group_index = m * Kx * Ky * channel_groups +
                                                        ky * Kx * channel_groups + kx * channel_groups + ch;
                                                uint32_t wgt_block_offset = wgt_group_index * 4 / 16 + wgt_group_start;
                                                uint32_t wgt_mem_pointer = wgt_group_index * 4 % 16;
                                                int wgt_width =
                                                        get_value(memory_map, wgt_block_offset, wgt_mem_pointer, 4) + 1;
                                                auto wgt_width_mask = (uint16_t) (1u << (wgt_width - 1u));

                                                /*int act_index = act_mem_pointer / 4;
                                                auto act_metadata = metadata[act_block_offset];
                                                if (std::get<1>(act_metadata[act_index]) != (ch * GROUP_SIZE))
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
                                                if (std::get<1>(wgt_metadata[wgt_index]) != (ch * GROUP_SIZE))
                                                    exit(-1);
                                                if (std::get<2>(wgt_metadata[wgt_index]) != kx)
                                                    exit(-1);
                                                if (std::get<3>(wgt_metadata[wgt_index]) != ky)
                                                    exit(-1);
                                                if (std::get<4>(wgt_metadata[wgt_index]) != wgt_width)
                                                    exit(-1);*/

                                                for (int ss = 0; ss < GROUP_SIZE; ++ss) {

                                                    // Activations values
                                                    act_block_offset = ss + act_next_blk[C] + act_base_addr[C];
                                                    uint16_t ch_act = get_value(memory_map, act_block_offset,
                                                                                act_blk_index, act_width);

                                                    if ((ch_act & act_width_mask) != 0) {
                                                        ch_act &= ~act_width_mask;
                                                        ch_act |= act_mask;
                                                    }

                                                    // Weights values
                                                    wgt_block_offset = ss + wgt_next_blk + wgt_base_addr;
                                                    uint16_t ch_wgt = get_value(memory_map, wgt_block_offset,
                                                                                wgt_blk_index, wgt_width);

                                                    if ((ch_wgt & wgt_width_mask) != 0) {
                                                        ch_wgt &= ~wgt_width_mask;
                                                        ch_wgt |= wgt_mask;
                                                    }

                                                    /*act_metadata = metadata[act_block_offset];
                                                    if (((ch * GROUP_SIZE) + ss) < act_channels &&
                                                        act.get(n, (ch * GROUP_SIZE) + ss, x + kx, y + ky) != ch_act)
                                                        exit(-1);


                                                    wgt_metadata = metadata[wgt_block_offset];
                                                    if (((ch * GROUP_SIZE) + ss) < wgt_channels &&
                                                        wgt.get(m, (ch * GROUP_SIZE) + ss, kx, ky) != ch_wgt)
                                                        exit(-1);*/

                                                    // Multiply - Accumulate
                                                    compressed_output_activations[m][x][y] += ch_act * ch_wgt;

                                                }

                                                if ((act_width + act_blk_index) > 15) {
                                                    act_next_blk[C] += GROUP_SIZE;
                                                }
                                                act_blk_index = (act_blk_index + act_width) % 16;

                                                if ((wgt_width + wgt_blk_index) > 15) {
                                                    wgt_next_blk += GROUP_SIZE;
                                                }
                                                wgt_blk_index = (wgt_blk_index + wgt_width) % 16;

                                            }

                                            if (act_blk_index != 0) act_next_blk[C] += GROUP_SIZE;

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
                auto num_act = (uint64_t) (Nx * Ny * ceil(act_channels / (double) GROUP_SIZE) * GROUP_SIZE);
                act_baseline_size->value[layer_it][n] = num_act * network_bits;

                auto proteus_size = ceil(act_channels / (double) GROUP_SIZE) * act_layer_prec;
                act_profiled_size->value[layer_it][n] = Nx * Ny * (uint64_t) ceil(proteus_size / 16.) * 16 * GROUP_SIZE;

                act_datawidth_size->value[layer_it][n] = batch_act_datawidth_size;
                act_datawidth_groups->value[layer_it][n] = batch_act_group_size;
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
                } else if (stride > 1) {
                    act_datawidth_overhead->value[layer_it][n] = N_COLUMNS * (16 * Ky) + row_overhead;
                    act_datawidth_row_overhead->value[layer_it][n] = N_COLUMNS * (16 * Ky) + row_overhead;
                    act_datawidth_max_overhead->value[layer_it][n] = Ny * out_x * 32;
                } else {
                    act_datawidth_overhead->value[layer_it][n] = N_COLUMNS * ((16 * Ky) + (16 * Ky) + 32);
                    act_datawidth_row_overhead->value[layer_it][n] = N_COLUMNS * (16 * Ky) + row_overhead;
                    act_datawidth_max_overhead->value[layer_it][n] = Ny * out_x * 32;
                }

                // Wgt Bits
                auto num_wgt = (uint64_t) (num_filters * Kx * Ky * ceil(wgt_channels/(double)GROUP_SIZE) * GROUP_SIZE);
                wgt_baseline_size->value[layer_it][n] = num_wgt * network_bits;

                auto filter_size = (uint64_t) (Kx * Ky * ceil(wgt_channels / (double) GROUP_SIZE) * GROUP_SIZE);
                wgt_profiled_size->value[layer_it][n] = num_filters *
                        (uint64_t)ceil(filter_size * wgt_layer_prec / 16.) * 16;
                wgt_datawidth_size->value[layer_it][n] = batch_wgt_datawidth_size;
                wgt_datawidth_groups->value[layer_it][n] = batch_wgt_group_size;
                wgt_datawidth_padding->value[layer_it][n] = batch_wgt_padding_size;
                wgt_datawidth_overhead->value[layer_it][n] = num_filters * 32;
                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = wgt_layer_prec;

            }
        }

        //Dump statistics
        std::string header = "DynamicStripes On Chip Compression for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template class DynamicStripes<uint16_t>;

}
