
#include <core/BitTacticalP.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t BitTacticalP<T>::computeTacticalPBitsPE(uint16_t wgt, uint8_t act_layer_prec, const int network_bits) {
        #ifdef ZERO_COUNT
        return wgt == 0 ? (uint8_t)1 : act_layer_prec * (uint8_t)network_bits;
        #else
        return wgt == 0 ? (uint8_t)0 : act_layer_prec * (uint8_t)NETWORK_BITS;
        #endif
    }

    template <typename T>
    uint8_t BitTacticalP<T>::computeTacticalPColumn(int batch, int recursion, int act_x, int act_y, int stride,
            const base::Array<T> &padded_act, const schedule &dense_schedule, int schedule_time, uint16_t act_mask,
            bool lstm) {

        uint8_t max_bit = 0, min_bit = 16;
        for (int row = 0; row < this->N_ROWS; row++) {
            for (int wgt_idx = 0; wgt_idx < this->N_LANES; wgt_idx++) {

                int pos = row * this->N_LANES + wgt_idx;
                auto wgt_tuple = dense_schedule[schedule_time][pos];
                int channel = std::get<0>(wgt_tuple);
                int kernel_x = std::get<1>(wgt_tuple);
                int kernel_y = std::get<2>(wgt_tuple);
                if(channel < 0) continue;

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
        }

        int cycles;
        if(!LEADING_BIT) cycles = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
        else cycles = max_bit + 1;
        return (uint8_t)cycles;
    }

    template <typename T>
    void BitTacticalP<T>::computeTacticalPTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int stride, const base::Array<T> &padded_act,
            const schedule &dense_schedule, int schedule_time, uint16_t act_mask,
            std::vector<uint32_t> &cycles_per_group, std::vector<uint32_t> &end_previous_pallet,
            uint64_t &stall_cycles) {

        uint32_t N_GROUPS = this->N_COLUMNS * 16 / PRECISION_GRANULARITY;
        uint32_t WINDOWS_PER_GROUP = this->N_COLUMNS / N_GROUPS;

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

            for (int row = 0; row < this->N_ROWS; row++) {
                for (int wgt_idx = 0; wgt_idx < this->N_LANES; wgt_idx++) {

                    int pos = row * this->N_LANES + wgt_idx;
                    auto wgt_tuple = dense_schedule[schedule_time][pos];
                    int channel = std::get<0>(wgt_tuple);
                    int kernel_x = std::get<1>(wgt_tuple);
                    int kernel_y = std::get<2>(wgt_tuple);
                    if(channel < 0) continue;

                    // Computation cycles
                    uint16_t act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
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

        if(this->COLUMN_REGISTERS > 0) {
            auto fastest_column = end_previous_pallet[0] + 1;
            for(auto &column_cycles : cycles_per_group) {
                if(column_cycles <= end_previous_pallet[0]) {
                    if(column_cycles < fastest_column) fastest_column = column_cycles;
                    column_cycles = end_previous_pallet[0] + 1;
                }
            }
            stall_cycles += (end_previous_pallet[0] + 1) - fastest_column;

            //Update end_previous_pallet
            for(int i = 0; i < this->COLUMN_REGISTERS - 1; i++) {
                end_previous_pallet[i] = end_previous_pallet[i + 1];
            }
            end_previous_pallet[this->COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_group.begin(),
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
    void BitTacticalP<T>::run(const base::Network<T> &network, const std::vector<schedule> &schedules) {

        // Initialize statistics
        int mux_entries = this->LOOKAHEAD_H + this->LOOKASIDE_D + 1;
        std::string filename = "BitTacticalP_C" + std::to_string(this->N_COLUMNS) + "_R" + std::to_string(this->N_ROWS)
                + "_PG_" + std::to_string(PRECISION_GRANULARITY) + "_CR" + std::to_string(this->COLUMN_REGISTERS) + "_"
                + this->SEARCH_SHAPE + std::to_string(mux_entries) + "(" + std::to_string(this->LOOKAHEAD_H) + "-" +
                std::to_string(this->LOOKASIDE_D) + ")" + (LEADING_BIT ? "_LB" : "");
        sys::Stats stats = sys::Stats(network.getNumLayers(), network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto baseline_cycles = stats.register_uint_t("baseline_cycles", 0, sys::AverageTotal);
        auto stall_cycles = stats.register_uint_t("stall_cycles", 0, sys::AverageTotal);
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

            base::Array<T> act = layer.getActivations();
            if(!conv && act.getDimensions() == 4) act.reshape_to_2D();
            act.sign_magnitude_representation(layer.getActPrecision());
            base::Array<T> wgt = layer.getWeights();

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

            auto groups = act_channels / wgt_channels;
            auto num_filters_sets = (uint32_t)ceil(num_filters/(double)this->N_ROWS/groups);

            auto act_layer_prec = layer.getActPrecision();
            auto act_mask = (uint16_t)(1u << (act_layer_prec - 1));

            schedule dense_schedule;
            const schedule &proto_dense_schedule = schedules[layer_it];
            if(proto_dense_schedule.empty())
                dense_schedule = this->scheduler(wgt,act_channels);
            else
                dense_schedule = proto_dense_schedule;

            for(int n = 0; n < batch_size; n++) {

                uint64_t batch_cycles = 0;
                uint64_t batch_stall_cycles = 0;
                uint64_t batch_weight_buff_reads = 0;
                uint64_t batch_act_buff_reads = 0;
                uint64_t batch_accumulator_updates = 0;
                uint64_t batch_scheduled_pe = 0;
                uint64_t batch_idle_pe = 0;

                if (conv) {

                    std::vector<int> list_x, list_y;
                    int x_counter = 0, y_counter = 0;
                    std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(this->COLUMN_REGISTERS, 0);
                    std::vector<uint32_t> cycles_per_group = std::vector<uint32_t>(this->N_COLUMNS * 16 /
                            PRECISION_GRANULARITY, 0);

                    while (this->iterateWindows(out_x, out_y, list_x, list_y, x_counter, y_counter, this->N_COLUMNS)) {
                        for(int schedule_time = 0; schedule_time < dense_schedule.size(); schedule_time++) {
                            computeTacticalPTile(n, list_x, list_y, stride, act, dense_schedule, schedule_time,
                                    act_mask, cycles_per_group, end_previous_pallet, batch_stall_cycles);

                            batch_act_buff_reads++;
                            batch_weight_buff_reads++;
                            batch_scheduled_pe += list_x.size() * this->N_ROWS;
                            batch_idle_pe += (this->N_COLUMNS - list_x.size()) * this->N_ROWS;
                        }
                        batch_accumulator_updates++;
                    }

                    cycles->value[layer_it][n] = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
                    stall_cycles->value[layer_it][n] = batch_stall_cycles;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates;
                    scheduled_pe->value[layer_it][n] = batch_scheduled_pe;
                    idle_pe->value[layer_it][n] = batch_idle_pe;

                } else {

                    int column_index = 0;
                    std::vector<uint64_t> column_end = std::vector<uint64_t>(this->N_COLUMNS, 0);

                    for (int r = 0; r < R; r++) {
                        for(int schedule_time = 0; schedule_time < dense_schedule.size(); schedule_time++) {
                            if(batch_cycles < column_end[column_index]) {
                                batch_stall_cycles = column_end[column_index] - batch_cycles;
                                batch_cycles = column_end[column_index];
                            }
                            auto column_cycles = computeTacticalPColumn(n,r,0,0,0,act,dense_schedule,schedule_time,
                                    act_mask,lstm);
                            column_end[column_index] = batch_cycles + column_cycles;
                            batch_cycles++;
                            column_index++;
                            if(column_index >= this->N_COLUMNS) column_index = 0;

                            batch_act_buff_reads++;
                            batch_weight_buff_reads++;
                        }
                        batch_accumulator_updates++;
                    }

                    uint64_t last_column_end = *std::max_element(column_end.begin(), column_end.end());
                    cycles->value[layer_it][n] = std::max(batch_cycles, last_column_end);
                    stall_cycles->value[layer_it][n] = batch_stall_cycles;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates * num_filters_sets;
                    scheduled_pe->value[layer_it][n] = (uint64_t)(num_filters * this->N_ROWS *
                                                                  ceil(act_channels/(double)this->N_LANES));
                    auto idle_rows = this->N_ROWS - (num_filters % this->N_ROWS);
                    idle_rows = idle_rows == 16 ? 0 : idle_rows;
                    idle_pe->value[layer_it][n] = (uint64_t)(idle_rows * ceil(act_channels/(double)this->N_LANES));

                }

                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = layer.getWgtPrecision();

            }

        }

        //Dump statistics
        std::string header = "BitTactical-P Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(this->N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(this->N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(this->N_ROWS) + "\n";
        header += "Number of values per group: " + std::to_string(PRECISION_GRANULARITY) + "\n";
        header += "Calculate only leading bit: " + std::to_string(LEADING_BIT) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(this->COLUMN_REGISTERS) + "\n";
        header += "Search shape: " + std::to_string(this->SEARCH_SHAPE) + "\n";
        header += "Lookahead H: " + std::to_string(this->LOOKAHEAD_H) + "\n";
        header += "Lookaside D: " + std::to_string(this->LOOKASIDE_D) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    /* POTENTIALS */

    template <typename T>
    void BitTacticalP<T>::potentials(const base::Network<T> &network) {


        // Initialize statistics
        std::string filename = "BitTacticalP_potentials";
        sys::Stats stats = sys::Stats(network.getNumLayers(), network.getBatches(), filename);

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

            base::Array<T> act = layer.getActivations();
            if (!conv && act.getDimensions() == 4) act.reshape_to_2D();
            act.powers_of_two_representation(layer.getActPrecision());
            const base::Array<T> &wgt = layer.getWeights();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            uint64_t batch_size, Nx, Ny, R;
            if (lstm) {
                R = act_shape[0];
                batch_size = act_shape[1];
                Nx = 1;
                Ny = 1;
            } else {
                R = 1;
                batch_size = act_shape[0];
                Nx = act_shape[2];
                Ny = act_shape[3];
            }

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            // Get layer precision
            auto act_layer_prec = layer.getActPrecision();

            auto network_bits = network.getNetwork_bits();

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                if (conv) {

                    for(int m=0; m < num_filters; m++) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = 0; k < wgt_channels; k++) {
                                    bit_counter += computeTacticalPBitsPE(wgt.get(m, k, i, j), (uint8_t)act_layer_prec,
                                            network_bits);
                                }
                            }
                        }
                    }
                    bit_counter *= out_x * out_y;

                } else {

                    for (int r = 0; r < R; r++) {
                        for (int m = 0; m < num_filters; m++) {
                            for (int k = 0; k < wgt_channels; k++) {
                                bit_counter += computeTacticalPBitsPE(wgt.get(m, k), (uint8_t) act_layer_prec,
                                        network_bits);
                            }
                        }
                    }

                }

                bit_multiplications->value[layer_it][n] = bit_counter;
                work_reduction->value[layer_it][n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
                speedup->value[layer_it][n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
                par_mult->value[layer_it][n] = parallel_mult;
                act_prec->value[layer_it][n] = layer.getActPrecision();
                wgt_prec->value[layer_it][n] = layer.getWgtPrecision();
            }

        }

        //Dump statistics
        std::string header = "BitTactical-P Potentials/Work Reduction for " + network.getName() + "\n";
        #ifdef ZERO_COUNT
        header += "Zero count as one cycle\n";
        #endif

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template class BitTacticalP<uint16_t>;

}
