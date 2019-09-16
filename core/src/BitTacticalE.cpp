
#include <core/BitTacticalE.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t BitTacticalE<T>::computeTacticalEBitsPE(uint16_t act, uint16_t wgt, const int network_bits) {

        #ifdef ZERO_COUNT
        if(wgt == 0) return 1;
        #else
        if(wgt == 0) return 0;
        #endif

        uint16_t act_bits = act;

        #ifdef BOOTH_ENCODING
        act_bits = this->booth_encoding(act_bits);
        #endif

        uint8_t act_effectual_bits = this->effectualBits(act_bits);

        uint8_t bit_multiplications = act_effectual_bits * (uint8_t)network_bits;
        #ifdef ZERO_COUNT
        if(bit_multiplications == 0) bit_multiplications = 1;
        #endif

        return bit_multiplications;
    }

    template <typename T>
    uint8_t BitTacticalE<T>::computeTacticalEPE(const std::vector<std::queue<uint8_t>> &offsets) {

        // Two stages shifting
        uint8_t PE_cycles = 0;
        auto tmp_offsets = offsets;
        auto max_offset_first_stage = (uint8_t)((1u << BITS_FIRST_STAGE) - 1);

        bool still_ones = this->check_act_bits(tmp_offsets);
        while (still_ones) {

            // Get the offset for the second stage shift
            std::vector<uint8_t> last_bit;
            for (const auto &act_bits : tmp_offsets) {
                if(!act_bits.empty()) last_bit.push_back(act_bits.front());
            }
            // Must be one to enter the while loop
            uint8_t two_stage_offset = *std::min_element(last_bit.begin(), last_bit.end());
            auto max_offset = two_stage_offset + max_offset_first_stage;

            //Update values
            for (auto &act_bits : tmp_offsets) {
                if(!act_bits.empty() && act_bits.front() <= max_offset) act_bits.pop();
            }

            PE_cycles++;
            still_ones = this->check_act_bits(tmp_offsets);
        }

        #ifdef ZERO_COUNT
        if(PE_cycles == 0) PE_cycles = 1;
        #endif

        return PE_cycles;
    }

    template <typename T>
    uint8_t BitTacticalE<T>::computeTacticalEColumn(int batch, int recursion, int act_x, int act_y, int stride,
            const base::Array<T> &padded_act, const set_schedule &dense_schedule, int schedule_time, bool lstm) {

        std::vector<uint8_t> filter_cycles;
        for (int row = 0; row < this->N_ROWS; row++) {

            std::vector<std::queue<uint8_t>> offsets;
            for (int wgt_idx = 0; wgt_idx < this->N_LANES; wgt_idx++) {

                int pos = row * this->N_LANES + wgt_idx;
                auto wgt_tuple = dense_schedule[schedule_time][pos];
                int channel = std::get<0>(wgt_tuple);
                int kernel_x = std::get<1>(wgt_tuple);
                int kernel_y = std::get<2>(wgt_tuple);
                if(channel < 0) continue;

                T act_bits;
                if(lstm)
                    act_bits = padded_act.get(recursion, batch, channel);
                else
                    act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x, stride * act_y + kernel_y);

                #ifdef BOOTH_ENCODING
                act_bits = this->booth_encoding(act_bits);
                #endif

                uint8_t count = 0;
                std::queue<uint8_t> act_offsets;
                while (act_bits) {
                    auto current_bit = act_bits & 1u;
                    if (current_bit) act_offsets.push(count);
                    act_bits >>= 1;
                    count++;
                }

                offsets.push_back(act_offsets);
            }

            filter_cycles.push_back(computeTacticalEPE(offsets));
        }

        return *std::max_element(filter_cycles.begin(), filter_cycles.end());

    }

    template <typename T>
    void BitTacticalE<T>::computeTacticalETile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int stride, const base::Array<T> &padded_act,
            const set_schedule &dense_schedule, int schedule_time, std::vector<uint32_t> &cycles_per_col,
            std::vector<uint32_t> &end_previous_pallet, uint64_t &stall_cycles) {

        //Get the slowest column
        for(int window = 0; window < list_act_x.size(); window++) {
            uint8_t column_cycles = computeTacticalEColumn(batch,0,list_act_x[window],list_act_y[window], stride,
                    padded_act,dense_schedule,schedule_time,false);
            cycles_per_col[window] += column_cycles;
        }

        // Column registers
        if(this->COLUMN_REGISTERS > 0) {
            auto fastest_column = end_previous_pallet[0] + 1;
            for(auto &column_cycles : cycles_per_col) {
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
            end_previous_pallet[this->COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_col.begin(),
                    cycles_per_col.end());
        } else {
            auto slowest_col = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            auto fastest_col = *std::min_element(cycles_per_col.begin(), cycles_per_col.end());
            cycles_per_col = std::vector<uint32_t>(this->N_COLUMNS, slowest_col);
            stall_cycles += slowest_col - fastest_col;
        }

    }

    /* CYCLES */

    template <typename T>
    void BitTacticalE<T>::run(const base::Network<T> &network, const std::vector<schedule> &schedules) {

        // Initialize statistics
        int mux_entries = this->LOOKAHEAD_H + this->LOOKASIDE_D + 1;
        std::string filename = "BitTacticalE_L" + std::to_string(this->N_LANES) + "_C" +
                std::to_string(this->N_COLUMNS) + "_R" + std::to_string(this->N_ROWS) + "_T" +
                std::to_string(this->N_TILES) + "_B" + std::to_string(BITS_FIRST_STAGE) + "_CR" +
                std::to_string(this->COLUMN_REGISTERS) + "_" + this->SEARCH_SHAPE + std::to_string(mux_entries) + "("
                + std::to_string(this->LOOKAHEAD_H) + "-" + std::to_string(this->LOOKASIDE_D) + ")" + "_cycles";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto stall_cycles = stats.register_uint_t("stall_cycles", 0, sys::AverageTotal);
        auto weight_buff_reads = stats.register_uint_t("weight_buff_reads", 0, sys::AverageTotal);
        auto act_buff_reads = stats.register_uint_t("act_buff_reads", 0, sys::AverageTotal);
        auto accumulator_updates = stats.register_uint_t("accumulator_updates", 0, sys::AverageTotal);
        auto scheduled_pe = stats.register_uint_t("scheduled_pe", 0, sys::AverageTotal);
        auto idle_pe = stats.register_uint_t("idle_pe", 0, sys::AverageTotal);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        for(auto layer_it = 0; layer_it < 1; ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            act.powers_of_two_representation(layer.getActPrecision());
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

            schedule dense_schedule;
            const schedule &proto_dense_schedule = schedules[layer_it];
            if(proto_dense_schedule.empty())
                dense_schedule = this->scheduler(wgt, act_channels, fc);
            else
                dense_schedule = proto_dense_schedule;

            for(int n = 0; n < batch_size; n++) {

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
                    std::vector<std::vector<uint32_t>> cycles_per_col = std::vector<std::vector<uint32_t>>(this->N_TILES,
                            std::vector<uint32_t>(this->N_COLUMNS, 0));

                    while (this->iterateWindows(out_x, out_y, list_x, list_y, x_counter, y_counter, this->N_COLUMNS)) {

                        for (int set = 0; set < dense_schedule.size(); set += this->N_TILES) {

                            std::vector<uint64_t> tile_cycles = std::vector<uint64_t>(this->N_TILES, 0);
                            for (int tile = 0; tile < this->N_TILES; tile++) {

                                if ((set + tile) >= dense_schedule.size()) continue;
                                const auto &set_dense_schedule = dense_schedule[set + tile];

                                for (int schedule_time = 0; schedule_time < set_dense_schedule.size(); schedule_time++) {
                                    computeTacticalETile(n, list_x, list_y, stride, act, set_dense_schedule,
                                            schedule_time, cycles_per_col[tile], end_previous_pallet,
                                            batch_stall_cycles);

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                    batch_scheduled_pe += list_x.size() * this->N_ROWS;
                                    batch_idle_pe += (this->N_COLUMNS - list_x.size()) * this->N_ROWS;
                                }
                                batch_accumulator_updates++;

                            }

                        }

                    }

                    uint64_t max_tile_cycles = 0;
                    for (int tile = 0; tile < this->N_TILES; tile++) {
                        uint64_t tile_cycles = *std::max_element(cycles_per_col[tile].begin(),
                                cycles_per_col[tile].end());
                        if (tile_cycles > max_tile_cycles)
                            max_tile_cycles = tile_cycles;
                    }

                    cycles->value[layer_it][n] = max_tile_cycles;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles / this->N_TILES;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates;
                    scheduled_pe->value[layer_it][n] = batch_scheduled_pe;
                    idle_pe->value[layer_it][n] = batch_idle_pe;

                } else {

                    int column_index = 0;
                    std::vector<uint64_t> batch_cycles = std::vector<uint64_t>(this->N_TILES, 0);
                    std::vector<std::vector<uint64_t>> column_end = std::vector<std::vector<uint64_t>>(this->N_TILES,
                            std::vector<uint64_t>(this->N_COLUMNS, 0));

                    for (int r = 0; r < R; r++) {

                        for (int set = 0; set < dense_schedule.size(); set += this->N_TILES) {

                            std::vector<uint64_t> tile_cycles = std::vector<uint64_t>(this->N_TILES, 0);
                            for (int tile = 0; tile < this->N_TILES; tile++) {

                                if ((set + tile) >= dense_schedule.size()) continue;
                                const auto &set_dense_schedule = dense_schedule[set + tile];

                                for (int schedule_time = 0; schedule_time < set_dense_schedule.size(); schedule_time++) {
                                    if (batch_cycles[tile] < column_end[tile][column_index]) {
                                        batch_stall_cycles += column_end[tile][column_index] - batch_cycles[tile];
                                        batch_cycles[tile] = column_end[tile][column_index];
                                    }
                                    auto column_cycles = computeTacticalEColumn(n, r, 0, 0, 0, act, set_dense_schedule,
                                            schedule_time, lstm);
                                    column_end[tile][column_index] = batch_cycles[tile] + column_cycles;
                                    batch_cycles[tile]++;
                                    column_index++;
                                    if (column_index >= this->N_COLUMNS) column_index = 0;

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                }
                                batch_accumulator_updates++;

                            }

                        }
                    }

                    uint64_t max_tile_cycles = 0;
                    for (int tile = 0; tile < this->N_TILES; tile++) {
                        uint64_t last_column_end = *std::max_element(column_end[tile].begin(), column_end[tile].end());
                        auto tile_cycles = std::max(batch_cycles[tile], last_column_end);
                        if (tile_cycles > max_tile_cycles)
                            max_tile_cycles = tile_cycles;
                    }

                    cycles->value[layer_it][n] = max_tile_cycles;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles / this->N_TILES;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates;
                    scheduled_pe->value[layer_it][n] = (uint64_t)(num_filters * this->N_ROWS *
                            ceil(act_channels/(double)this->N_LANES));
                    auto idle_rows = this->N_ROWS - (num_filters % this->N_ROWS);
                    idle_rows = idle_rows == 16 ? 0 : idle_rows;
                    idle_pe->value[layer_it][n] = (uint64_t)(idle_rows * ceil(act_channels/(double)this->N_LANES));

                }

                act_prec->value[layer_it][n] = layer.getActPrecision();
                wgt_prec->value[layer_it][n] = layer.getWgtPrecision();

            }

        }

        //Dump statistics
        std::string header = "BitTactical-E Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(this->N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(this->N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(this->N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(this->N_TILES) + "\n";
        header += "Number of bits for first stage shifter: " + std::to_string(BITS_FIRST_STAGE) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(this->COLUMN_REGISTERS) + "\n";
        header += "Search shape: " + std::string(1, this->SEARCH_SHAPE) + "\n";
        header += "Lookahead H: " + std::to_string(this->LOOKAHEAD_H) + "\n";
        header += "Lookaside D: " + std::to_string(this->LOOKASIDE_D) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    /* POTENTIALS */

    template <typename T>
    void BitTacticalE<T>::potentials(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "BitTacticalE_potentials";
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
            act.powers_of_two_representation(layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();

            base::Array<T> wgt = layer.getWeights();
            if(conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

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

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            auto groups = act_channels / wgt_channels;
            auto it_per_group = num_filters / groups;

            auto network_bits = network.getNetwork_bits();

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                if (conv) {

                    for(int m = 0; m < num_filters; m++) {

                        // Two towers alexnet
                        int start_group = 0;
                        if(m >= it_per_group)
                            start_group = (int)wgt_channels;

                        // Fix for MobileNet
                        if(wgt_channels == 1 && act_channels != 1)
                            start_group = m;

                        for(int x = 0; x < out_x; x++) {
                            for(int y = 0; y < out_y; y++) {
                                for(int i = 0; i < Kx; i++) {
                                    for(int j = 0; j < Ky; j++) {
                                        for(int k = 0; k < wgt_channels; k++) {
                                            bit_counter += computeTacticalEBitsPE(act.get(n, start_group + k,
                                                    stride * x + i, stride * y + j),wgt.get(m, k, i, j), network_bits);
                                        }
                                    }
                                }
                            }
                        }
                    }

                } else {

                    for (int r = 0; r < R; r++) {
                        for (int m = 0; m < num_filters; m++) {
                            for (int k = 0; k < wgt_channels; k++) {
                                auto act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                                bit_counter += computeTacticalEBitsPE(act_bits, wgt.get(m, k), network_bits);
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
        std::string header = "BitTactical-E Potentials/Work Reduction for " + network.getName() + "\n";
        #ifdef BOOTH_ENCODING
        header += "Booth-like Encoding\n";
        #endif
        #ifdef ZERO_COUNT
        header += "Zero count as one cycle\n";
        #endif

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template class BitTacticalE<uint16_t>;

}