
#include <core/Loom.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t Loom<T>::computeLoomBitsPE(uint8_t act_prec, uint8_t wgt_prec) {
        return act_prec * wgt_prec;
    }

    template <typename T>
    uint8_t Loom<T>::computeLoomColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
            int init_channel, int init_filter, int stride, const base::Array<T> &padded_act,
            const base::Array<T> &wgt, int start_group, int max_channel, int max_filter, uint16_t act_mask,
            uint16_t wgt_mask, int wgt_prec, bool lstm) {

        uint32_t N_GROUPS = N_ROWS * 16 / PRECISION_GRANULARITY;
        uint32_t FILTERS_PER_GROUP = N_ROWS / N_GROUPS;

        std::vector<uint8_t> per_group_cycles (N_GROUPS, 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_act_group_bit = 0, min_act_group_bit = 16;
        uint8_t max_wgt_group_bit = 0, min_wgt_group_bit = 16;
        for (int filter = init_filter; filter < std::min(init_filter + (int)N_ROWS, max_filter); filter++) {

            if(group_counter == FILTERS_PER_GROUP)  {
                max_wgt_group_bit = 0, min_wgt_group_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for(int channel = init_channel; channel < std::min(init_channel + (int)N_LANES,max_channel); channel++) {

                // Dynamic activation precision
                if(filter == init_channel) {
                    uint16_t act_bits;
                    if (lstm)
                        act_bits = padded_act.get(recursion, batch, channel);
                    else
                        act_bits = padded_act.get(batch, start_group + channel, stride * act_x + kernel_x,
                                stride * act_y + kernel_y);

                    bool act_neg = false;
                    if ((act_bits & act_mask) != 0) {
                        act_bits = act_bits & ~act_mask;
                        act_neg = true;
                    }

                    const auto &min_max_act_bits = this->minMax(act_bits);

                    auto min_act_bit = std::get<0>(min_max_act_bits);
                    auto max_act_bit = std::get<1>(min_max_act_bits);

                    if (act_neg) max_act_bit += 1;

                    if (min_act_bit < min_act_group_bit) min_act_group_bit = min_act_bit;
                    if (max_act_bit > max_act_group_bit) max_act_group_bit = max_act_bit;
                }

                // Dynamic weight precisions
                uint16_t wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

                bool wgt_neg = false;
                if((wgt_bits & wgt_mask) != 0) {
                    wgt_bits = wgt_bits & ~wgt_mask;
                    wgt_neg = true;
                }

                const auto &min_max_wgt_bits = this->minMax(wgt_bits);

                auto min_wgt_bit = std::get<0>(min_max_wgt_bits);
                auto max_wgt_bit = std::get<1>(min_max_wgt_bits);

                if(wgt_neg) max_wgt_bit += 1;

                if(min_wgt_bit < min_wgt_group_bit) min_wgt_group_bit = min_wgt_bit;
                if(max_wgt_bit > max_wgt_group_bit) max_wgt_group_bit = max_wgt_bit;

            }

            group_counter++;
            if(group_counter == FILTERS_PER_GROUP) {
                if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_wgt_group_bit > max_wgt_group_bit) ? 
                		1 : max_wgt_group_bit - min_wgt_group_bit + 1);
                else per_group_cycles[group_index] = (uint8_t)(max_wgt_group_bit + 1);
            }
        }

        if(group_counter < FILTERS_PER_GROUP) {
            if(!LEADING_BIT) per_group_cycles[group_index] = (uint8_t)((min_wgt_group_bit > max_wgt_group_bit) ? 
            		1 : max_wgt_group_bit - min_wgt_group_bit + 1);
            else per_group_cycles[group_index] = (uint8_t)(max_wgt_group_bit + 1);
        }

        uint8_t act_cycles;
        if(!LEADING_BIT) act_cycles = (uint8_t)((min_act_group_bit > max_act_group_bit) ? 
        		1 : max_act_group_bit - min_act_group_bit + 1);
        else act_cycles = (uint8_t)(max_act_group_bit + 1);

        // Slowest PE
        auto max_wgt_cycles = *std::max_element(per_group_cycles.begin(), per_group_cycles.end());

        act_cycles = (uint8_t)ceil(act_cycles/(double)PE_SERIAL_BITS);
        max_wgt_cycles = DYNAMIC_WEIGHTS ? max_wgt_cycles : (uint8_t)wgt_prec;
        max_wgt_cycles = max_wgt_cycles;
        return max_wgt_cycles * act_cycles;

    }

    template <typename T>
    uint8_t Loom<T>::computeLoomTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int init_filter,
            int stride, const base::Array<T> &padded_act, const base::Array<T> &wgt, int start_group,
            int max_act_channel, int max_wgt_channel, int max_filter, uint16_t act_mask, uint16_t wgt_mask,
            int wgt_prec, uint64_t &stall_cycles) {

        uint32_t ACT_N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        uint32_t WINDOWS_PER_GROUP = N_COLUMNS / ACT_N_GROUPS;

        // Dynamic activation precisions
        std::vector<uint8_t> act_per_group_cycles (ACT_N_GROUPS, 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_act_group_bit = 0, min_act_group_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {

            if(group_counter == WINDOWS_PER_GROUP)  {
                max_act_group_bit = 0, min_act_group_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for (int channel = init_channel; channel < std::min(init_channel + (int)N_LANES, max_act_channel); channel++) {

                uint16_t act_bits = padded_act.get(batch, start_group + channel, stride * list_act_x[window] + kernel_x,
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

                if(min_act_bit < min_act_group_bit) min_act_group_bit = min_act_bit;
                if(max_act_bit > max_act_group_bit) max_act_group_bit = max_act_bit;

            }

            group_counter++;
            if(group_counter == WINDOWS_PER_GROUP) {
                if(!LEADING_BIT) act_per_group_cycles[group_index] = (uint8_t)((min_act_group_bit > max_act_group_bit) ? 
                		1 : max_act_group_bit - min_act_group_bit + 1);
                else act_per_group_cycles[group_index] = (uint8_t)(max_act_group_bit + 1);
            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(!LEADING_BIT) act_per_group_cycles[group_index] = (uint8_t)((min_act_group_bit > max_act_group_bit) ? 
            		1 : max_act_group_bit - min_act_group_bit + 1);
            else act_per_group_cycles[group_index] = (uint8_t)(max_act_group_bit + 1);
        }

        uint32_t WGT_N_GROUPS = N_ROWS * 16 / PRECISION_GRANULARITY;
        uint32_t FILTERS_PER_GROUP = N_ROWS / WGT_N_GROUPS;

        // Dynamic weight precisions
        std::vector<uint8_t> wgt_per_group_cycles (WGT_N_GROUPS, 0);
        group_counter = 0;
        group_index = 0;
        uint8_t max_wgt_group_bit = 0, min_wgt_group_bit = 16;
        for (int filter = init_filter; filter < std::min(init_filter + (int)N_ROWS, max_filter); filter++) {

            if(group_counter == FILTERS_PER_GROUP)  {
                max_wgt_group_bit = 0, min_wgt_group_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for(int channel = init_channel; channel < std::min(init_channel + (int)N_LANES,max_wgt_channel); channel++){

                // Dynamic weight precisions
                uint16_t wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

                bool wgt_neg = false;
                if((wgt_bits & wgt_mask) != 0) {
                    wgt_bits = wgt_bits & ~wgt_mask;
                    wgt_neg = true;
                }

                const auto &min_max_wgt_bits = this->minMax(wgt_bits);

                auto min_wgt_bit = std::get<0>(min_max_wgt_bits);
                auto max_wgt_bit = std::get<1>(min_max_wgt_bits);

                if(wgt_neg) max_wgt_bit += 1;

                if(min_wgt_bit < min_wgt_group_bit) min_wgt_group_bit = min_wgt_bit;
                if(max_wgt_bit > max_wgt_group_bit) max_wgt_group_bit = max_wgt_bit;

            }

            group_counter++;
            if(group_counter == FILTERS_PER_GROUP) {
                if(!LEADING_BIT) wgt_per_group_cycles[group_index] = (uint8_t)((min_wgt_group_bit > max_wgt_group_bit) ? 
                		1 : max_wgt_group_bit - min_wgt_group_bit + 1);
                else wgt_per_group_cycles[group_index] = (uint8_t)(max_wgt_group_bit + 1);
            }
        }

        if(group_counter < FILTERS_PER_GROUP) {
            if(!LEADING_BIT) wgt_per_group_cycles[group_index] = (uint8_t)((min_wgt_group_bit > max_wgt_group_bit) ? 
            		1 : max_wgt_group_bit - min_wgt_group_bit + 1);
            else wgt_per_group_cycles[group_index] = (uint8_t)(max_wgt_group_bit + 1);
        }

        // Slowest PE
        auto max_act_cycles = *std::max_element(act_per_group_cycles.begin(), act_per_group_cycles.end());
        auto min_act_cycles = *std::min_element(act_per_group_cycles.begin(), act_per_group_cycles.end());
        stall_cycles += max_act_cycles - min_act_cycles;

        auto max_wgt_cycles = *std::max_element(wgt_per_group_cycles.begin(), wgt_per_group_cycles.end());

        max_act_cycles = (uint8_t)ceil(max_act_cycles/(double)PE_SERIAL_BITS);
        max_wgt_cycles = DYNAMIC_WEIGHTS ? max_wgt_cycles : (uint8_t)wgt_prec;
        return max_wgt_cycles * max_act_cycles;

    }

    /* CYCLES */

    template <typename T>
    void Loom<T>::run(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "Loom_L" + std::to_string(N_LANES) + "_C" + std::to_string(N_COLUMNS) + "_R" +
                std::to_string(N_ROWS) + "_T" + std::to_string(N_TILES) + "_PG" + std::to_string(PRECISION_GRANULARITY)
                + "_PSB" + std::to_string(PE_SERIAL_BITS) + (DYNAMIC_WEIGHTS ? "_DW" : "") +
                (DYNAMIC_WEIGHTS && LEADING_BIT ? "_LB" : "") + "_cycles";
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

        auto TOTAL_ROWS = N_ROWS * N_TILES;

        for (auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            act.sign_magnitude_representation(layer.getActPrecision());
            if (fc && act.getDimensions() == 4) act.reshape_to_2D();
            if (fc) act.reshape_to_4D();

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

            long out_x = (Nx - Kx) / stride + 1;
            long out_y = (Ny - Ky) / stride + 1;

            auto groups = act_channels / wgt_channels;
            auto it_per_group = num_filters / groups;;

            auto act_layer_prec = layer.getActPrecision();
            auto act_mask = (uint16_t) (1u << (act_layer_prec - 1));

            auto wgt_layer_prec = layer.getWgtPrecision();
            auto wgt_mask = (uint16_t) (1u << (wgt_layer_prec - 1));

            for (int n = 0; n < batch_size; n++) {

                uint64_t batch_stall_cycles = 0;
                uint64_t batch_weight_buff_reads = 0;
                uint64_t batch_act_buff_reads = 0;
                uint64_t batch_accumulator_updates = 0;
                uint64_t batch_scheduled_pe = 0;
                uint64_t batch_idle_pe = 0;

                if (conv) {

                    uint64_t batch_cycles = 0;

                    std::vector<int> list_x, list_y;
                    int x_counter = 0, y_counter = 0;

                    for (int m = 0; m < num_filters; m += TOTAL_ROWS) {

                        // Two towers alexnet
                        int start_group = 0;
                        if (m >= it_per_group)
                            start_group = (int) wgt_channels;

                        // Fix for MobileNet
                        if (wgt_channels == 1 && act_channels != 1)
                            start_group = m;

                        while (this->iterateWindows(out_x, out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {

                            std::vector<uint64_t> tile_cycles = std::vector<uint64_t>(N_TILES, 0);
                            for (int tile = 0; tile < N_TILES; tile++) {
                                auto init_m = tile * N_ROWS + m;

                                for (int i = 0; i < Kx; i++) {
                                    for (int j = 0; j < Ky; j++) {
                                        for (int k = 0; k < wgt_channels; k += N_LANES) {

                                            auto tmp_tile_cycles = computeLoomTile(n, list_x, list_y, i, j, k, init_m,
                                                    stride, act, wgt, start_group, (int)act_channels, (int)wgt_channels,
                                                    (int)num_filters, act_mask, wgt_mask, wgt_layer_prec,
                                                    batch_stall_cycles);
                                            tile_cycles[tile] += tmp_tile_cycles;

                                            batch_act_buff_reads++;
                                            batch_weight_buff_reads++;
                                            batch_scheduled_pe += list_x.size() * N_ROWS;
                                            batch_idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                                        }
                                    }
                                }
                                batch_accumulator_updates++;
                            }
                            batch_cycles += *std::max_element(tile_cycles.begin(), tile_cycles.end());
                        }
                    }

                    cycles->value[layer_it][n] = batch_cycles;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads / N_TILES;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads / N_TILES;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates / N_TILES;
                    scheduled_pe->value[layer_it][n] = batch_scheduled_pe;
                    idle_pe->value[layer_it][n] = batch_idle_pe;

                } else {

                    int column_index = 0;
                    std::vector<uint64_t> batch_cycles = std::vector<uint64_t>(N_TILES, 0);
                    std::vector<std::vector<uint64_t>> column_end = std::vector<std::vector<uint64_t>>(N_TILES,
                            std::vector<uint64_t>(N_COLUMNS, 0));

                    for (int r = 0; r < R; r++) {
                        for (int m = 0; m < num_filters; m += TOTAL_ROWS) {

                            std::vector<uint64_t> tile_cycles = std::vector<uint64_t>(N_TILES, 0);
                            for (int tile = 0; tile < N_TILES; tile++) {
                                auto init_m = tile * N_ROWS + m;

                                for (int k = 0; k < wgt_channels; k += N_LANES) {
                                    if (batch_cycles[tile] < column_end[tile][column_index]) {
                                        batch_stall_cycles += column_end[tile][column_index] - batch_cycles[tile];
                                        batch_cycles[tile] = column_end[tile][column_index];
                                    }

                                    auto column_cycles = computeLoomColumn(n, r, 0, 0, 0, 0, k, init_m, 0, act, wgt, 0,
                                            (int)wgt_channels, (int)num_filters, act_mask, wgt_mask, wgt_layer_prec,
                                            lstm);

                                    column_end[tile][column_index] = batch_cycles[tile] + column_cycles;
                                    batch_cycles[tile]++;
                                    column_index++;
                                    if (column_index >= N_COLUMNS) column_index = 0;

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                }
                                batch_accumulator_updates++;
                            }
                        }
                    }

                    uint64_t max_tile_cycles = 0;
                    for (int tile = 0; tile < N_TILES; tile++) {
                        uint64_t last_column_end = *std::max_element(column_end[tile].begin(), column_end[tile].end());
                        auto tile_cycles = std::max(batch_cycles[tile], last_column_end);
                        if (tile_cycles > max_tile_cycles)
                            max_tile_cycles = tile_cycles;
                    }

                    cycles->value[layer_it][n] = max_tile_cycles;
                    stall_cycles->value[layer_it][n] = batch_stall_cycles / N_TILES;
                    weight_buff_reads->value[layer_it][n] = batch_weight_buff_reads / N_TILES;
                    act_buff_reads->value[layer_it][n] = batch_act_buff_reads / N_TILES;
                    accumulator_updates->value[layer_it][n] = batch_accumulator_updates;
                    scheduled_pe->value[layer_it][n] = (uint64_t) (num_filters * TOTAL_ROWS *
                            ceil(act_channels / (double) N_LANES));
                    auto idle_rows = TOTAL_ROWS - (num_filters % TOTAL_ROWS);
                    idle_rows = idle_rows == 16 ? 0 : idle_rows;
                    idle_pe->value[layer_it][n] = (uint64_t) (idle_rows * ceil(act_channels / (double) N_LANES));

                }

                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = wgt_layer_prec;

            }

        }

        //Dump statistics
        std::string header = "Loom Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(N_TILES) + "\n";
        header += "Number of values per group: " + std::to_string(PRECISION_GRANULARITY) + "\n";
        std::string ldn_bit = LEADING_BIT ? "True" : "False";
        header += "Calculate only leading bit: " + ldn_bit + "\n";
        header += "Number of activations processing bits per PE: " + std::to_string(PE_SERIAL_BITS) + "\n";
        std::string dyn_wgts = DYNAMIC_WEIGHTS ? "True" : "False";
        header += "Dynamic precision for weights: " + dyn_wgts + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    /* POTENTIALS */

    template <typename T>
    void Loom<T>::potentials(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "Loom_potentials";
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

            auto groups = act_channels / wgt_channels;
            auto it_per_group = num_filters / groups;

            // Get layer precision
            auto act_layer_prec = layer.getActPrecision();
            auto wgt_layer_prec = layer.getWgtPrecision();

            auto network_bits = network.getNetwork_bits();

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                bit_counter = (uint64_t)computeLoomBitsPE((uint8_t)act_layer_prec, (uint8_t)wgt_layer_prec);
                bit_counter *= conv ? out_x * out_y * Kx * Ky * wgt_channels * num_filters:
                        wgt_channels * num_filters * R;

                bit_multiplications->value[layer_it][n] = bit_counter;
                work_reduction->value[layer_it][n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
                speedup->value[layer_it][n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
                par_mult->value[layer_it][n] = parallel_mult;
                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = wgt_layer_prec;
            }

        }

        //Dump statistics
        std::string header = "Loom Potentials/Work Reduction for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template class Loom<uint16_t>;

}