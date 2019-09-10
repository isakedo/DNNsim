
#include <core/BitPragmatic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t BitPragmatic<T>::computePragmaticBitsPE(uint16_t act, const int network_bits) {

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
    uint8_t BitPragmatic<T>::computePragmaticPE(const std::vector<std::queue<uint8_t>> &offsets) {

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
    uint8_t BitPragmatic<T>::computePragmaticColumn(int batch, int recursion, int act_x, int act_y, int kernel_x,
            int kernel_y, int init_channel, int stride, const base::Array<T> &padded_act, int max_channel, bool lstm) {

        std::vector<std::queue<uint8_t>> offsets;
        for(int channel = init_channel; channel < std::min(init_channel + (int)N_LANES,max_channel); channel++) {

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
                if(current_bit) act_offsets.push(count);
                act_bits >>= 1;
                count++;
            }

            offsets.push_back(act_offsets);
        }

        return computePragmaticPE(offsets);

    }

    template <typename T>
    void BitPragmatic<T>::computePragmaticTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int stride,
            const base::Array<T> &padded_act, int act_mask, int max_channel, std::vector<uint32_t> &cycles_per_col,
            std::vector<uint32_t> &end_previous_pallet, uint64_t &stall_cycles) {

        for(int window = 0; window < list_act_x.size(); window++) {

            std::vector<std::queue<uint8_t>> offsets;
            for(int channel = init_channel; channel < std::min(init_channel + (int)N_LANES,max_channel); channel++) {

                // Computation cycles
                uint16_t act_bits;
                if(DIFFY) {
                    short raw_act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                            stride * list_act_y[window] + kernel_y);
                    short prev_act_bits = (stride * list_act_y[window] - stride < 0) ? 0 :
                            padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                                    stride * list_act_y[window] + kernel_y - stride);

                    raw_act_bits = raw_act_bits - prev_act_bits;

                    act_bits = this->sign_magnitude(raw_act_bits,(uint16_t)act_mask);
                } else {
                    act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                            stride * list_act_y[window] + kernel_y);
                }

                #ifdef BOOTH_ENCODING
                act_bits = this->booth_encoding(act_bits);
                #endif

                uint8_t count = 0;
                std::queue<uint8_t> act_offsets;
                while (act_bits) {
                    auto current_bit = act_bits & 1u;
                    if(current_bit) act_offsets.push(count);
                    act_bits >>= 1;
                    count++;
                }

                offsets.push_back(act_offsets);
            }

            uint8_t column_cycles = computePragmaticPE(offsets);
            cycles_per_col[window] += column_cycles;
        }

        // Column registers
        if(COLUMN_REGISTERS > 0) {
            auto fastest_column = end_previous_pallet[0] + 1;
            for(auto &column_cycles : cycles_per_col) {
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
            end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
        } else {
            auto slowest_column = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            auto fastest_column = *std::min_element(cycles_per_col.begin(), cycles_per_col.end());
            cycles_per_col = std::vector<uint32_t>(N_COLUMNS,slowest_column);
            stall_cycles += slowest_column - fastest_column;
        }

    }

    template <typename T>
    void BitPragmatic<T>::computePragmatic2DTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_filter, int stride,
            const base::Array<T> &padded_act, const base::Array<T> &wgt, int max_filter,
            std::vector<uint32_t> &cycles_per_col, std::vector<uint32_t> &end_previous_pallet,
            uint64_t &stall_cycles) {

        //Get the slowest column
        for(int window = 0; window < list_act_x.size(); window++) {
            std::vector<std::queue<uint8_t>> offsets;
            for (int filter = init_filter; filter < std::min(init_filter + (int)N_ROWS, max_filter); filter++) {

                auto act_bits = padded_act.get(batch, filter, stride * list_act_x[window] + kernel_x,
	            	    stride * list_act_y[window] + kernel_y);

                #ifdef BOOTH_ENCODING
                act_bits = this->booth_encoding(act_bits);
                #endif

                uint8_t count = 0;
                std::queue<uint8_t> act_offsets;
                while (act_bits) {
                    auto current_bit = act_bits & 1u;
                    if(current_bit) act_offsets.push(count);
                    act_bits >>= 1;
                    count++;
                }

                offsets.push_back(act_offsets);

	        }
            uint8_t column_cycles = computePragmaticPE(offsets);
            cycles_per_col[window] += column_cycles;
        }

        // Column registers
        if(COLUMN_REGISTERS > 0) {
            auto fastest_column = end_previous_pallet[0] + 1;
            for(auto &column_cycles : cycles_per_col) {
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
            end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
        } else {
            auto slowest_column = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            auto fastest_column = *std::min_element(cycles_per_col.begin(), cycles_per_col.end());
            cycles_per_col = std::vector<uint32_t>(N_COLUMNS,slowest_column);
            stall_cycles += slowest_column - fastest_column;
        }

    }

    /* CYCLES */

    template <typename T>
    void BitPragmatic<T>::run(const base::Network<T> &network) {

        // Initialize statistics
        std::string arch = "BitPragmatic";
        arch += (DIFFY ? "_Diffy" : "");
        std::string filename = arch + "_C" + std::to_string(N_COLUMNS) + "_R" +
                std::to_string(N_ROWS) + "_B" + std::to_string(BITS_FIRST_STAGE) + "_CR" +
                std::to_string(COLUMN_REGISTERS) + "_cycles";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto baseline_cycles = stats.register_uint_t("baseline_cycles", 0, sys::AverageTotal);
        auto speedup = stats.register_double_t("speedup", 0, sys::Average);
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
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            if(!DIFFY) act.powers_of_two_representation(layer.getActPrecision());
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

            auto groups = act_channels / wgt_channels;
            auto num_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS/groups);
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
                    std::vector<uint32_t> cycles_per_col = std::vector<uint32_t>(N_COLUMNS, 0);

                    for(int m = 0; m < num_filters; m += N_ROWS) {
                        while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                            for (int i = 0; i < Kx; i++) {
                                for (int j = 0; j < Ky; j++) {
                                    computePragmatic2DTile(n,list_x, list_y, i, j, m, stride, act, wgt,
                                            (int)num_filters, cycles_per_col, end_previous_pallet, batch_stall_cycles);

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                    batch_scheduled_pe += list_x.size() * N_ROWS;
                                    batch_idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                                }
                            }
                            batch_accumulator_updates++;
                        }
                    }

                    cycles->value[layer_it][n] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
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
                    std::vector<uint32_t> cycles_per_col = std::vector<uint32_t>(N_COLUMNS, 0);

                    while (this->iterateWindows(out_x, out_y, list_x, list_y, x_counter, y_counter, N_COLUMNS)) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = 0; k < act_channels; k += N_LANES) {
                                    computePragmaticTile(n, list_x, list_y, i, j, k, stride, act, act_mask,
                                            (int)act_channels, cycles_per_col, end_previous_pallet, batch_stall_cycles);

                                    batch_act_buff_reads++;
                                    batch_weight_buff_reads++;
                                    batch_scheduled_pe += list_x.size() * N_ROWS;
                                    batch_idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                                }
                            }
                        }
                        batch_accumulator_updates++;
                    }

                    batch_cycles = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
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
                                batch_stall_cycles = column_end[column_index] - batch_cycles;
                                batch_cycles = column_end[column_index];
                            }
                            auto column_cycles = computePragmaticColumn(n,r,0,0,0,0,k,0,act,(int)act_channels,lstm);
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
                    auto idle_rows = N_ROWS - (num_filters % N_ROWS);
                    idle_rows = idle_rows == 16 ? 0 : idle_rows;
                    idle_pe->value[layer_it][n] = (uint64_t)(idle_rows * ceil(act_channels/(double)N_LANES));
                    baseline_cycles->value[layer_it][n] = base_cycles;
                    speedup->value[layer_it][n] = base_cycles / (double)cycles->value[layer_it][n];

                }

                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = layer.getWgtPrecision();

            }

        }

        //Dump statistics
        std::string header = "BitPragmatic Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of bits for first stage shifter: " + std::to_string(BITS_FIRST_STAGE) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(COLUMN_REGISTERS) + "\n";
        header += "Diffy: " + std::to_string(DIFFY) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    /* POTENTIALS */

    template <typename T>
    void BitPragmatic<T>::potentials(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "BitPragmatic_potentials";
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
            auto num_filters_sets = (uint32_t)ceil((double)num_filters/groups);

            auto network_bits = network.getNetwork_bits();

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                if (conv) {

                    for(int x = 0; x < out_x; x++) {
                        for(int y = 0; y < out_y; y++) {
                            for (int i = 0; i < Kx; i++) {
                                for (int j = 0; j < Ky; j++) {
                                    for (int k = 0; k < act_channels; k ++) {
                                        bit_counter += computePragmaticBitsPE(act.get(n, k, stride * x + i,
                                                stride * y + j), network_bits);
                                    }
                                }
                            }
                        }
                    }
                    bit_counter *= num_filters_sets;

                } else {

                    bit_counter = 0;
                    for(int r = 0; r < R; r++) {
                        for (int k = 0; k < wgt_channels; k++) {
                            auto act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                            bit_counter += computePragmaticBitsPE(act_bits, network_bits);
                        }
                    }
                    bit_counter *= num_filters;

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
        std::string header = "BitPragmatic Potentials/Work Reduction for " + network.getName() + "\n";
        #ifdef BOOTH_ENCODING
        header += "Booth-like Encoding\n";
        #endif
        #ifdef ZERO_COUNT
        header += "Zero count as one cycle\n";
        #endif

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template class BitPragmatic<uint16_t>;

}