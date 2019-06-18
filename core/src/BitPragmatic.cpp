
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
        auto max_offset_first_stage = (uint8_t)((1 << BITS_FIRST_STAGE) - 1);

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
            int kernel_y, int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel, bool lstm) {

        std::vector<std::queue<uint8_t>> offsets;
        for(int channel = init_channel; channel < std::min(init_channel + N_LANES,max_channel); channel++) {

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
                auto current_bit = act_bits & 1;
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
            const cnpy::Array<T> &padded_act, int act_mask, int max_channel, std::vector<uint32_t> &cycles_per_col,
            std::vector<uint32_t> &end_previous_pallet, sys::Statistics::Stats &stats) {

        for(int window = 0; window < list_act_x.size(); window++) {

            std::vector<std::queue<uint8_t>> offsets;
            for(int channel = init_channel; channel < std::min(init_channel + N_LANES,max_channel); channel++) {

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
                    auto current_bit = act_bits & 1;
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
            stats.stall_cycles.back()[batch] += (end_previous_pallet[0] + 1) - fastest_column;

            //Update end_previous_pallet
            for(int i = 0; i < COLUMN_REGISTERS - 1; i++) {
                end_previous_pallet[i] = end_previous_pallet[i + 1];
            }
            end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
        } else {
            auto slowest_column = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            auto fastest_column = *std::min_element(cycles_per_col.begin(), cycles_per_col.end());
            cycles_per_col = std::vector<uint32_t>(N_COLUMNS,slowest_column);
            stats.stall_cycles.back()[batch] += slowest_column - fastest_column;
        }

    }

    template <typename T>
    void BitPragmatic<T>::computePragmatic2DTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_filter, int stride,
            const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int max_filter,
            std::vector<uint32_t> &cycles_per_col, std::vector<uint32_t> &end_previous_pallet,
            sys::Statistics::Stats &stats) {

        //Get the slowest column
        for(int window = 0; window < list_act_x.size(); window++) {
            std::vector<std::queue<uint8_t>> offsets;
            for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {

                auto act_bits = padded_act.get(batch, filter, stride * list_act_x[window] + kernel_x,
	            	    stride * list_act_y[window] + kernel_y);

                #ifdef BOOTH_ENCODING
                act_bits = this->booth_encoding(act_bits);
                #endif

                uint8_t count = 0;
                std::queue<uint8_t> act_offsets;
                while (act_bits) {
                    auto current_bit = act_bits & 1;
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
            stats.stall_cycles.back()[batch] += (end_previous_pallet[0] + 1) - fastest_column;

            //Update end_previous_pallet
            for(int i = 0; i < COLUMN_REGISTERS - 1; i++) {
                end_previous_pallet[i] = end_previous_pallet[i + 1];
            }
            end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
        } else {
            auto slowest_column = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            auto fastest_column = *std::min_element(cycles_per_col.begin(), cycles_per_col.end());
            cycles_per_col = std::vector<uint32_t>(N_COLUMNS,slowest_column);
            stats.stall_cycles.back()[batch] += slowest_column - fastest_column;
        }

    }

    /* CYCLES */

    template <typename T>
    void BitPragmatic<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(!DIFFY) act.powers_of_two_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);

        if(act.getShape()[1] == 3 && stride > 1) {
            act.reshape_first_layer_act((uint16_t)stride);
            wgt.reshape_first_layer_wgt((uint16_t)stride);
            stride = 1;
        }

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        int Nx = act_shape[2];
        int Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        long out_x = (Nx - Kx)/stride + 1;
        long out_y = (Ny - Ky)/stride + 1;

        auto act_prec = layer.getActPrecision();
        auto act_mask = (uint16_t)(1 << (act_prec - 1));

        int groups = act_channels / wgt_channels;
        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS/groups);
        auto baseline_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS/groups);

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.weight_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.accumulator_updates.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.scheduled_pe.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_pe.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n = 0; n < batch_size; n++) {

            std::vector<int> list_x, list_y;
            int x_counter = 0, y_counter = 0;
            std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(COLUMN_REGISTERS, 0);
            std::vector<uint32_t> cycles_per_col = std::vector<uint32_t>(N_COLUMNS, 0);
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;
            uint64_t scheduled_pe = 0;
            uint64_t idle_pe = 0;

            while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter, y_counter, N_COLUMNS)) {
                for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = 0; k < act_channels; k += N_LANES) {
                            computePragmaticTile(n,list_x, list_y, i, j, k, stride, act, act_mask, act_channels,
                                    cycles_per_col, end_previous_pallet, stats);

                            act_buff_reads++;
                            weight_buff_reads++;
                            scheduled_pe += list_x.size() * N_ROWS;
                            idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                        }
                    }
                }
                accumulator_updates++;
            }

            auto batch_cycles = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            stats.cycles.back()[n] = batch_cycles * num_filters_sets;
            stats.stall_cycles.back()[n] *= num_filters_sets;
            stats.weight_buff_reads.back()[n] = weight_buff_reads * num_filters_sets;
            stats.act_buff_reads.back()[n] = act_buff_reads * num_filters_sets;
            stats.accumulator_updates.back()[n] = accumulator_updates * num_filters_sets;
            stats.scheduled_pe.back()[n] = scheduled_pe * num_filters_sets;
            stats.idle_pe.back()[n] = idle_pe * num_filters_sets;

        }

        auto base_cycles = (uint64_t)(out_x * out_y * ceil(act_channels/(double)N_LANES) * Kx * Ky *
                baseline_filters_sets);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

	template <typename T>
    void BitPragmatic<T>::computeConvolution2D(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int Nx = act_shape[2];
        int Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        long out_x = (Nx - Kx)/stride + 1;
        long out_y = (Ny - Ky)/stride + 1;

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.weight_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.accumulator_updates.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.scheduled_pe.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_pe.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n = 0; n < batch_size; n++) {

            std::vector<int> list_x, list_y;
            int x_counter = 0, y_counter = 0;
            std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(COLUMN_REGISTERS, 0);
            std::vector<uint32_t> cycles_per_col = std::vector<uint32_t>(N_COLUMNS, 0);
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;
            uint64_t scheduled_pe = 0;
            uint64_t idle_pe = 0;

            for(int m = 0; m < num_filters; m += N_ROWS) {
                while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            computePragmatic2DTile(n,list_x, list_y, i, j, m, stride, act, wgt, num_filters,
                                    cycles_per_col, end_previous_pallet, stats);

                            act_buff_reads++;
                            weight_buff_reads++;
                            scheduled_pe += list_x.size() * N_ROWS;
                            idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                        }
                    }
                    accumulator_updates++;
                }
            }

            stats.cycles.back()[n] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            stats.weight_buff_reads.back()[n] = weight_buff_reads;
            stats.act_buff_reads.back()[n] = act_buff_reads;
            stats.accumulator_updates.back()[n] = accumulator_updates;
            stats.scheduled_pe.back()[n] = scheduled_pe;
            stats.idle_pe.back()[n] = idle_pe;

        }

        auto base_cycles = (uint64_t)(out_x * out_y * ceil(wgt_channels/(double)N_LANES) * Kx * Ky *
                ceil(num_filters/(double)N_ROWS));

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

    template <typename T>
    void BitPragmatic<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());

        if(layer.getType() == "InnerProduct") {
            if (act.getDimensions() == 4) act.reshape_to_2D();
            act.reshape_to_4D();
        }

        bool lstm = layer.getType() == "LSTM";

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = layer.getWeights().getShape();

        int batch_size, act_channels, R;
        if(lstm) {
            R = act_shape[0];
            batch_size = act_shape[1];
            act_channels = act_shape[2];
        } else {
            R = 1;
            batch_size = act_shape[0];
            act_channels = act_shape[1];
        }
        if(this->FAST_MODE) batch_size = 1;

        int num_filters = wgt_shape[0];

        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS);
        auto baseline_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS);

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.weight_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_buff_reads.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.accumulator_updates.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.scheduled_pe.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_pe.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        #ifndef FC_MULTIPLEX_COLUMNS

        // All FC in one column
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n < batch_size; n++) {

            uint64_t cycles = 0;
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;

            for (int r = 0; r < R; r++) {
                for (int k = 0; k < act_channels; k += WEIGHT_LANES) {
                    cycles += computePragmaticColumn(n,r,0,0,0,0,k,0,act,act_channels,lstm);
                    act_buff_reads++;
                    weight_buff_reads++;
                }
                accumulator_updates++;
            }

            stats.cycles.back()[n] = stats.cycles.back()[n] * num_filters_sets;
            stats.weight_buff_reads.back()[n] = weight_buff_reads * num_filters_sets;
            stats.act_buff_reads.back()[n] = act_buff_reads * num_filters_sets;
            stats.accumulator_updates.back()[n] = accumulator_updates * num_filters_sets;
            stats.scheduled_pe.back()[n] = num_filters * N_ROWS * ceil(act_channels/(double)WEIGHT_LANES);
            auto idle_rows = N_ROWS - (num_filters % N_ROWS);
            stats.idle_pe.back()[n] = idle_rows * ceil(act_channels/(double)WEIGHT_LANES);

        }

        #else

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n < batch_size; n++) {

            int column_index = 0;
            std::vector<int> column_end = std::vector<int>(N_COLUMNS, 0);
            uint64_t cycles = 0;
            uint64_t stall_cycles = 0;
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;

            for (int r = 0; r < R; r++) {
                for (int k = 0; k < act_channels; k += N_LANES) {
                    if(cycles < column_end[column_index]) {
                        stall_cycles = column_end[column_index] - cycles;
                        cycles = column_end[column_index];
                    }
                    auto column_cycles = computePragmaticColumn(n,r,0,0,0,0,k,0,act,act_channels,lstm);
                    column_end[column_index] = cycles + column_cycles;
                    cycles++;
                    column_index++;
                    if(column_index >= N_COLUMNS) column_index = 0;

                    act_buff_reads++;
                    weight_buff_reads++;
                }
                accumulator_updates++;
            }

            uint64_t last_column_end = *std::max_element(column_end.begin(), column_end.end());
            uint64_t last_column_rem_cycles = last_column_end - cycles;
            stats.cycles.back()[n] = cycles * num_filters_sets;
            stats.cycles.back()[n] += last_column_rem_cycles;
            stats.stall_cycles.back()[n] = stall_cycles * num_filters_sets;
            stats.weight_buff_reads.back()[n] = weight_buff_reads * num_filters_sets;
            stats.act_buff_reads.back()[n] = act_buff_reads * num_filters_sets;
            stats.accumulator_updates.back()[n] = accumulator_updates * num_filters_sets;
            stats.scheduled_pe.back()[n] = num_filters * N_ROWS * ceil(act_channels/(double)N_LANES);
            auto idle_rows = N_ROWS - (num_filters % N_ROWS);
            idle_rows = idle_rows == 16 ? 0 : idle_rows;
            stats.idle_pe.back()[n] = idle_rows * ceil(act_channels/(double)N_LANES);

        }

        #endif

        auto base_cycles = (uint64_t)(ceil(act_channels/(double)N_LANES) * baseline_filters_sets * R);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

    template <typename T>
    void BitPragmatic<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        std::string arch = "BitPragmatic";
        arch += (DIFFY ? "_Diffy" : "");
        stats.arch = arch + "_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS) + "_B" +
                std::to_string(BITS_FIRST_STAGE) + "_CR" + std::to_string(COLUMN_REGISTERS);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                if(layer.getWeights().getShape()[1] == 1 && layer.getActivations().getShape()[1] != 1)
                    computeConvolution2D(layer, stats);
                else
                    computeConvolution(layer, stats);
            } else if (layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                computeInnerProduct(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);

    }

    /* POTENTIALS */

    template <typename T>
    void BitPragmatic<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        int Nx = act_shape[2];
        int Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        int groups = act_channels / wgt_channels;
        auto num_filters_sets = (uint32_t)ceil((double)num_filters/groups);

        // Operations
        const auto parallel_mult = (uint64_t)(num_filters * out_x * out_y * Kx * Ky * wgt_channels);
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n = 0; n < batch_size; n++) {
            uint64_t bit_counter = 0;
            for(int x = 0; x < out_x; x++) {
                for(int y = 0; y < out_y; y++) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            for (int k = 0; k < act_channels; k ++) {
                                bit_counter += computePragmaticBitsPE(act.get(n, k, stride * x + i, stride * y + j),
                                        network_bits);
                            }
                        }
                    }
                }
            }
            bit_counter *= num_filters_sets;
            double MAX_BITS = network_bits * network_bits;
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
            stats.speedup.back()[n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
            stats.bit_multiplications.back()[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.parallel_multiplications.push_back(parallel_mult);

    }

    template <typename T>
    void BitPragmatic<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());
        if(act.getDimensions() == 4) act.reshape_to_2D();
        const cnpy::Array<T> &wgt = layer.getWeights();

        bool lstm = layer.getType() == "LSTM";

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int R = lstm ? act_shape[0] : 1;

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        if(this->FAST_MODE) batch_size = 1;

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels * R;
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));

        int n;

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n < batch_size; n++) {
            uint64_t bit_counter = 0;
            for(int r = 0; r < R; r++) {
                for (int k = 0; k < wgt_channels; k++) {
                    auto act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                    bit_counter += computePragmaticBitsPE(act_bits,network_bits);
                }
            }
            bit_counter *= num_filters;
            double MAX_BITS = network_bits * network_bits;
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
            stats.speedup.back()[n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
            stats.bit_multiplications.back()[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.parallel_multiplications.push_back(parallel_mult);

    }

    template <typename T>
    void BitPragmatic<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "BitPragmatic";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(0);
                computePotentialsConvolution(layer,stats,network.getNetwork_bits());
            } else if (layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getWgtPrecision());
                stats.wgt_prec.push_back(0);
                computePotentialsInnerProduct(layer,stats,network.getNetwork_bits());
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class BitPragmatic<uint16_t>;

}