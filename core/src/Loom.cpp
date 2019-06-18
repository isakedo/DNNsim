
#include <core/Loom.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t Loom<T>::computeLoomBitsPE(uint8_t act_prec, uint8_t wgt_prec) {
        return act_prec * wgt_prec;
    }

    template <typename T>
    uint8_t Loom<T>::computeLoomColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
            int init_channel, int init_filter, int stride, const cnpy::Array<T> &padded_act,
            const cnpy::Array<T> &wgt, int start_group, int max_channel, int max_filter, int act_mask, int wgt_mask,
            int wgt_prec, bool lstm) {

        int N_GROUPS = N_ROWS * 16 / PRECISION_GRANULARITY;
        int FILTERS_PER_GROUP = N_ROWS / N_GROUPS;

        std::vector<uint8_t> per_group_cycles (N_GROUPS, 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_act_group_bit = 0, min_act_group_bit = 16;
        uint8_t max_wgt_group_bit = 0, min_wgt_group_bit = 16;
        for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {

            if(group_counter == FILTERS_PER_GROUP)  {
                max_wgt_group_bit = 0, min_wgt_group_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for(int channel = init_channel; channel < std::min(init_channel + N_LANES,max_channel); channel++) {

                // Dynamic activation precision
                if(filter == init_channel) {
                    T act_bits;
                    if (lstm)
                        act_bits = padded_act.get(recursion, batch, channel);
                    else
                        act_bits = padded_act.get(batch, start_group + channel, stride * act_x + kernel_x,
                                stride * act_y + kernel_y);

                    bool act_neg = false;
                    if ((act_bits & act_mask) != 0) {
                        act_bits = act_bits & ~(uint16_t) act_mask;
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
                auto wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

                bool wgt_neg = false;
                if((wgt_bits & wgt_mask) != 0) {
                    wgt_bits = wgt_bits & ~(uint16_t)wgt_mask;
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
        max_wgt_cycles = (uint8_t)ceil(max_wgt_cycles/(double)PE_SERIAL_BITS);
        return max_wgt_cycles * act_cycles;

    }

    template <typename T>
    uint8_t Loom<T>::computeLoomTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int init_filter,
            int stride, const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int start_group,
            int max_act_channel, int max_wgt_channel, int max_filter, int act_mask, int wgt_mask, int wgt_prec,
            sys::Statistics::Stats &stats) {

        int ACT_N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        int WINDOWS_PER_GROUP = N_COLUMNS / ACT_N_GROUPS;

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

            for (int channel = init_channel; channel < std::min(init_channel + N_LANES, max_act_channel); channel++) {

                auto act_bits = padded_act.get(batch, start_group + channel, stride * list_act_x[window] + kernel_x,
                        stride * list_act_y[window] + kernel_y);

                bool neg = false;
                if((act_bits & act_mask) != 0) {
                    act_bits = act_bits & ~(uint16_t)act_mask;
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

        int WGT_N_GROUPS = N_ROWS * 16 / PRECISION_GRANULARITY;
        int FILTERS_PER_GROUP = N_ROWS / WGT_N_GROUPS;

        // Dynamic weight precisions
        std::vector<uint8_t> wgt_per_group_cycles (WGT_N_GROUPS, 0);
        group_counter = 0;
        group_index = 0;
        uint8_t max_wgt_group_bit = 0, min_wgt_group_bit = 16;
        for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {

            if(group_counter == FILTERS_PER_GROUP)  {
                max_wgt_group_bit = 0, min_wgt_group_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for(int channel = init_channel; channel < std::min(init_channel + N_LANES,max_wgt_channel); channel++){

                // Dynamic weight precisions
                auto wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

                bool wgt_neg = false;
                if((wgt_bits & wgt_mask) != 0) {
                    wgt_bits = wgt_bits & ~(uint16_t)wgt_mask;
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
        stats.stall_cycles.back()[batch] += max_act_cycles - min_act_cycles;

        auto max_wgt_cycles = *std::max_element(wgt_per_group_cycles.begin(), wgt_per_group_cycles.end());

        max_act_cycles = (uint8_t)ceil(max_act_cycles/(double)PE_SERIAL_BITS);
        max_wgt_cycles = DYNAMIC_WEIGHTS ? max_wgt_cycles : (uint8_t)wgt_prec;
        max_wgt_cycles = (uint8_t)ceil(max_wgt_cycles/(double)PE_SERIAL_BITS);
        return max_wgt_cycles * max_act_cycles;

    }

    /* CYCLES */

    template <typename T>
    void Loom<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.sign_magnitude_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.sign_magnitude_representation(layer.getWgtPrecision());
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

        int groups = act_channels / wgt_channels;
        int it_per_group = num_filters / groups;

        auto act_prec = layer.getActPrecision();
        auto act_mask = (uint16_t)(1 << (act_prec - 1));

        auto wgt_prec = layer.getWgtPrecision();
        auto wgt_mask = (uint16_t)(1 << (wgt_prec - 1));

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
            uint64_t cycles = 0;
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;
            uint64_t scheduled_pe = 0;
            uint64_t idle_pe = 0;

            for(int m = 0; m < num_filters; m += N_ROWS) {

                // Two towers alexnet
                int start_group = 0;
                if(m >= it_per_group)
                    start_group = wgt_channels;

                // Fix for MobileNet
                if(wgt_channels == 1 && act_channels != 1)
                    start_group = m;

                while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            for (int k = 0; k < wgt_channels; k += N_LANES) {
                                cycles += computeLoomTile(n,list_x, list_y, i, j, k, m, stride, act, wgt, start_group,
                                        act_channels, wgt_channels, num_filters, act_mask, wgt_mask, wgt_prec, stats);

                                act_buff_reads++;
                                weight_buff_reads++;
                                scheduled_pe += list_x.size() * N_ROWS;
                                idle_pe += (N_COLUMNS - list_x.size()) * N_ROWS;
                            }
                        }
                    }
                    accumulator_updates++;
                }
            }

            stats.cycles.back()[n] = cycles;
            stats.weight_buff_reads.back()[n] = weight_buff_reads;
            stats.act_buff_reads.back()[n] = act_buff_reads;
            stats.accumulator_updates.back()[n] = accumulator_updates;
            stats.scheduled_pe.back()[n] = scheduled_pe;
            stats.idle_pe.back()[n] = idle_pe;

        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void Loom<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.sign_magnitude_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.sign_magnitude_representation(layer.getWgtPrecision());
        wgt.reshape_to_4D();

        if(layer.getType() == "InnerProduct") {
            if (act.getDimensions() == 4) act.reshape_to_2D();
            act.reshape_to_4D();
        }

        bool lstm = layer.getType() == "LSTM";

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

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
        int wgt_channels = wgt_shape[1];

        auto act_prec = layer.getActPrecision();
        auto act_mask = (uint16_t)(1 << (act_prec - 1));

        auto wgt_prec = layer.getWgtPrecision();
        auto wgt_mask = (uint16_t)(1 << (wgt_prec - 1));

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
                for (int m = 0; m < num_filters; m += N_ROWS) {
                    for (int k = 0; k < wgt_channels; k += WEIGHT_LANES) {
                        cycles += computeLoomColumn(n,r,0,0,0,0,k,m,0,act,wgt,0,wgt_channels,num_filters,act_mask,
                                wgt_mask,wgt_prec,lstm);
                        act_buff_reads++;
                        weight_buff_reads++;
                    }
                    accumulator_updates++;
                }
            }

            stats.cycles.back()[n] = stats.cycles.back()[n];
            stats.weight_buff_reads.back()[n] = weight_buff_reads;
            stats.act_buff_reads.back()[n] = act_buff_reads;
            stats.accumulator_updates.back()[n] = accumulator_updates;
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
            std::vector<int>column_end = std::vector<int>(N_COLUMNS, 0);
            uint64_t cycles = 0;
            uint64_t stall_cycles = 0;
            uint64_t weight_buff_reads = 0;
            uint64_t act_buff_reads = 0;
            uint64_t accumulator_updates = 0;

            for (int r = 0; r < R; r++) {
                for (int m = 0; m < num_filters; m += N_ROWS) {
                    for (int k = 0; k < wgt_channels; k += N_LANES) {
                        if(cycles < column_end[column_index]) {
                            stall_cycles = column_end[column_index] - cycles;
                            cycles = column_end[column_index];
                        }
                        auto column_cycles = computeLoomColumn(n,r,0,0,0,0,k,m,0,act,wgt,0,wgt_channels,num_filters,
                                act_mask,wgt_mask,wgt_prec,lstm);
                        column_end[column_index] = cycles + column_cycles;
                        cycles++;
                        column_index++;
                        if(column_index >= N_COLUMNS) column_index = 0;

                        act_buff_reads++;
                        weight_buff_reads++;
                    }
                    accumulator_updates++;
                }
            }

            uint64_t last_column_end = *std::max_element(column_end.begin(), column_end.end());
            stats.cycles.back()[n] = std::max(cycles, last_column_end);
            stats.stall_cycles.back()[n] = stall_cycles;
            stats.weight_buff_reads.back()[n] = weight_buff_reads;
            stats.act_buff_reads.back()[n] = act_buff_reads;
            stats.accumulator_updates.back()[n] = accumulator_updates;
            stats.scheduled_pe.back()[n] = num_filters * N_ROWS * ceil(act_channels/(double)N_LANES);
            auto idle_rows = N_ROWS - (num_filters % N_ROWS);
            idle_rows = idle_rows == 16 ? 0 : idle_rows;
            stats.idle_pe.back()[n] = idle_rows * ceil(act_channels/(double)N_LANES);

        }

        #endif

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void Loom<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        stats.arch = "Loom_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS) + "_PG" +
                std::to_string(PRECISION_GRANULARITY) + "_PSB" + std::to_string(PE_SERIAL_BITS) +
                (DYNAMIC_WEIGHTS ? "_DW" : "") + (DYNAMIC_WEIGHTS && LEADING_BIT ? "_LB" : "");

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(layer.getWgtPrecision());
                computeConvolution(layer, stats);
            } else if(layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(layer.getWgtPrecision());
                computeInnerProduct(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* POTENTIALS */

    template <typename T>
    void Loom<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        const cnpy::Array<T> &act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = 1;
        int Nx = act_shape[2];
        int Ny = act_shape[3];

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        // Operations
        const auto parallel_mult = (uint64_t)(num_filters * out_x * out_y * Kx * Ky * wgt_channels);
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));
        uint64_t bit_counter = 0;

        // Get layer precision
        auto act_prec = layer.getActPrecision();
        auto wgt_prec = layer.getWgtPrecision();

        // Convolution
        for(int n = 0; n < batch_size; n++) {
            bit_counter = (uint64_t)computeLoomBitsPE((uint8_t)act_prec, (uint8_t)wgt_prec) * out_x * out_y * Kx * Ky *
                          wgt_channels * num_filters;
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
    void Loom<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        const cnpy::Array<T> &wgt = layer.getWeights();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = 1;
        int R = (layer.getType() == "LSTM") ? act_shape[0] : 1;

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels * R;
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));
        uint64_t bit_counter = 0;

        // Get layer precision
        auto act_prec = layer.getActPrecision();
        auto wgt_prec = layer.getWgtPrecision();

        for (int n = 0; n < batch_size; n++) {
            bit_counter = (uint64_t)computeLoomBitsPE((uint8_t)act_prec, (uint8_t)wgt_prec) * wgt_channels *
                    num_filters * R;
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
    void Loom<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "Loom";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(layer.getWgtPrecision());
                computePotentialsConvolution(layer,stats,network.getNetwork_bits());
            } else if (layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(layer.getWgtPrecision());
                computePotentialsInnerProduct(layer,stats,network.getNetwork_bits());
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class Loom<uint16_t>;

}