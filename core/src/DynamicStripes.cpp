
#include <core/DynamicStripes.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t DynamicStripes<T>::computeDynamicStripesBitsPE(uint8_t layer_prec, const int network_bits) {
        return layer_prec * (uint8_t)network_bits;
    }

    template <typename T>
    uint8_t DynamicStripes<T>::computeDynamicStripesColumn(int batch, int recursion, int act_x, int act_y, int kernel_x,
            int kernel_y, int init_channel, int stride, const cnpy::Array<T> &padded_act, int act_mask, int max_channel,
            bool lstm) {

        uint8_t max_bit = 0, min_bit = 16;
        for (int channel = init_channel; channel < std::min(init_channel + WEIGHT_LANES, max_channel); channel++) {

            // Computation cycles
            T act_bits;
            if(lstm)
                act_bits = padded_act.get(recursion, batch, channel);
            else
                act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x, stride * act_y + kernel_y);

            bool neg = false;
            if((act_bits & act_mask) != 0) {
                act_bits = act_bits & ~(uint16_t)act_mask;
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
        if(MINOR_BIT) cycles = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
        else cycles = max_bit + 1;
        return (uint8_t)cycles;

    }

    template <typename T>
    void DynamicStripes<T>::computeDynamicStripesTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int stride,
            const cnpy::Array<T> &padded_act, int act_mask, int max_channel, std::vector<uint32_t> &cycles_per_group,
            std::vector<uint32_t> &end_previous_pallet, sys::Statistics::Stats &stats) {

        int N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        int WINDOWS_PER_GROUP = N_COLUMNS / N_GROUPS;

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

            for (int channel = init_channel; channel < std::min(init_channel + WEIGHT_LANES, max_channel); channel++) {

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

                bool neg = false;
                if((act_bits & act_mask) != 0) {
                    act_bits = act_bits & ~(uint16_t)act_mask;
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
                if(MINOR_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 : max_bit - min_bit + 1);
                else per_group_cycles[group_index] = (uint8_t)(max_bit + 1);
            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(MINOR_BIT) per_group_cycles[group_index] = (uint8_t)((min_bit > max_bit) ? 1 : max_bit - min_bit + 1);
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
            stats.stall_cycles.back()[batch] += (end_previous_pallet[0] + 1) - fastest_column;

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
            stats.stall_cycles.back()[batch] += slowest_group - fastest_group;
        }

    }

    template <typename T>
    void DynamicStripes<T>::computeDynamicStripes2DTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int init_filter,
            int stride, const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int act_mask, int max_filter,
            std::vector<uint32_t> &cycles_per_group, std::vector<uint32_t> &end_previous_pallet,
            sys::Statistics::Stats &stats) {

        //Get the slowest column
        int N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        int WINDOWS_PER_GROUP = N_COLUMNS / N_GROUPS;

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

            for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {

                auto act_bits = padded_act.get(batch, filter + init_channel, stride * list_act_x[window] + kernel_x,
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

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }

            group_counter++;
            if(group_counter == WINDOWS_PER_GROUP) {
                if(MINOR_BIT) per_group_cycles[group_index] = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
                else per_group_cycles[group_index] = max_bit + 1;

            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(MINOR_BIT) per_group_cycles[group_index] = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
            else per_group_cycles[group_index] = max_bit + 1;

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
            stats.stall_cycles.back()[batch] += (end_previous_pallet[0] + 1) - fastest_column;

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
            stats.stall_cycles.back()[batch] += slowest_group - fastest_group;
        }

    }

    /* CYCLES */

    template <typename T>
    void DynamicStripes<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(!DIFFY) act.sign_magnitude_representation(layer.getAct_precision());
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

        auto act_prec = layer.getAct_precision();
        auto act_mask = (uint16_t)(1 << (act_prec - 1));

        auto wgt_layer_prec = layer.getWgt_precision();
        auto rows_per_wgt = (int)ceil(wgt_layer_prec / (double)BITS_PE);
        auto filters_per_tile = N_ROWS/rows_per_wgt;

        int groups = act_channels / wgt_channels;
        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)filters_per_tile/groups);
        auto baseline_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS/groups);

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_rows.push_back((uint64_t)(N_ROWS - filters_per_tile*rows_per_wgt));
        stats.rows_per_wgt.push_back((uint64_t)rows_per_wgt);

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n=0; n<batch_size; n++) {

            std::vector<int> list_x, list_y;
            int x_counter = 0, y_counter = 0;
            std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(COLUMN_REGISTERS, 0);
            std::vector<uint32_t> cycles_per_group = std::vector<uint32_t>(N_COLUMNS * 16 / PRECISION_GRANULARITY, 0);

            while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter, y_counter, N_COLUMNS)) {
                for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = 0; k < act_channels; k += WEIGHT_LANES) {
                            computeDynamicStripesTile(n, list_x, list_y, i, j, k, stride, act, act_mask, act_channels,
                                    cycles_per_group, end_previous_pallet, stats);
                        }
                    }
                }
            }
            auto batch_cycles = *std::max_element(cycles_per_group.begin(), cycles_per_group.end());
            stats.cycles.back()[n] = batch_cycles*num_filters_sets;
            stats.stall_cycles.back()[n] *= num_filters_sets;
        }

        auto base_cycles = (uint64_t)(out_x * out_y * ceil(act_channels/(double)WEIGHT_LANES) * Kx * Ky *
                baseline_filters_sets);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

    template <typename T>
    void DynamicStripes<T>::computeConvolution2D(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.sign_magnitude_representation(layer.getAct_precision());
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

        auto act_prec = layer.getAct_precision();
        auto act_mask = (uint16_t)(1 << (act_prec - 1));

        auto wgt_layer_prec = layer.getWgt_precision();
        auto rows_per_wgt = (int)ceil(wgt_layer_prec / (double)BITS_PE);
        auto filters_per_tile = N_ROWS/rows_per_wgt;

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_rows.push_back((uint64_t)(N_ROWS - filters_per_tile*rows_per_wgt));
        stats.rows_per_wgt.push_back((uint64_t)rows_per_wgt);

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n=0; n<batch_size; n++) {

            std::vector<int> list_x, list_y;
            int x_counter = 0, y_counter = 0;
            std::vector<uint32_t> end_previous_pallet = std::vector<uint32_t>(COLUMN_REGISTERS, 0);
            std::vector<uint32_t> cycles_per_col = std::vector<uint32_t>(N_COLUMNS, 0);

            for(int m=0; m<num_filters; m+=filters_per_tile) {
                while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            for (int k = 0; k < wgt_channels; k+=WEIGHT_LANES) {
                                computeDynamicStripes2DTile(n,list_x, list_y, i, j, k, m, stride, act, wgt, act_mask,
                                        num_filters, cycles_per_col, end_previous_pallet, stats);
                            }
                        }
                    }
                }
            }
            stats.cycles.back()[n] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
        }

        auto base_cycles = (uint64_t)(out_x * out_y * ceil(wgt_channels/(double)WEIGHT_LANES) * Kx * Ky *
                ceil(num_filters/(double)N_ROWS));

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

    template <typename T>
    void DynamicStripes<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.sign_magnitude_representation(layer.getAct_precision());

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

        auto act_prec = layer.getAct_precision();
        auto act_mask = (uint16_t)(1 << (act_prec - 1));

        auto wgt_layer_prec = layer.getWgt_precision();
        auto rows_per_wgt = (int)ceil(wgt_layer_prec / (double)BITS_PE);
        auto filters_per_tile = N_ROWS/rows_per_wgt;

        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)filters_per_tile);
        auto baseline_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS);

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.idle_rows.push_back((uint64_t)(N_ROWS - filters_per_tile*rows_per_wgt));
        stats.rows_per_wgt.push_back((uint64_t)rows_per_wgt);

        int n;

        #ifndef FC_MULTIPLEX_COLUMNS

        // All FC in one column
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n<batch_size; n++) {
            for (int r = 0; r < R; r++) {
                for (int k = 0; k<act_channels; k += WEIGHT_LANES) {
                    stats.cycles.back()[n] += computeDynamicStripesColumn(n,r,0,0,0,0,k,0,act,act_mask,act_channels,lstm);
                }
            }
            stats.cycles.back()[n] *= num_filters_sets;
        }

        #else

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n<batch_size; n++) {

            int column_index = 0;
            std::vector<int> column_end = std::vector<int>(N_COLUMNS, 0);
            for (int r = 0; r < R; r++) {
                for (int k = 0; k<act_channels; k += WEIGHT_LANES) {
                    if(stats.cycles.back()[n] < column_end[column_index])
                        stats.cycles.back()[n] = column_end[column_index];
                    auto column_cycles = computeDynamicStripesColumn(n,r,0,0,0,0,k,0,act,act_mask,act_channels,lstm);
                    column_end[column_index] = stats.cycles.back()[n] + column_cycles;
                    stats.cycles.back()[n]++;
                    column_index++;
                    if(column_index >= N_COLUMNS) column_index = 0;
                }
            }
            uint64_t last_column_end = *std::max_element(column_end.begin(), column_end.end());
            uint64_t last_column_rem_cycles = last_column_end - stats.cycles.back()[n];
            stats.cycles.back()[n] *= num_filters_sets;
            stats.cycles.back()[n] += last_column_rem_cycles;
        }

        #endif

        auto base_cycles = (uint64_t)(ceil(act_channels/(double)WEIGHT_LANES) * baseline_filters_sets * R);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

    template <typename T>
    void DynamicStripes<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        std::string arch = "DynamicStripes";
        arch += (DIFFY ? "_Diffy" : "");
        stats.arch = arch + "_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS) + "_PG" +
                std::to_string(PRECISION_GRANULARITY) + "_CR" + std::to_string(COLUMN_REGISTERS) + "_BP" +
                std::to_string(BITS_PE) + (MINOR_BIT ? "_MB" : "");

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(layer.getWgt_precision());
                if(layer.getWeights().getShape()[1] == 1 && layer.getActivations().getShape()[1] != 1)
                    computeConvolution2D(layer, stats);
                else
                    computeConvolution(layer, stats);
            } else if(layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(layer.getWgt_precision());
                computeInnerProduct(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);

    }

    /* POTENTIALS */

    template <typename T>
    void DynamicStripes<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
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
        auto layer_prec = layer.getAct_precision();

        // Convolution
        for(int n=0; n<batch_size; n++) {
            bit_counter = (uint64_t)computeDynamicStripesBitsPE((uint8_t)layer_prec,network_bits) * out_x * out_y * Kx *
                    Ky * wgt_channels * num_filters;
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
    void DynamicStripes<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
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
        auto layer_prec = layer.getAct_precision();

        for (int n = 0; n<batch_size; n++) {
            bit_counter = (uint64_t)computeDynamicStripesBitsPE((uint8_t)layer_prec,network_bits) * wgt_channels *
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
    void DynamicStripes<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "DynamicStripes";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(0);
                computePotentialsConvolution(layer,stats,network.getNetwork_bits());
            } else if (layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(0);
                computePotentialsInnerProduct(layer,stats,network.getNetwork_bits());
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* AVERAGE WIDTH */

    template <typename T>
    std::vector<double> DynamicStripes<T>::computeAvgWidthDynamicStripesActTile(int batch, int recursion,
            const std::vector<int> &list_act_x, const std::vector<int> &list_act_y, int kernel_x, int kernel_y,
            int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel, int act_mask, bool lstm) {

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

            for(int channel = init_channel; channel < std::min(init_channel + WEIGHT_LANES,max_channel); channel++) {

                T act_bits;
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
                if(MINOR_BIT) act_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                else act_width[group_index] = max_bit + 1;
            }

        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(MINOR_BIT) act_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
            else act_width[group_index] = max_bit + 1;
        }

        return act_width;

    }

    template <typename T>
    std::vector<double> DynamicStripes<T>::computeAvgWidthDynamicStripesWgtTile(int kernel_x, int kernel_y,
            int init_channel, int init_filter, const cnpy::Array<T> &wgt, int max_channel, int max_filter,
            int wgt_mask) {

        int N_GROUPS = N_COLUMNS * 16 / PRECISION_GRANULARITY;
        int WINDOWS_PER_GROUP = N_COLUMNS / N_GROUPS;

        // Weights
        std::vector<double> wgt_width = std::vector<double>(std::min(N_GROUPS,max_filter-init_filter), 0);
        uint16_t group_counter = 0;
        uint16_t group_index = 0;
        uint8_t max_bit = 0, min_bit = 16;
        for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {

            if(group_counter == WINDOWS_PER_GROUP)  {
                max_bit = 0, min_bit = 16;
                group_counter = 0;
                group_index++;
            }

            for(int channel = init_channel; channel < std::min(init_channel + WEIGHT_LANES,max_channel); channel++) {

                auto wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

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
                if(MINOR_BIT) wgt_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                else wgt_width[group_index] = max_bit + 1;
            }
        }

        if(group_counter < WINDOWS_PER_GROUP) {
            if(MINOR_BIT) wgt_width[group_index] = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
            else wgt_width[group_index] = max_bit + 1;
        }

        return wgt_width;

    }

    template <typename T>
    void DynamicStripes<T>::computeAvgWidthLayer(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.sign_magnitude_representation(layer.getAct_precision());
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.sign_magnitude_representation(layer.getWgt_precision());
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        int padding = layer.getPadding();
        int stride = layer.getStride();

        if(layer.getType() == "InnerProduct") {
            if(act.getDimensions() == 4) act.reshape_to_2D();
            act.reshape_to_4D();
        }

        if(layer.getType() == "Convolution")
            act.zero_pad(padding);

        if(act.getShape()[1] == 3 && stride > 1) {
            act.reshape_first_layer_act((uint16_t)stride);
            wgt.reshape_first_layer_wgt((uint16_t)stride);
            stride = 1;
        }

        bool lstm = layer.getType() == "LSTM";

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size, act_channels, Nx, Ny, R;
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

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        long out_x = (Nx - Kx)/stride + 1;
        long out_y = (Ny - Ky)/stride + 1;

        auto act_prec = layer.getAct_precision();
        auto act_mask = (uint16_t)(1 << (act_prec - 1));

        auto wgt_prec = layer.getWgt_precision();
        auto wgt_mask = (uint16_t)(1 << (wgt_prec - 1));

        // Stats
        stats.act_avg_width.emplace_back(std::vector<double>(batch_size,0));
        stats.act_width_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.act_bits_baseline.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_bits_profiled.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_bits_datawidth.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.act_bits_scnn.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.wgt_avg_width.emplace_back(std::vector<double>(batch_size,0));
        stats.wgt_width_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.wgt_bits_baseline.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.wgt_bits_profiled.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.wgt_bits_datawidth.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.wgt_bits_scnn.emplace_back(std::vector<uint64_t>(batch_size,0));

        for(int i = 0; i <= 16; i++) {
            stats.act_width_need[i].emplace_back(std::vector<double>(batch_size, 0));
            stats.wgt_width_need[i].emplace_back(std::vector<double>(batch_size, 0));
        }

        // Activations
        for(int n=0; n<batch_size; n++) {

            std::vector<int> list_x, list_y;
            int x_counter = 0, y_counter = 0;
            std::vector<double> act_width;

            for(int r = 0; r < R; r++) {
                while (this->iterateWindows(out_x, out_y, list_x, list_y, x_counter, y_counter, N_COLUMNS)) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            for (int k = 0; k < act_channels; k += WEIGHT_LANES) {
                                auto tile_act_width = computeAvgWidthDynamicStripesActTile(n, r, list_x, list_y, i, j,
                                        k, stride, act, act_channels, act_mask, lstm);
                                act_width.insert(act_width.end(), tile_act_width.begin(), tile_act_width.end());
                            }
                        }
                    }
                }
            }

            double act_avg_width = stats.get_average(act_width);

            // Calculate bits needed
            std::vector<uint64_t> act_width_need (network_bits + 1, 0);
            std::vector<double> act_width_need_per (network_bits + 1 ,0);
            for(auto act_group : act_width)
                for(int a = (int)act_group; a <= network_bits; a++)
                    act_width_need[a]++;
            for(int a = 0; a < act_width_need.size(); a++)
                act_width_need_per[a] = act_width_need[a] / (double)act_width.size() * 100.;

            stats.act_avg_width.back()[n] = act_avg_width;
            stats.act_width_reduction.back()[n] = (act_prec - act_avg_width) * 100. / act_prec;

            for(int i = 0; i <= network_bits; i++)
                stats.act_width_need[i].back()[n] = act_width_need_per[i];

        }

        for(int n=0; n<batch_size; n++) {

            uint64_t act_bits_datawidth = 0;
            for(int r = 0; r < R; r++) {
                for (int k = 0; k < act_channels; k += WEIGHT_LANES) {
                    for (int j = 0; j < Ny; j++) {
                        for (int i = 0; i < Nx; i++) {
                            uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                            for(int channel = k; channel < std::min(k + WEIGHT_LANES,act_channels); channel++) {
                                T act_bits;
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
                            if(MINOR_BIT) width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                            else width = max_bit + 1;
                            act_bits_datawidth = act_bits_datawidth + (width * non_zeroes);
                        }
                    }
                }
            }

            // Calculate data from off-chip
            auto num_act = R * Nx * Ny * act_channels;
            stats.act_bits_baseline.back()[n] = num_act * network_bits;
            stats.act_bits_profiled.back()[n] = 4 + num_act * act_prec;
            auto overhead = (uint64_t)((16 + log2(network_bits)) * ceil(num_act / 16.));
            stats.act_bits_datawidth.back()[n] = overhead + act_bits_datawidth;

        }

        // Weights
        std::vector<double> wgt_width;
        for(int m=0; m<num_filters; m+=N_ROWS) {

            for (int i = 0; i < Kx; i++) {
                for (int j = 0; j < Ky; j++) {
                    for (int k = 0; k < wgt_channels; k+=WEIGHT_LANES) {
                        auto tile_wgt_width = computeAvgWidthDynamicStripesWgtTile(i,j,k,m,wgt,wgt_channels,num_filters,
                                wgt_mask);
                        wgt_width.insert(wgt_width.end(),tile_wgt_width.begin(),tile_wgt_width.end());

                    }
                }

            }
        }

        uint64_t wgt_bits_datawidth = 0;
        for(int m=0; m<num_filters; m++) {
            for (int k = 0; k < wgt_channels; k+=WEIGHT_LANES) {
                for (int j = 0; j < Ky; j++) {
                    for (int i = 0; i < Kx; i++) {
                        uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                        for(int channel = k; channel < std::min(k + WEIGHT_LANES,wgt_channels); channel++) {

                            auto wgt_bits = wgt.get(m, channel, i, j);

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
                        if(MINOR_BIT) width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                        else width = max_bit + 1;
                        wgt_bits_datawidth = wgt_bits_datawidth + (width * non_zeroes);
                    }
                }

            }
        }

        double wgt_avg_width = stats.get_average(wgt_width);

        // Calculate bits needed
        std::vector<uint64_t> wgt_width_need (network_bits + 1, 0);
        std::vector<double> wgt_width_need_per (network_bits + 1, 0);
        for(auto wgt_group : wgt_width)
            for(int w = (int)wgt_group; w <= network_bits; w++)
                wgt_width_need[w]++;
        for(int w = 0; w < wgt_width_need.size(); w++)
            wgt_width_need_per[w] = wgt_width_need[w] / (double)wgt_width.size() * 100.;

        for(int n=0; n<batch_size; n++) {

            // Calculate data from off-chip
            auto num_wgt = wgt.getMax_index();
            stats.wgt_bits_baseline.back()[n] = num_wgt * network_bits;
            stats.wgt_bits_profiled.back()[n] = 4 + num_wgt * wgt_prec;
            auto overhead = (uint64_t)((16 + log2(network_bits)) * ceil(num_wgt / 16.));
            stats.wgt_bits_datawidth.back()[n] = overhead + wgt_bits_datawidth;

            stats.wgt_avg_width.back()[n] = wgt_avg_width;
            stats.wgt_width_reduction.back()[n] = (wgt_prec - wgt_avg_width) * 100. / wgt_prec;

            for (int i = 0; i <= network_bits; i++) {
                stats.wgt_width_need[i].back()[n] = wgt_width_need_per[i];
            }
        }

        // SCNN
        for(int n=0; n < batch_size; n++) {

            int skips = 0;
            uint64_t act_bits_scnn = 0;
            for(int r = 0; r < R; r++) {
                for (int k = 0; k < act_channels; k++) {
                    for (int y = 0; y < Ny; y++) {
                        for (int x = 0; x < Nx; x++) {
                            T act_bits;
                            if(lstm) act_bits = act.get(r, n, k);
                            else act_bits = act.get(n, k, x, y);
                            if (act_bits != 0) {
                                act_bits_scnn += network_bits + 4;
                                skips = 0;
                            } else {
                                skips++;
                                if (skips == 16) {
                                    act_bits_scnn += network_bits + 4;
                                    skips = 0;
                                }
                            }
                        }
                    }
                }
            }

            stats.act_bits_scnn.back()[n] = act_bits_scnn;
        }

        int skips = 0;
        uint64_t wgt_bits_scnn = 0;
        for(int m=0; m < num_filters; m++) {

            for (int k = 0; k < wgt_channels; k++) {
                for (int y = 0; y < Ky; y++) {
                    for (int x = 0; x < Kx; x++) {
                        auto act_bits = wgt.get(m, k, x, y);
                        if(act_bits != 0) {
                            wgt_bits_scnn += network_bits + 4;
                            skips = 0;
                        } else {
                            skips++;
                            if(skips == 16) {
                                wgt_bits_scnn += network_bits + 4;
                                skips = 0;
                            }
                        }
                    }
                }
            }
        }

        for(int n = 0; n < batch_size; n++)
            stats.wgt_bits_scnn.back()[n] = wgt_bits_scnn;

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void DynamicStripes<T>::average_width(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "average_width";
        stats.net_name = network.getName();
        stats.arch = "DynamicStripes_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS) + "_PG" +
                std::to_string(PRECISION_GRANULARITY);

        stats.act_width_need = std::vector<std::vector<std::vector<double>>>(16 + 1);
        stats.wgt_width_need = std::vector<std::vector<std::vector<double>>>(16 + 1);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution" || layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(layer.getWgt_precision());
                computeAvgWidthLayer(layer, stats, network.getNetwork_bits());
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class DynamicStripes<uint16_t>;

}
