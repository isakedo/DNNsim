
#include <core/DynamicStripes.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t DynamicStripes<T>::computeDynamicStripesBitsPE(uint8_t layer_prec) {
        return layer_prec * (uint8_t)16;
    }

    template <typename T>
    uint8_t DynamicStripes<T>::computeDynamicStripesColumn(int batch, int act_x, int act_y, int kernel_x, int kernel_y,
            int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel) {

        uint8_t max_bit = 0, min_bit = 16;
        for (int channel = init_channel; channel < std::min(init_channel + WEIGHT_LANES, max_channel); channel++) {

            // Computation cycles
            uint16_t act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x, stride * act_y + kernel_y);

            const auto &min_max_act_bits = this->minMax(act_bits);

            auto min_act_bit = std::get<0>(min_max_act_bits);
            auto max_act_bit = std::get<1>(min_max_act_bits);

            if(min_act_bit < min_bit) min_bit = min_act_bit;
            if(max_act_bit > max_bit) max_bit = max_act_bit;

        }

        return min_bit > max_bit ? (uint8_t)1 : max_bit - min_bit + (uint8_t)1;

    }

    template <typename T>
    void DynamicStripes<T>::computeDynamicStripesTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int stride,
            const cnpy::Array<T> &padded_act, int max_channel, std::vector<uint32_t> &cycles_per_col,
            std::vector<uint32_t> &end_previous_pallet, sys::Statistics::Stats &stats) {

        std::vector<uint8_t> per_SIP_n_bits (N_COLUMNS, 0);
        uint8_t max_bit = 0, min_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {
            if(PRECISION_GRANULARITY == "SIP") max_bit = 0, min_bit = 16;
            for (int channel = init_channel; channel < std::min(init_channel + WEIGHT_LANES, max_channel); channel++) {

                // Computation cycles
                uint16_t act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                        stride * list_act_y[window] + kernel_y);

                const auto &min_max_act_bits = this->minMax(act_bits);

                auto min_act_bit = std::get<0>(min_max_act_bits);
                auto max_act_bit = std::get<1>(min_max_act_bits);

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }
            per_SIP_n_bits[window] = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
        }

        if(PRECISION_GRANULARITY == "Tile") {
            uint8_t n_bits = min_bit > max_bit ? (uint8_t)1 : max_bit - min_bit + (uint8_t)1;
            cycles_per_col = std::vector<uint32_t>(N_COLUMNS,cycles_per_col[0] + n_bits);
        } else {

            for(int window = 0; window < list_act_x.size(); window++) {
                cycles_per_col[window] += per_SIP_n_bits[window];
            }

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
                end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_col.begin(),
                        cycles_per_col.end());
            } else {
                auto slowest_column = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
                auto fastest_column = *std::min_element(cycles_per_col.begin(), cycles_per_col.end());
                cycles_per_col = std::vector<uint32_t>(N_COLUMNS, slowest_column);
                stats.stall_cycles.back()[batch] += slowest_column - fastest_column;
            }
        }

    }

    template <typename T>
    void DynamicStripes<T>::computeDynamicStripes2DTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int init_filter,
            int stride, const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int max_filter,
            std::vector<uint32_t> &cycles_per_col, std::vector<uint32_t> &end_previous_pallet,
            sys::Statistics::Stats &stats) {

        //Get the slowest column
        std::vector<uint8_t> per_SIP_n_bits (N_COLUMNS, 0);
        uint8_t max_bit = 0, min_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {
            if(PRECISION_GRANULARITY == "SIP") max_bit = 0, min_bit = 16;
            for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {

                std::vector<std::queue<uint8_t>> offsets;
                auto act_bits = padded_act.get(batch, filter + init_channel, stride * list_act_x[window] + kernel_x,
                        stride * list_act_y[window] + kernel_y);

                const auto &min_max_act_bits = this->minMax(act_bits);

                auto min_act_bit = std::get<0>(min_max_act_bits);
                auto max_act_bit = std::get<1>(min_max_act_bits);

                if(min_act_bit < min_bit) min_bit = min_act_bit;
                if(max_act_bit > max_bit) max_bit = max_act_bit;

            }
            per_SIP_n_bits[window] = (min_bit > max_bit) ? 1 : max_bit - min_bit + 1;
        }

        if(PRECISION_GRANULARITY == "Tile") {
            uint8_t n_bits = min_bit > max_bit ? (uint8_t)1 : max_bit - min_bit + (uint8_t)1;
            cycles_per_col = std::vector<uint32_t>(N_COLUMNS,cycles_per_col[0] + n_bits);
        } else {

            for(int window = 0; window < list_act_x.size(); window++) {
                cycles_per_col[window] += per_SIP_n_bits[window];
            }

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
                end_previous_pallet[COLUMN_REGISTERS - 1] = *std::max_element(cycles_per_col.begin(),
                                                                              cycles_per_col.end());
            } else {
                auto slowest_column = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
                auto fastest_column = *std::min_element(cycles_per_col.begin(), cycles_per_col.end());
                cycles_per_col = std::vector<uint32_t>(N_COLUMNS, slowest_column);
                stats.stall_cycles.back()[batch] += slowest_column - fastest_column;
            }
        }

    }

    /* CYCLES */

    template <typename T>
    void DynamicStripes<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
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

        int groups = act_channels / wgt_channels;
        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS/groups);
        auto baseline_filters_sets = (uint32_t)ceil(num_filters/(double)16./groups);

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));

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

            while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter, y_counter, N_COLUMNS)) {
                for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = 0; k < act_channels; k += WEIGHT_LANES) {
                            computeDynamicStripesTile(n, list_x, list_y, i, j, k, stride, act, act_channels,
                                    cycles_per_col, end_previous_pallet, stats);
                        }
                    }
                }
            }
            auto batch_cycles = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
            stats.cycles.back()[n] = batch_cycles*num_filters_sets;
        }

        auto base_cycles = (uint64_t)(out_x * out_y * ceil(act_channels/16.) * Kx * Ky * baseline_filters_sets);

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

    template <typename T>
    void DynamicStripes<T>::computeConvolution2D(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
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

            for(int m=0; m<num_filters; m+=N_ROWS) {
                while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            for (int k = 0; k < wgt_channels; k+=WEIGHT_LANES) {
                                computeDynamicStripes2DTile(n,list_x, list_y, i, j, k, m, stride, act, wgt, num_filters,
                                        cycles_per_col, end_previous_pallet, stats);
                            }
                        }
                    }
                }
            }
            stats.cycles.back()[n] = *std::max_element(cycles_per_col.begin(), cycles_per_col.end());
        }

        auto base_cycles = (uint64_t)(out_x * out_y * ceil(wgt_channels/16.) * Kx * Ky * ceil(num_filters/16.));

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.baseline_cycles.push_back(base_cycles);

    }

    template <typename T>
    void DynamicStripes<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        act.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = layer.getWeights().getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        if(this->FAST_MODE) batch_size = 1;

        int num_filters = wgt_shape[0];

        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS);
        auto baseline_filters_sets = (uint32_t)ceil(num_filters/16.);

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.stall_cycles.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        #ifndef FC_MULTIPLEX_COLUMNS

        // All FC in one column
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n<batch_size; n++) {
            for (int k = 0; k<act_channels; k += WEIGHT_LANES) {
                stats.cycles.back()[n] += computeDynamicStripesColumn(n,0,0,0,0,k,0,act,act_channels,rowMap);
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

            for (int k = 0; k<act_channels; k += WEIGHT_LANES) {
                if(stats.cycles.back()[n] < column_end[column_index])
                    stats.cycles.back()[n] = column_end[column_index];
                auto column_cycles = computeDynamicStripesColumn(n,0,0,0,0,k,0,act,act_channels);
                column_end[column_index] = stats.cycles.back()[n] + column_cycles;
                stats.cycles.back()[n]++;
                column_index++;
                if(column_index >= N_COLUMNS) column_index = 0;
            }
            uint64_t last_column_end = *std::max_element(column_end.begin(), column_end.end());
            uint64_t last_column_rem_cycles = last_column_end - stats.cycles.back()[n];
            stats.cycles.back()[n] *= num_filters_sets;
            stats.cycles.back()[n] += last_column_rem_cycles;
        }

        #endif

        auto base_cycles = (uint64_t)(ceil(act_channels/16.) * baseline_filters_sets);

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
        stats.arch = "DynamicStripes_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS) + "_PG_" +
                PRECISION_GRANULARITY + "_CR" + std::to_string(COLUMN_REGISTERS);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                if(layer.getWeights().getShape()[1] == 1)
                    computeConvolution2D(layer, stats);
                else
                    computeConvolution(layer, stats);
            } else if(layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                computeInnerProduct(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);

    }

    /* POTENTIALS */

    template <typename T>
    void DynamicStripes<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

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
            bit_counter = (uint64_t)computeDynamicStripesBitsPE((uint8_t)layer_prec) * out_x * out_y * Kx * Ky *
                    wgt_channels * num_filters;
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / 256. * 100);
            stats.speedup.back()[n] = (double)parallel_mult * 256. / (double)bit_counter;
            stats.bit_multiplications.back()[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.parallel_multiplications.push_back(parallel_mult);

    }

    template <typename T>
    void DynamicStripes<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        const cnpy::Array<T> &wgt = layer.getWeights();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = 1;
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels;
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));
        uint64_t bit_counter = 0;

        // Get layer precision
        auto layer_prec = layer.getAct_precision();

        for (int n = 0; n<batch_size; n++) {
            bit_counter = (uint64_t)computeDynamicStripesBitsPE((uint8_t)layer_prec)*wgt_channels*num_filters;
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / 256. * 100);
            stats.speedup.back()[n] = (double)parallel_mult * 256. / (double)bit_counter;
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
                computePotentialsConvolution(layer,stats);
            } else if (layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(0);
                computePotentialsInnerProduct(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class DynamicStripes<uint16_t>;

}
