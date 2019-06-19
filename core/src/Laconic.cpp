
#include <core/Laconic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t Laconic<T>::computeLaconicPE(uint16_t act, uint16_t wgt) {

        uint16_t act_bits = act;
        uint16_t wgt_bits = wgt;

        #ifdef BOOTH_ENCODING
        act_bits = this->booth_encoding(act_bits);
        wgt_bits = this->booth_encoding(wgt_bits);
        #endif

        uint8_t act_effectual_bits = this->effectualBits(act_bits);
        uint8_t wgt_effectual_bits = this->effectualBits(wgt_bits);

        uint8_t bit_multiplications = act_effectual_bits * wgt_effectual_bits;
        #ifdef ZERO_COUNT
        if(bit_multiplications == 0) bit_multiplications = 1;
        #endif

        return bit_multiplications;
    }

    template <typename T>
    uint8_t Laconic<T>::computeLaconicColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
            int init_channel, int init_filter, int stride, const cnpy::Array<T> &padded_act,
            const cnpy::Array<T> &wgt, int start_group, int max_channel, int max_filter, bool lstm, bool conv2D) {

        //Get the slowest PE
        std::vector<uint8_t> cycles;
        for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {
            for(int channel = init_channel; channel < std::min(init_channel + N_LANES,max_channel); channel++) {

                // Fix for MobileNet
                if(conv2D)
                    start_group = filter;

                T act_bits;
                if(lstm)
                    act_bits = padded_act.get(recursion, batch, channel);
                else
                    act_bits = padded_act.get(batch, start_group + channel, stride * act_x + kernel_x,
                            stride * act_y + kernel_y);

                auto wgt_bits = wgt.get(filter, channel, kernel_x, kernel_y);

                uint8_t PE_cycles = computeLaconicPE(act_bits, wgt_bits);
                cycles.push_back(PE_cycles);
            }
        }

        return *std::max_element(cycles.begin(), cycles.end());

    }

    template <typename T>
    uint8_t Laconic<T>::computeLaconicTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_channel, int init_filter,
            int stride, const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int start_group, int max_channel,
            int max_filter, bool conv2D, sys::Statistics::Stats &stats) {

        //Get the slowest column
        std::vector<uint8_t> cycles;
        for(int window = 0; window < list_act_x.size(); window++) {
            uint8_t PE_cycles = computeLaconicColumn(batch,0,list_act_x[window],list_act_y[window],kernel_x,kernel_y,
                    init_channel,init_filter,stride,padded_act,wgt,start_group,max_channel,max_filter,false,conv2D);
            cycles.push_back(PE_cycles);
        }

        auto slowest_column = *std::max_element(cycles.begin(), cycles.end());
        auto fastest_column = *std::min_element(cycles.begin(), cycles.end());
        stats.stall_cycles.back()[batch] += slowest_column - fastest_column;
        return slowest_column;
    }

    /* CYCLES */

    template <typename T>
    void Laconic<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation(layer.getWgtPrecision());
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

        auto batch_size = act_shape[0];
        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        long out_x = (Nx - Kx)/stride + 1;
        long out_y = (Ny - Ky)/stride + 1;

        auto groups = act_channels / wgt_channels;
        auto it_per_group = num_filters / groups;

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

                int start_group = 0;
                if(m >= it_per_group)
                    start_group = (int)wgt_channels;

                bool conv2D = false;
                if(wgt_channels == 1 && act_channels != 1)
                    conv2D = true;

                while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            for (int k = 0; k < wgt_channels; k += N_LANES) {
                                cycles += computeLaconicTile(n,list_x, list_y, i, j, k, m, stride, act,
                                        wgt, start_group, (int)wgt_channels, (int)num_filters, conv2D, stats);

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
    void Laconic<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation(layer.getWgtPrecision());
        wgt.reshape_to_4D();

        if(layer.getType() == "InnerProduct") {
            if (act.getDimensions() == 4) act.reshape_to_2D();
            act.reshape_to_4D();
        }

        bool lstm = layer.getType() == "LSTM";

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        uint64_t batch_size, act_channels, R;
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

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];

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
                        cycles += computeLaconicColumn(n,r,0,0,0,0,k,m,0,act,wgt,0,wgt_channels,num_filters,lstm);
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
            std::vector<uint64_t>column_end = std::vector<uint64_t>(N_COLUMNS, 0);
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
                        auto column_cycles = computeLaconicColumn(n,r,0,0,0,0,k,m,0,act,wgt,0,(int)wgt_channels,
                                (int)num_filters,lstm, false);
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
            stats.scheduled_pe.back()[n] = (uint64_t)(num_filters * N_ROWS * ceil(act_channels/(double)N_LANES));
            auto idle_rows = N_ROWS - (num_filters % N_ROWS);
            idle_rows = idle_rows == 16 ? 0 : idle_rows;
            stats.idle_pe.back()[n] = (uint64_t)(idle_rows * ceil(act_channels/(double)N_LANES));

        }

        #endif

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void Laconic<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        stats.arch = "Laconic_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                computeConvolution(layer, stats);
            } else if(layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                computeInnerProduct(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* POTENTIALS */

    template <typename T>
    void Laconic<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation(layer.getWgtPrecision());
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        auto batch_size = act_shape[0];
        auto act_channels = act_shape[1];
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        auto groups = act_channels / wgt_channels;
        auto it_per_group = num_filters / groups;

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
        for(n=0; n<batch_size; n++) {
            uint64_t bit_counter = 0;
            for(int m=0; m<num_filters; m++) {

                // Two towers alexnet
                int start_group = 0;
                if(m >= it_per_group)
                    start_group = (int)wgt_channels;

                // Fix for MobileNet
                if(wgt_channels == 1 && act_channels != 1)
                    start_group = m;

                for(int x = 0; x < out_x; x++) {
                    for(int y = 0; y < out_y; y++) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = 0; k < wgt_channels; k++) {
                                    bit_counter += computeLaconicPE(act.get(n, start_group + k, stride * x + i,
                                            stride * y + j), wgt.get(m, k, i, j));
                                }
                            }
                        }
                    }
                }
            }
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
    void Laconic<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation(layer.getActPrecision());
        if(act.getDimensions() == 4) act.reshape_to_2D();
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation(layer.getWgtPrecision());

        bool lstm = layer.getType() == "LSTM";

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        auto batch_size = act_shape[0];
        auto R = lstm ? act_shape[0] : 1;

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
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
            for (int r = 0; r < R; r++) {
                for (int m = 0; m < num_filters; m++) {
                    for (int k = 0; k < wgt_channels; k++) {
                        auto act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                        bit_counter += computeLaconicPE(act_bits, wgt.get(m, k));
                    }
                }
            }
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
    void Laconic<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "Laconic";

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

    template class Laconic<uint16_t>;

}