
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
    uint8_t Laconic<T>::computeLaconicColumn(int batch, int act_x, int act_y, int kernel_x, int kernel_y,
            int init_channel, int init_filter, int stride, const cnpy::Array<T> &padded_act,
            const cnpy::Array<T> &wgt, int start_group, int max_channel, int max_filter) {

        //Get the slowest PE
        std::vector<uint8_t> cycles;
        for (int filter = init_filter; filter < std::min(init_filter + N_ROWS, max_filter); filter++) {
            for(int channel = init_channel; channel < std::min(init_channel + WEIGHT_LANES,max_channel); channel++) {

                auto act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x,
                        stride * act_y + kernel_y);
                auto wgt_bits = wgt.get(filter, channel - start_group, kernel_x, kernel_y);

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
            int max_filter) {

        //Get the slowest column
        std::vector<uint8_t> cycles;
        for(int window = 0; window < list_act_x.size(); window++) {
            uint8_t PE_cycles = computeLaconicColumn(batch,list_act_x[window],list_act_y[window],kernel_x,kernel_y,
                    init_channel,init_filter,stride,padded_act,wgt,start_group,max_channel,max_filter);
            cycles.push_back(PE_cycles);
        }

        return *std::max_element(cycles.begin(), cycles.end());

    }

    /* CYCLES */

    template <typename T>
    void Laconic<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation();
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

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n=0; n<batch_size; n++) {

            std::vector<int> list_x, list_y;
            int current_group = 0, group_m = 0, start_group = 0;
            int x_counter = 0, y_counter = 0;

            current_group = 0, group_m = 0, start_group = 0, x_counter = 0, y_counter = 0;
            for(int m=0; m<num_filters; m+=N_ROWS) {
                while(this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,N_COLUMNS)) {
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            for (int k = start_group; k < wgt_channels + start_group; k+=WEIGHT_LANES) {
                                stats.cycles.back()[n] += computeLaconicTile(n,list_x, list_y, i, j, k, m, stride, act,
                                        wgt, start_group, act_channels, num_filters);
                            }
                        }
                    }
                }
                group_m += N_ROWS;
                if(group_m >= it_per_group) {
                    group_m = 0;
                    current_group++;
                    start_group = wgt_channels*current_group;
                }
            }
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void Laconic<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        act.reshape_to_4D();
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation();
        wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        if(this->FAST_MODE) batch_size = 1;

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));

        int n;

        #ifndef FC_MULTIPLEX_COLUMNS

        // All FC in one column
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n<batch_size; n++) {
            for (int m = 0; m<num_filters; m+=N_ROWS) {
                for (int k = 0; k<wgt_channels; k+=WEIGHT_LANES) {
                    stats.cycles.back()[n] += computeLaconicColumn(n,0,0,0,0,k,m,0,act,wgt,0,wgt_channels,num_filters);
                }
            }
        }

        #else

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n<batch_size; n++) {

            int column_index = 0;
            std::vector<int>column_end = std::vector<int>(N_COLUMNS, 0);

            for (int m = 0; m<num_filters; m+=N_ROWS) {
                for (int k = 0; k<wgt_channels; k+=WEIGHT_LANES) {
                    if(stats.cycles.back()[n] < column_end[column_index])
                        stats.cycles.back()[n] = column_end[column_index];
                    auto column_cycles = computeLaconicColumn(n,0,0,0,0,k,m,0,act,wgt,0,wgt_channels,num_filters);
                    column_end[column_index] = stats.cycles.back()[n] + column_cycles;
                    stats.cycles.back()[n]++;
                    column_index++;
                    if(column_index >= N_COLUMNS) column_index = 0;
                }
            }
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
            } else if(layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                computeInnerProduct(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* POTENTIALS */

    template <typename T>
    void Laconic<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation();
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
        int it_per_group = num_filters / groups;

        // Operations
        const auto parallel_mult = (uint64_t)(num_filters * out_x * out_y * Kx * Ky * wgt_channels);
        std::vector<uint64_t> bit_multiplications (batch_size,0);
        std::vector<double> work_reduction (batch_size,0);
        std::vector<double> speedup (batch_size,0);

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n=0; n<batch_size; n++) {

            int current_group = 0, group_m = 0, start_group = 0;
            uint64_t bit_counter = 0;

            for(int m=0; m<num_filters; m++) {
                for(int x=0; x<out_x; x++) {
                    for(int y=0; y<out_y; y++) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = start_group; k < wgt_channels + start_group; k++) {
                                    bit_counter += computeLaconicPE(act.get(n, k, stride * x + i,stride * y + j),
                                            wgt.get(m, k - start_group, i, j));
                                }
                            }
                        }
                    }
                }
                group_m++;
                if(group_m >= it_per_group) {
                    group_m = 0;
                    current_group++;
                    start_group = wgt_channels*current_group;
                }
            }
            work_reduction[n] = 100 - ((double)bit_counter / (double)parallel_mult / 256. * 100);
            speedup[n] = (double)parallel_mult * 256. / (double)bit_counter;
            bit_multiplications[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.work_reduction.push_back(work_reduction);
        stats.speedup.push_back(speedup);
        stats.parallel_multiplications.push_back(parallel_mult);
        stats.bit_multiplications.push_back(bit_multiplications);

    }

    template <typename T>
    void Laconic<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        cnpy::Array<T> wgt = layer.getWeights();
        wgt.powers_of_two_representation();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        if(this->FAST_MODE) batch_size = 1;

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels;
        std::vector<uint64_t> bit_multiplications (batch_size,0);
        std::vector<double> work_reduction (batch_size,0);
        std::vector<double> speedup (batch_size,0);

        int n;

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n<batch_size; n++) {
            uint64_t bit_counter = 0;
            for (int m = 0; m<num_filters; m++) {
                for (int k = 0; k<wgt_channels; k++) {
                    bit_counter += computeLaconicPE(act.get(n, k), wgt.get(m, k));
                }
            }
            work_reduction[n] = 100 - ((double) bit_counter / (double) parallel_mult / 256. * 100);
            speedup[n] = (double)parallel_mult * 256. / (double)bit_counter;
            bit_multiplications[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.work_reduction.push_back(work_reduction);
        stats.speedup.push_back(speedup);
        stats.parallel_multiplications.push_back(parallel_mult);
        stats.bit_multiplications.push_back(bit_multiplications);

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
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(std::get<0>(layer.getWgt_precision()) + std::get<1>(layer.getWgt_precision()));
                computePotentialsConvolution(layer,stats);
            } else if (layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(std::get<0>(layer.getWgt_precision()) + std::get<1>(layer.getWgt_precision()));
                computePotentialsInnerProduct(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class Laconic<uint16_t>;

}