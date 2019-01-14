
#include <core/BitTacticalP.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t BitTacticalP<T>::computeTacticalPBitsPE(uint16_t wgt, uint8_t act_layer_prec) {
        return wgt == 0 ? (uint8_t)0 : act_layer_prec * (uint8_t)16;
    }

    template <typename T>
    uint8_t BitTacticalP<T>::computeTacticalPColumn(int batch, int act_x, int act_y, int init_filter, int stride,
            const cnpy::Array<T> &padded_act, int max_filter, schedule &dense_schedule) {

        uint8_t max_bit = 0, min_bit = 16;
        for (int filter = init_filter; filter < std::min(init_filter + this->N_ROWS, max_filter); filter++) {
            for (int wgt_idx = 0; wgt_idx < WEIGHT_LANES; wgt_idx++) {

                if(dense_schedule[init_filter].empty()) continue;
                auto wgt_tuple = dense_schedule[init_filter].front()[wgt_idx];
                int channel = std::get<0>(wgt_tuple);
                int kernel_x = std::get<1>(wgt_tuple);
                int kernel_y = std::get<2>(wgt_tuple);

                // Computation cycles
                uint16_t act_bits = padded_act.get(batch, channel, stride * act_x + kernel_x,
                        stride * act_y + kernel_y);

                uint8_t count = 0;
                std::vector<uint8_t> act_offsets;
                while (act_bits) {
                    auto current_bit = act_bits & 1;
                    if (current_bit) act_offsets.push_back(count);
                    act_bits >>= 1;
                    count++;
                }

                auto max_act_bit = act_offsets.empty() ? 0 : *std::max_element(act_offsets.begin(), act_offsets.end());
                auto min_act_bit = act_offsets.empty() ? 16 : *std::min_element(act_offsets.begin(), act_offsets.end());

                if (max_act_bit > max_bit) max_bit = max_act_bit;
                if (min_act_bit < min_bit) min_bit = min_act_bit;

            }
        }

        uint8_t n_bits = min_bit > max_bit ? (uint8_t)1 : max_bit - min_bit + (uint8_t)1;
        return n_bits;


    }

    template <typename T>
    uint8_t BitTacticalP<T>::computeTacticalPTile(int batch, const std::vector<int> &list_act_x,
            const std::vector<int> &list_act_y, int init_filter, int stride, const cnpy::Array<T> &padded_act,
            int max_filter, schedule &dense_schedule) {

        std::vector<uint8_t> per_SIP_n_bits (list_act_x.size(), 0);
        uint8_t max_bit = 0, min_bit = 16;
        for(int window = 0; window < list_act_x.size(); window++) {
            if(PRECISION_GRANULARITY == "SIP") max_bit = 0, min_bit = 16;
            for (int filter = init_filter; filter < std::min(init_filter + this->N_ROWS, max_filter); filter++) {
                for (int wgt_idx = 0; wgt_idx < WEIGHT_LANES; wgt_idx++) {

                    if(dense_schedule[init_filter].empty()) continue;
                    auto wgt_tuple = dense_schedule[init_filter].front()[wgt_idx];
                    int channel = std::get<0>(wgt_tuple);
                    int kernel_x = std::get<1>(wgt_tuple);
                    int kernel_y = std::get<2>(wgt_tuple);

                    // Computation cycles
                    uint16_t act_bits = padded_act.get(batch, channel, stride * list_act_x[window] + kernel_x,
                            stride * list_act_y[window] + kernel_y);

                    uint8_t count = 0;
                    std::vector<uint8_t> act_offsets;
                    while (act_bits) {
                        auto current_bit = act_bits & 1;
                        if (current_bit) act_offsets.push_back(count);
                        act_bits >>= 1;
                        count++;
                    }

                    auto max_act_bit = act_offsets.empty() ? 0 : *std::max_element(act_offsets.begin(),
                            act_offsets.end());
                    auto min_act_bit = act_offsets.empty() ? 16 : *std::min_element(act_offsets.begin(),
                            act_offsets.end());

                    if (max_act_bit > max_bit) max_bit = max_act_bit;
                    if (min_act_bit < min_bit) min_bit = min_act_bit;

                }
            }
            per_SIP_n_bits[window] = min_bit > max_bit ? 1 : max_bit - min_bit + 1;
        }

        uint8_t n_bits = PRECISION_GRANULARITY == "SIP" ?
                         *std::max_element(per_SIP_n_bits.begin(), per_SIP_n_bits.end()) :
                         min_bit > max_bit ? 1 : max_bit - min_bit + 1;
        return n_bits;
    }

    /* CYCLES */

    template <typename T>
    void BitTacticalP<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

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

        int num_filters = wgt_shape[0];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        // Stats
        std::vector<uint32_t> cycles (batch_size,0);
        uint32_t batch_cycles;

        std::vector<int> list_x, list_y;
        int n, x_counter, y_counter;
        schedule tmp_schedule;

        const auto &dense_schedule = this->scheduler(wgt,act_channels);

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,batch_cycles,tmp_schedule,x_counter,y_counter,list_x,list_y)
        #endif
        for(n=0; n<batch_size; n++) {
            batch_cycles = 0, x_counter = 0, y_counter = 0;
            tmp_schedule = dense_schedule;
            for(int m=0; m<num_filters; m+=this->N_ROWS) {
                while(this->check_schedule(tmp_schedule,m,num_filters)) {
                    while (this->iterateWindows(out_x,out_y,list_x,list_y,x_counter,y_counter,this->N_COLUMNS)) {
                        batch_cycles += computeTacticalPTile(n, list_x, list_y, m, stride, padded_act, num_filters,
                                tmp_schedule);
                    }
                    this->update_schedule(tmp_schedule,m,num_filters);
                }
            }
            cycles[n] = batch_cycles;
        }

        auto avg_cycles = accumulate(cycles.begin(), cycles.end(), 0.0)/cycles.size();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.TCLP_cycles.push_back(cycles);
        stats.TCLP_avg_cycles.push_back((uint32_t)avg_cycles);

    }

    template <typename T>
    void BitTacticalP<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

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
        int act_channels = act_shape[1];
        int num_filters = wgt_shape[0];

        // Stats
        std::vector<uint32_t> cycles (batch_size,0);
        uint32_t batch_cycles;

        int n;
        schedule tmp_schedule;

        const auto &dense_schedule = this->scheduler(wgt,act_channels);

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,batch_cycles)
        #endif
        for (n = 0; n<batch_size; n++) {
            batch_cycles = 0;
            tmp_schedule = dense_schedule;
            for (int m = 0; m<num_filters; m+=this->N_ROWS) {
                while(this->check_schedule(tmp_schedule,m,num_filters)) {
                    batch_cycles += computeTacticalPColumn(n,0,0,m,0,act,act_channels,tmp_schedule);
                    this->update_schedule(tmp_schedule,m,num_filters);
                }
            }
            cycles[n] = batch_cycles;
        }

        auto avg_cycles = accumulate(cycles.begin(), cycles.end(), 0.0)/cycles.size();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.TCLP_cycles.push_back(cycles);
        stats.TCLP_avg_cycles.push_back((uint32_t)avg_cycles);

    }

    template <typename T>
    void BitTacticalP<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        int mux_entries = this->LOOKAHEAD_H + this->LOOKASIDE_D + 1;
        stats.arch = "BitTacticalP_C" + std::to_string(this->N_COLUMNS) + "_R" + std::to_string(this->N_ROWS) + "_PG_" +
                PRECISION_GRANULARITY + "_" + this->SEARCH_SHAPE + std::to_string(mux_entries) + "(" +
                std::to_string(this->LOOKAHEAD_H) + "-" + std::to_string(this->LOOKASIDE_D) + ")";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                computeConvolution(layer, stats);
            } else if(layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                computeInnerProduct(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* POTENTIALS */

    template <typename T>
    void BitTacticalP<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        const cnpy::Array<T> &act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        int Nx = act_shape[2];
        int Ny = act_shape[3];

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        int groups = act_channels / wgt_channels;
        int it_per_group = num_filters / groups;

        // Operations
        const auto parallel_mult = (uint64_t)(num_filters * out_x * out_y * Kx * Ky * wgt_channels);
        std::vector<uint64_t> bit_multiplications (batch_size,0);
        std::vector<double> work_reduction (batch_size,0);
        std::vector<double> speedup (batch_size,0);
        uint64_t bit_counter = 0;

        int current_group = 0, group_m =0, start_group = 0;
        int n;

        // Get layer precision
        auto act_layer_prec = std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision());

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,current_group,group_m,start_group,bit_counter)
        #endif
        for(n=0; n<batch_size; n++) {
            current_group = 0; group_m =0; start_group = 0; bit_counter = 0;
            for(int m=0; m<num_filters; m++) {
                for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = start_group; k < wgt_channels + start_group; k++) {
                            bit_counter += computeTacticalPBitsPE(wgt.get(m, k - start_group, i, j),
                                    (uint8_t)act_layer_prec);
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
            work_reduction[n] = 100 - ((double)(bit_counter * out_x * out_y) / (double)parallel_mult / 256. * 100);
            speedup[n] = (double)parallel_mult * 256. / (double)(bit_counter * out_x * out_y);
            bit_multiplications[n] = bit_counter * out_x * out_y;
        }

        auto avg_bit_multiplications = (uint64_t)accumulate(bit_multiplications.begin(), bit_multiplications.end(), 0.0)
                                       / bit_multiplications.size();
        auto avg_work_reduction = accumulate(work_reduction.begin(), work_reduction.end(), 0.0) / work_reduction.size();
        auto avg_speedup = accumulate(speedup.begin(), speedup.end(), 0.0) / speedup.size();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.work_reduction.push_back(work_reduction);
        stats.avg_work_reduction.push_back(avg_work_reduction);
        stats.speedup.push_back(speedup);
        stats.avg_speedup.push_back(avg_speedup);
        stats.parallel_multiplications.push_back(parallel_mult);
        stats.bit_multiplications.push_back(bit_multiplications);
        stats.avg_bit_multiplications.push_back(avg_bit_multiplications);

    }

    template <typename T>
    void BitTacticalP<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        const cnpy::Array<T> &wgt = layer.getWeights();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels;
        std::vector<uint64_t> bit_multiplications (batch_size,0);
        std::vector<double> work_reduction (batch_size,0);
        std::vector<double> speedup (batch_size,0);
        uint64_t bit_counter = 0;

        int n;

        // Get layer precision
        auto act_layer_prec = std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision());

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,bit_counter)
        #endif
        for (n = 0; n<batch_size; n++) {
            bit_counter = 0;
            for (int m = 0; m<num_filters; m++) {
                for (int k = 0; k<wgt_channels; k++) {
                    bit_counter += computeTacticalPBitsPE(wgt.get(m, k), (uint8_t)act_layer_prec);
                }
            }
            work_reduction[n] = 100 - ((double) bit_counter / (double) parallel_mult / 256. * 100);
            speedup[n] = (double)parallel_mult * 256. / (double)bit_counter;
            bit_multiplications[n] = bit_counter;
        }

        auto avg_bit_multiplications = (uint64_t)accumulate(bit_multiplications.begin(), bit_multiplications.end(), 0.0)
                                       / bit_multiplications.size();
        auto avg_work_reduction = accumulate(work_reduction.begin(), work_reduction.end(), 0.0) / work_reduction.size();
        auto avg_speedup = accumulate(speedup.begin(), speedup.end(), 0.0) / speedup.size();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.work_reduction.push_back(work_reduction);
        stats.avg_work_reduction.push_back(avg_work_reduction);
        stats.speedup.push_back(speedup);
        stats.avg_speedup.push_back(avg_speedup);
        stats.parallel_multiplications.push_back(parallel_mult);
        stats.bit_multiplications.push_back(bit_multiplications);
        stats.avg_bit_multiplications.push_back(avg_bit_multiplications);

    }

    template <typename T>
    void BitTacticalP<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "BitTacticalP";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(0);
                computePotentialsConvolution(layer,stats);
            } else if (layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(0);
                computePotentialsInnerProduct(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class BitTacticalP<uint16_t>;

}

