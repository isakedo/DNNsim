
#include <core/BitTacticalE.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint8_t BitTacticalE<T>::computeBitTacticalEBitsPE(uint16_t act, uint16_t wgt) {

        if(wgt == 0) return 0;

        uint16_t act_bits = act;

        #ifdef BOOTH_ENCODING
        act_bits = this->booth_encoding(act_bits);
        #endif

        uint8_t act_effectual_bits = 0;
        while (act_bits) {
            act_effectual_bits += act_bits & 1;
            act_bits >>= 1;
        }

        uint8_t bit_multiplications = act_effectual_bits * (uint8_t)16;
        #ifdef ZERO_COUNT
        if(bit_multiplications == 0) bit_multiplications = 1;
        #endif

        return bit_multiplications;
    }

    /* CYCLES */

    template <typename T>
    void BitTacticalE<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

    }

    template <typename T>
    void BitTacticalE<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

    }

    template <typename T>
    void BitTacticalE<T>::run(const Network<T> &network) {

    }

    /* POTENTIALS */

    template <typename T>
    void BitTacticalE<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
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

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,current_group,group_m,start_group,bit_counter)
        #endif
        for(n=0; n<batch_size; n++) {
            current_group = 0; group_m =0; start_group = 0; bit_counter = 0;
            for(int m=0; m<num_filters; m++) {
                for(int x=0; x<out_x; x++) {
                    for(int y=0; y<out_y; y++) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = start_group; k < wgt_channels + start_group; k++) {
                                    bit_counter += computeBitTacticalEBitsPE(padded_act.get(n, k, stride * x + i,
                                            stride * y + j),wgt.get(m, k - start_group, i, j));
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
    void BitTacticalE<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        cnpy::Array<T> wgt = layer.getWeights();

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

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,bit_counter)
        #endif
        for (n = 0; n<batch_size; n++) {
            bit_counter = 0;
            for (int m = 0; m<num_filters; m++) {
                for (int k = 0; k<wgt_channels; k++) {
                    bit_counter += computeBitTacticalEBitsPE(act.get(n, k), wgt.get(m, k));
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
    void BitTacticalE<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "BitTacticalE";

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

    template class BitTacticalE<uint16_t>;

}