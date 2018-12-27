
#include <core/Laconic.h>

#define ZERO_COUNT
#define BOOTH_ENCODING

namespace core {

    template <typename T>
    void Laconic<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {}

    template <typename T>
    void Laconic<T>::run(const Network<T> &network) {}

    template <typename T>
    uint8_t Laconic<T>::calculateOneBitMultiplications(uint16_t act, uint16_t wgt) {

        uint16_t act_bits = act;
        uint16_t wgt_bits = wgt;

        #ifdef BOOTH_ENCODING
            act_bits = this->booth_encoding(act_bits);
            wgt_bits = this->booth_encoding(wgt_bits);
        #endif

        uint8_t act_effectual_bits = 0;
        while (act_bits) {
            act_effectual_bits += act_bits & 1;
            act_bits >>= 1;
        }
        uint8_t wgt_effectual_bits = 0;
        while (wgt_bits) {
            wgt_effectual_bits += wgt_bits & 1;
            wgt_bits >>= 1;
        }

        uint8_t one_bit_multiplications = act_effectual_bits * wgt_effectual_bits;
        #ifdef ZERO_COUNT
        if(one_bit_multiplications == 0) one_bit_multiplications = 1;
        #endif

        return one_bit_multiplications;
    }

    template <typename T>
    void Laconic<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> wgt = layer.getWeights();
        wgt.two_exponents_representation();
        cnpy::Array<T> act = layer.getActivations();
        act.two_exponents_representation();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int padding = layer.getPadding();
        int stride = layer.getStride();
        int Kx = layer.getKx();
        int Ky = layer.getKy();

        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (act_shape[2] - wgt_shape[2] + 2*padding)/stride + 1;
        long out_y = (act_shape[3] - wgt_shape[3] + 2*padding)/stride + 1;

        // Set filter grouping
        int groups = act_channels / wgt_channels;
        int it_per_group = num_filters / groups;

        // Operations
        const auto mult_16bit = (uint64_t)(num_filters * out_x * out_y * Kx * Ky * wgt_channels);
        std::vector<uint64_t> one_bit_multiplications (act_shape[0],0);
        std::vector<double> potentials (act_shape[0],0);
        int current_group = 0, group_m =0, start_group = 0;
        uint64_t one_bit_counter = 0;
        int n;

        // Convolution
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,current_group,group_m,start_group,one_bit_counter)
        #endif
        for(n=0; n<batch_size; n++) {
            current_group = 0; group_m =0; start_group = 0; one_bit_counter = 0;
            for(int m=0; m<num_filters; m++) {
                for(int x=0; x<out_x; x++) {
                    for(int y=0; y<out_y; y++) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = start_group; k < wgt_channels + start_group; k++) {
                                    one_bit_counter += calculateOneBitMultiplications(
                                            padded_act.get(n, k, stride * x + i, stride * y + j),
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
            potentials[n] = 100 - ((double)one_bit_counter / (double)mult_16bit / 256. * 100);
            one_bit_multiplications[n] = one_bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.potentials.push_back(potentials);
        stats.multiplications.push_back(mult_16bit);
        stats.one_bit_multiplications.push_back(one_bit_multiplications);
    }


    template <typename T>
    void Laconic<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> wgt = layer.getWeights();
        wgt.two_exponents_representation();
        cnpy::Array<T> act = layer.getActivations();
        act.two_exponents_representation();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];

        // Operations
        const uint64_t mult_16bit = wgt_shape[0] * wgt_shape[1];
        std::vector<uint64_t> one_bit_multiplications (act_shape[0],0);
        std::vector<double> potentials (act_shape[0],0);
        uint64_t one_bit_counter = 0;
        int n;

        if(act.getDimensions() == 2) {

            #ifdef OPENMP
            auto max_threads = omp_get_max_threads();
            omp_set_num_threads(max_threads);
            #pragma omp parallel for private(n,one_bit_counter)
            #endif
            for (n = 0; n<batch_size; n++) {
                one_bit_counter = 0;
                for (uint16_t m = 0; m<num_filters; m++) {
                    for (uint16_t k = 0; k<wgt_channels; k++) {
                        one_bit_counter += calculateOneBitMultiplications(act.get(n, k), wgt.get(m, k));
                    }
                }
                potentials[n] = 100 - ((double) one_bit_counter / (double) mult_16bit / 256. * 100);
                one_bit_multiplications[n] = one_bit_counter;
            }
        } else if (act.getDimensions() == 4) {

            #ifdef OPENMP
            auto max_threads = omp_get_max_threads();
            omp_set_num_threads(max_threads);
            #pragma omp parallel for private(n,one_bit_counter)
            #endif
            for (n = 0; n<batch_size; n++) {
                one_bit_counter = 0;
                for (uint16_t m = 0; m<num_filters; m++) {
                    for (uint16_t k = 0; k<wgt_channels; k++) {
                        int f_dim = (int)(k / (act_shape[2]*act_shape[3]));
                        auto rem = k % (act_shape[2]*act_shape[3]);
                        int s_dim = (int)(rem / act_shape[3]);
                        int t_dim = (int)(rem % act_shape[3]);
                        one_bit_counter += calculateOneBitMultiplications(act.get(n,f_dim,s_dim,t_dim), wgt.get(m, k));
                    }
                }
                potentials[n] = 100 - ((double) one_bit_counter / (double) mult_16bit / 256. * 100);
                one_bit_multiplications[n] = one_bit_counter;
            }
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.potentials.push_back(potentials);
        stats.multiplications.push_back(mult_16bit);
        stats.one_bit_multiplications.push_back(one_bit_multiplications);

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
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision())+std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(std::get<0>(layer.getWgt_precision())+std::get<1>(layer.getWgt_precision()));
                computePotentialsConvolution(layer,stats);
            } else if (layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision())+std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(std::get<0>(layer.getWgt_precision())+std::get<1>(layer.getWgt_precision()));
                computePotentialsInnerProduct(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class Laconic<uint16_t>;

}