
#include <core/BitFusion.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t BitFusion<T>::computeBitFusionBitsPE(uint8_t act_rounded_precision, uint8_t wgt_rounded_precision) {
        return act_rounded_precision * wgt_rounded_precision;
    }

    /* CYCLES */

    template <typename T>
    void BitFusion<T>::computeLayer(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
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

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = 1;
        uint64_t Nx, Ny, R;
        if(layer.getType() == "LSTM") {
            R = act_shape[0];
            Nx = 1;
            Ny = 1;
        } else {
            R = 1;
            Nx = act_shape[2];
            Ny = act_shape[3];
        }

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        long out_x = (Nx - Kx)/stride + 1;
        long out_y = (Ny - Ky)/stride + 1;

        // Get layer precision
        auto act_layer_prec = layer.getActPrecision();
        auto wgt_layer_prec = layer.getWgtPrecision();

        uint8_t time_multiplex = 1;
        if(act_layer_prec > 8) {
            act_layer_prec = 8;
            time_multiplex *= 2;
        }
        if(wgt_layer_prec > 8) {
            wgt_layer_prec = 8;
            time_multiplex *= 2;
        }

        act_layer_prec = std::max(act_layer_prec, (int)PMIN);
        wgt_layer_prec = std::max(wgt_layer_prec, (int)PMIN);
        auto perf_factor = (PMAX/act_layer_prec) * (PMAX/wgt_layer_prec);

        auto filter_sets = (int)ceil(num_filters / (double)M);
        auto activation_sets = (int)ceil(wgt_channels / (double)(N * perf_factor));
        auto compute_cycles = filter_sets * out_x * out_y * Kx * Ky * activation_sets * R;

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(batch_size,0));

        for (int n = 0; n < batch_size; n++) {
            stats.cycles.back()[n] = compute_cycles * time_multiplex;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.perf_factor.push_back((unsigned)perf_factor);
        stats.time_multiplex.push_back((unsigned)time_multiplex);

    }

    template <typename T>
    void BitFusion<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        stats.arch = "BitFusion_N" + std::to_string(N) + "_M" + std::to_string(M) + "_PMAX" + std::to_string(PMAX) +
                "_PMIN" + std::to_string(PMIN);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution" || layer.getType() == "InnerProduct" || layer.getType() == "LSTM") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getActPrecision());
                stats.wgt_prec.push_back(layer.getWgtPrecision());
                computeLayer(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);

    }

    /* POTENTIALS */

    template <typename T>
    void BitFusion<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        const cnpy::Array<T> &act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = 1;
        auto Nx = act_shape[2];
        auto Ny = act_shape[3];

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

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
        auto act_layer_prec = layer.getActPrecision();
        auto act_rounded_log2 = ceil(log(act_layer_prec)/log(2));
        auto act_rounded_precision = (uint8_t)pow(2,act_rounded_log2);

        auto wgt_layer_prec = layer.getWgtPrecision();
        auto wgt_rounded_log2 = ceil(log(wgt_layer_prec)/log(2));
        auto wgt_rounded_precision = (uint8_t)pow(2,wgt_rounded_log2);

        // Convolution
        for(int n=0; n<batch_size; n++) {
            double MAX_BITS = network_bits * network_bits;
            bit_counter = (uint64_t)computeBitFusionBitsPE(act_rounded_precision,wgt_rounded_precision) * out_x * out_y
                    * Kx * Ky * wgt_channels * num_filters;
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
    void BitFusion<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
            const int network_bits) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        const cnpy::Array<T> &wgt = layer.getWeights();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = 1;
        auto R = (layer.getType() == "LSTM") ? act_shape[0] : 1;

        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels * R;
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));
        uint64_t bit_counter = 0;

        // Get layer precision
        auto act_layer_prec = layer.getActPrecision();
        auto act_rounded_log2 = ceil(log(act_layer_prec)/log(2));
        auto act_rounded_precision = (uint8_t)pow(2,act_rounded_log2);

        auto wgt_layer_prec = layer.getWgtPrecision();
        auto wgt_rounded_log2 = ceil(log(wgt_layer_prec)/log(2));
        auto wgt_rounded_precision = (uint8_t)pow(2,wgt_rounded_log2);

        for (int n = 0; n < batch_size; n++) {
            double MAX_BITS = network_bits * network_bits;
            bit_counter = (uint64_t)computeBitFusionBitsPE(act_rounded_precision,wgt_rounded_precision) *
                    wgt_channels * num_filters * R;
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
    void BitFusion<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "BitFusion";

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

    template class BitFusion<uint16_t>;

}
