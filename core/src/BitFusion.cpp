
#include <core/BitFusion.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t BitFusion<T>::computeBitFusionBitsPE(uint8_t act_rounded_precision, uint8_t wgt_rounded_precision) {
        return act_rounded_precision * wgt_rounded_precision;
    }

    /* CYCLES */

    template <typename T>
    void BitFusion<T>::run(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = network.getName() + "_BitFusion_N" + std::to_string(N) + "_M" + std::to_string(M) +
                "_PMAX" + std::to_string(PMAX) + "_PMIN" + std::to_string(PMIN) + "_cycles";
        sys::Stats stats = sys::Stats(network.getNumLayers(), network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto performance = stats.register_uint_t("perfomance_factor", 0, sys::Average);
        auto time_mux = stats.register_uint_t("time_multiplex", 0, sys::Average);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";

            base::Array<T> act = layer.getActivations();
            if(!conv && act.getDimensions() == 4) act.reshape_to_2D();
            base::Array<T> wgt = layer.getWeights();

            int padding = layer.getPadding();
            int stride = layer.getStride();

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

            for (int n = 0; n < batch_size; n++) {
                cycles->value[layer_it][n] = compute_cycles * time_multiplex;
                performance->value[layer_it][n] = perf_factor;
                time_mux->value[layer_it][n] = time_multiplex;
                act_prec->value[layer_it][n] = act_layer_prec;
                wgt_prec->value[layer_it][n] = wgt_layer_prec;
            }

        }

        //Dump statistics
        stats.dump_csv(network.getName(), network.getLayersName(), this->QUIET);

    }

    /* POTENTIALS */

    template <typename T>
    void BitFusion<T>::potentials(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = network.getName() + "_BitFusion_potentials";
        sys::Stats stats = sys::Stats(network.getNumLayers(), network.getBatches(), filename);

        auto bit_multiplications = stats.register_uint_t("bit_multiplications", 0, sys::AverageTotal);
        auto work_reduction = stats.register_double_t("work_reduction", 0, sys::Average);
        auto speedup = stats.register_double_t("speedup", 0, sys::Average);
        auto par_mult = stats.register_double_t("parallel_multiplication", 0, sys::AverageTotal);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        for(auto layer_it = 0; layer_it < network.getLayers().size(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";

            base::Array<T> act = layer.getActivations();
            if(!conv && act.getDimensions() == 4) act.reshape_to_2D();
            const base::Array<T> &wgt = layer.getWeights();

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

            int padding = layer.getPadding();
            int stride = layer.getStride();

            long out_x = (Nx - Kx + 2*padding)/stride + 1;
            long out_y = (Ny - Ky + 2*padding)/stride + 1;

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;

            // Get layer precision
            auto act_layer_prec = layer.getActPrecision();
            auto act_rounded_log2 = ceil(log(act_layer_prec)/log(2));
            auto act_rounded_precision = (uint8_t)pow(2,act_rounded_log2);

            auto wgt_layer_prec = layer.getWgtPrecision();
            auto wgt_rounded_log2 = ceil(log(wgt_layer_prec)/log(2));
            auto wgt_rounded_precision = (uint8_t)pow(2,wgt_rounded_log2);

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network.getNetwork_bits() * network.getNetwork_bits();
                uint64_t bit_counter = 0;

                bit_counter = (uint64_t)computeBitFusionBitsPE(act_rounded_precision, wgt_rounded_precision);
                bit_counter *= conv ? out_x * out_y * Kx * Ky * wgt_channels * num_filters :
                        wgt_channels * num_filters * R;

                bit_multiplications->value[layer_it][n] = bit_counter;
                work_reduction->value[layer_it][n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
                speedup->value[layer_it][n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
                par_mult->value[layer_it][n] = parallel_mult;
            }

        }

        //Dump statistics
        stats.dump_csv(network.getName(), network.getLayersName(), this->QUIET);

    }

    template class BitFusion<uint16_t>;

}
