
#include <core/Simulator.h>

namespace core {

    /* CYCLES */

    template <typename T>
    void Simulator<T>::run(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch) {

    }

    /* POTENTIALS */

    template <typename T>
    void Simulator<T>::potentials(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch) {

        // Initialize statistics
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(),
                arch->filename_pot());

        auto work_reduction = stats.register_double_t("work_reduction", 0, sys::Average);
        auto speedup = stats.register_double_t("speedup", 0, sys::Average);
        auto par_mult = stats.register_double_t("parallel_multiplication", 0, sys::AverageTotal);
        auto bit_multiplications = stats.register_uint_t("bit_multiplications", 0, sys::AverageTotal);
        auto act_precision = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_precision = stats.register_uint_t("weights_precision", 0, sys::Average);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            arch->dataConversion(act, layer.getActPrecision());
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();

            base::Array<T> wgt = layer.getWeights();
            arch->dataConversion(wgt, layer.getWgtPrecision());
            if(conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            uint64_t batch_size, act_channels, Nx, Ny, R;
            if (lstm) {
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

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long out_x = (Nx - Kx)/stride + 1;
            long out_y = (Ny - Ky)/stride + 1;

            auto groups = act_channels / wgt_channels;
            auto it_per_group = num_filters / groups;

            auto network_bits = network.getNetwork_bits();
            auto act_prec = layer.getActPrecision();
            auto wgt_prec = layer.getWgtPrecision();

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                                     num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                if (conv) {

                    for(int m = 0; m < num_filters; m++) {

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
                                            T act_bits = act.get(n, start_group + k, stride * x + i, stride * y + j);
                                            T wgt_bits = wgt.get(m, k, i, j);
                                            bit_counter += arch->computeBits(act_bits, wgt_bits, act_prec, wgt_prec,
                                                    network_bits);
                                        }
                                    }
                                }
                            }
                        }
                    }

                } else {

                    for (int r = 0; r < R; r++) {
                        for (int m = 0; m < num_filters; m++) {
                            for (int k = 0; k < wgt_channels; k++) {
                                T act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                                T wgt_bits = wgt.get(m, k);
                                bit_counter += arch->computeBits(act_bits, wgt_bits, act_prec, wgt_prec, network_bits);
                            }
                        }
                    }

                }

                bit_multiplications->value[layer_it][n] = bit_counter;
                work_reduction->value[layer_it][n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
                speedup->value[layer_it][n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
                par_mult->value[layer_it][n] = parallel_mult;
                act_precision->value[layer_it][n] = layer.getActPrecision();
                wgt_precision->value[layer_it][n] = layer.getWgtPrecision();
            }

        }

        //Dump statistics
        stats.dump_csv(network.getName(), network.getLayersName(), arch->header_pot(network.getName()), this->QUIET);

    }


    INITIALISE_DATA_TYPES(Simulator);

}
