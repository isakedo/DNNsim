
#include <core/Inference.h>

namespace core {

    /* CYCLES */

    template <typename T>
    void Inference<T>::run(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch) {

        // Initialize statistics
        std::string filename = arch->name() + "_L" + std::to_string(N_LANES) + "_C" + std::to_string(N_COLUMNS) +
                "_R" + std::to_string(N_ROWS) + "_T" + std::to_string(N_TILES) + "_BP" + std::to_string(BITS_PE) +
                arch->filename() + "_cycles";

        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);

        auto act_precision = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_precision = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto network_bits = network.getNetwork_bits();
        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            if (!QUIET) std::cout << "Simulating layer: " << layer.getName() << std::endl;

            base::Array<T> act = layer.getActivations();
            arch->dataConversion(act, layer.getActPrecision());
            if (act.getDimensions() == 4) act.reshape_to_2D();

            base::Array<T> wgt = layer.getWeights();
            arch->dataConversion(wgt, layer.getWgtPrecision());
            if (conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

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
            if (FAST_MODE) batch_size = 1;

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long Ox = (Nx - Kx) / stride + 1;
            long Oy = (Ny - Ky) / stride + 1;

            auto groups = act_channels / wgt_channels;
            auto filters_per_group = num_filters / groups;

            auto act_prec = layer.getActPrecision();
            auto wgt_prec = layer.getWgtPrecision();

            // TODO BITS PER PE

            // Generate weight buffer
            auto filter_sets = (uint64_t)ceil(num_filters / (double)N_ROWS);

            auto round_wgt_channels = (int)ceil(wgt_channels / (double)N_LANES) * N_LANES;
            auto time_per_filter = (uint64_t)ceil(round_wgt_channels * Kx * Ky / (double)N_LANES);

            Buffer<ValueTuple<T>> weight_buffer;
            weight_buffer = Buffer<ValueTuple<T>>(filter_sets, BufferSet<ValueTuple<T>>(time_per_filter,
                    BufferRow<ValueTuple<T>>(N_ROWS * N_LANES, std::make_tuple(0, 0, 0))));

            int set_wgt = -1;
            for(int m = 0; m < num_filters; ++m) {

                if ((m % N_ROWS) == 0)
                    set_wgt++;

                int time = 0;
                for (int y = 0; y < Ky; ++y) {
                    for (int x = 0; x < Kx; ++x) {
                        for (int k = 0; k < wgt_channels; k += N_LANES) {
                            int index = 0;
                            for(int ch = k; ch < std::min((uint64_t)k + N_LANES, wgt_channels); ++ch) {

                                auto wgt_bits = wgt.get(m, ch, x, y);
                                int pos = (m % N_ROWS) * N_LANES + index;
                                weight_buffer[set_wgt][time][pos] = std::make_tuple(wgt_bits, time, index);

                                index++;
                                if(index == N_LANES) {
                                    time++;
                                    index = 0;
                                }
                            }
                            if(index != 0)
                                time++;
                        }
                    }
                }

            }

            if (arch->schedule()) {
                // TODO SCHEDULE
            }

            // Iterate over the images
            for (int n = 0; n < batch_size; ++n) {

                if (conv) {

                } else {

                }

                if (CHECK) {

                }

                act_precision->value[layer_it][n] = act_prec;
                wgt_precision->value[layer_it][n] = wgt_prec;

            }

        }


        //Dump statistics
        std::string header = arch->name() + " Number of Cycles for " + network.getName() + "\n";
        header += "Number of lanes/terms per PE: " + std::to_string(N_LANES) + "\n";
        header += "Number of columns/windows in parallel: " + std::to_string(N_COLUMNS) + "\n";
        header += "Number of rows/filters in parallel: " + std::to_string(N_ROWS) + "\n";
        header += "Number of tiles: " + std::to_string(N_TILES) + "\n";
        header += "Size of the PE in bits: " + std::to_string(BITS_PE) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header + arch->header(), QUIET);
    }

    /* POTENTIALS */

    template <typename T>
    void Inference<T>::potentials(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch) {

        // Initialize statistics
        std::string filename = arch->name() + arch->filename_pot() + "_potentials";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto work_reduction = stats.register_double_t("work_reduction", 0, sys::Special);
        auto speedup = stats.register_double_t("speedup", 0, sys::Special);
        auto bit_mult = stats.register_uint_t("bit_multiplications", 0, sys::AverageTotal);
        auto max_bit_mult = stats.register_uint_t("max_bit_multiplications", 0, sys::AverageTotal);
        auto max_par_mult = stats.register_double_t("max_parallel_multiplication", 0, sys::AverageTotal);
        auto act_precision = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_precision = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto network_bits = network.getNetwork_bits();
        double MAX_BITS = network_bits * network_bits;
        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            if (!QUIET) std::cout << "Simulating layer: " << layer.getName() << std::endl;

            base::Array<T> act = layer.getActivations();
            arch->dataConversion(act, layer.getActPrecision());
            if (act.getDimensions() == 4) act.reshape_to_2D();

            base::Array<T> wgt = layer.getWeights();
            arch->dataConversion(wgt, layer.getWgtPrecision());
            if (conv && wgt.getDimensions() == 2) wgt.reshape_to_4D();

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
            if (FAST_MODE) batch_size = 1;

            auto num_filters = wgt_shape[0];
            auto wgt_channels = wgt_shape[1];
            auto Kx = wgt_shape[2];
            auto Ky = wgt_shape[3];

            long Ox = (Nx - Kx) / stride + 1;
            long Oy = (Ny - Ky) / stride + 1;

            auto groups = act_channels / wgt_channels;
            auto filters_per_group = num_filters / groups;

            auto act_prec = layer.getActPrecision();
            auto wgt_prec = layer.getWgtPrecision();

            // Operations
            uint64_t max_par_counter = conv ? num_filters * Ox * Oy * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;
            uint64_t max_bit_counter = max_par_counter * MAX_BITS;

            for(int n = 0; n < batch_size; ++n) {

                // Stats
                uint64_t bit_counter = 0;

                if (conv) {

                    for(int m = 0; m < num_filters; ++m) {

                        // Two towers alexnet
                        int start_group = 0;
                        if(m >= filters_per_group)
                            start_group = (int)wgt_channels;

                        // Fix for MobileNet
                        if(wgt_channels == 1 && act_channels != 1)
                            start_group = m;

                        for(int x = 0; x < Ox; ++x) {
                            for(int y = 0; y < Oy; ++y) {
                                for (int i = 0; i < Kx; ++i) {
                                    for (int j = 0; j < Ky; ++j) {
                                        for (int k = 0; k < wgt_channels; ++k) {
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

                    for (int r = 0; r < R; ++r) {
                        for (int m = 0; m < num_filters; ++m) {
                            for (int k = 0; k < wgt_channels; ++k) {
                                T act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                                T wgt_bits = wgt.get(m, k);
                                bit_counter += arch->computeBits(act_bits, wgt_bits, act_prec, wgt_prec, network_bits);
                            }
                        }
                    }

                }

                bit_mult->value[layer_it][n] = bit_counter;
                max_bit_mult->value[layer_it][n] = max_bit_counter;
                max_par_mult->value[layer_it][n] = max_par_counter;

                work_reduction->value[layer_it][n] = 100 - (bit_counter / (double)max_bit_counter * 100.);
                speedup->value[layer_it][n] = max_bit_counter / (double)bit_counter;

                act_precision->value[layer_it][n] = layer.getActPrecision();
                wgt_precision->value[layer_it][n] = layer.getWgtPrecision();
            }

            work_reduction->special_value_vector.push_back(100 - (sys::get_total(bit_mult->value[layer_it]) /
                    (double)sys::get_total(max_bit_mult->value[layer_it]) * 100.));
            speedup->special_value_vector.push_back(sys::get_total(max_bit_mult->value[layer_it]) /
                    (double)(sys::get_total(bit_mult->value[layer_it])));

        }

        work_reduction->special_value = 100 - (sys::get_total(bit_mult->value) /
                (double)sys::get_total(max_bit_mult->value) * 100.);
        speedup->special_value = sys::get_total(max_bit_mult->value) / (double)(sys::get_total(bit_mult->value));

        //Dump statistics
        std::string header = arch->name() + " Potentials/Work Reduction for " + network.getName() + "\n";
        stats.dump_csv(network.getName(), network.getLayersName(), header + arch->header_pot(), QUIET);

    }


    INITIALISE_DATA_TYPES(Inference);

}
