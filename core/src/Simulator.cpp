
#include <core/Simulator.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void check_result(const OutputTensor &sim_output, const std::shared_ptr<base::Array<T>> &act,
            const std::shared_ptr<base::Array<T>> &wgt, uint64_t Ox, uint64_t Oy, int stride, bool _3dim) {

        const std::vector<size_t> &act_shape = act->getShape();
        const std::vector<size_t> &wgt_shape = wgt->getShape();

        // Activations
        auto R = _3dim ? act_shape[1] : 1;
        auto act_channels = _3dim ? act_shape[2] : act_shape[1];

        // Weights
        auto num_filters = wgt_shape[0];
        auto wgt_channels = wgt_shape[1];
        auto Kx = wgt_shape[2];
        auto Ky = wgt_shape[3];

        auto groups = act_channels / wgt_channels;
        auto filters_per_group = num_filters / groups;

        OutputTensor output = OutputTensor(num_filters, std::vector<std::vector<double>>(Ox,
                std::vector<double>(Oy, 0)));

        // Actual convolution
        for (int r = 0; r < R; ++r) {

            for (int m = 0; m < num_filters; ++m) {

                // Two towers alexnet
                int start_group = 0;
                if (m >= filters_per_group)
                    start_group = (int) wgt_channels;

                // Fix for MobileNet
                if (wgt_channels == 1 && act_channels != 1)
                    start_group = m;

                // Number of Windows
                for (int x = 0; x < Ox; ++x) {
                    for (int y = 0; y < Oy; ++y) {

                        double sum = 0;

                        // Window dimension
                        for (int j = 0; j < Ky; ++j) {
                            for (int i = 0; i < Kx; ++i) {
                                for (int k = 0; k < wgt_channels; ++k) {
                                    auto act_bits = _3dim ? act->get(0, r, k) :
                                            act->get(0, start_group + k, stride * x + i, stride * y + j);
                                    sum += act_bits * wgt->get(m, k, i, j);
                                }
                            }
                        }

                        output[m][x][y] += sum;
                    }
                }
            }
        }

        // Check values
        for (int ch = 0; ch < num_filters; ++ch) {
            for (int x = 0; x < Ox; ++x) {
                for (int y = 0; y < Oy; ++y) {
                    auto actual_value = output[ch][x][y];
                    auto sim_value = sim_output[ch][x][y];
                    auto error = (actual_value - sim_value) / sim_value;
                    if (abs(error) > 1e-10)
                        throw std::runtime_error("Wrong value.");
                }
            }
        }
    }

    template <typename T>
    void calculate_output(OutputTensor &output, const std::vector<TileData<T>> &tiles_data) {

        for (const auto &tile_data : tiles_data) {

            if (!tile_data.valid)
                continue;

            for (int w = 0; w < tile_data.windows.size(); ++w) {
                auto window_idx = w * tile_data.lanes;
                auto x_window = std::get<0>(tile_data.windows[w]);
                auto y_window = std::get<1>(tile_data.windows[w]);

                for (int f = 0; f < tile_data.filters.size(); ++f) {
                    auto filter_idx = f * tile_data.lanes;
                    auto filter = tile_data.filters[f];

                    for (int lane = 0; lane < tile_data.lanes; ++lane) {

                        auto wgt_bits = std::get<0>(tile_data.wgt_row[filter_idx + lane]);
                        auto time_h = (std::get<1>(tile_data.wgt_row[filter_idx + lane]) - tile_data.time);
                        auto lane_d = std::get<2>(tile_data.wgt_row[filter_idx + lane]);

                        if (time_h < 0) continue;

                        auto act_bits = std::get<0>(tile_data.act_row[time_h][window_idx + lane_d]);

                        output[filter][x_window][y_window] += act_bits * wgt_bits;

                    } // Multiply 16 weights and 16 activations values
                } // Filter
            } // Window
        } // Tiles

    }

    /* CYCLES */

    template <typename T>
    void Simulator<T>::run(const base::Network<T> &network, const std::shared_ptr<Control<T>> &control) {

        // Get components from control
        auto dram = control->getDram();
        auto gbuffer = control->getGbuffer();
        auto abuffer = control->getAbuffer();
        auto wbuffer = control->getWbuffer();
        auto obuffer = control->getObuffer();
        auto arch = control->getArch();

        std::shared_ptr<uint64_t> global_cycle = std::make_shared<uint64_t>(0);
        dram->setGlobalCycle(global_cycle);
        gbuffer->setGlobalCycle(global_cycle);
        abuffer->setGlobalCycle(global_cycle);
        wbuffer->setGlobalCycle(global_cycle);
        obuffer->setGlobalCycle(global_cycle);
        arch->setGlobalCycle(global_cycle);

        if(!QUIET) std::cout << "Starting cycles simulation for architecture " << arch->name() << std::endl;

        // Initialize statistics
        std::string filename = arch->name() + arch->filename() + "_cycles";

        auto images = this->FAST_MODE ? 1 : network.getImages();
        sys::Stats stats = sys::Stats(network.getNumLayers(), images, filename);

        // Architecture stats
        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto compute_cycles = stats.register_uint_t("compute_cycles", 0, sys::AverageTotal);
        auto compute_stall_cycles = stats.register_uint_t("compute stall cycles", 0, sys::AverageTotal);
        auto scheduled_pe = stats.register_uint_t("scheduled PEs", 0, sys::AverageTotal);
        auto idle_pe = stats.register_uint_t("idle PEs", 0, sys::AverageTotal);

        // DRAM stats
        auto dram_act_reads = stats.register_uint_t("dram_act_reads", 0, sys::AverageTotal);
        auto dram_wgt_reads = stats.register_uint_t("dram_wgt_reads", 0, sys::AverageTotal);
        auto dram_out_writes = stats.register_uint_t("dram_out_writes", 0, sys::AverageTotal);
        auto dram_stall_cycles = stats.register_uint_t("dram_stall_cycles", 0, sys::AverageTotal);

        // Global Buffer stats
        auto gbuffer_act_reads = stats.register_uint_t("gbuffer_act_reads", 0, sys::AverageTotal);
        auto gbuffer_wgt_reads = stats.register_uint_t("gbuffer_wgt_reads", 0, sys::AverageTotal);
        auto gbuffer_out_writes = stats.register_uint_t("gbuffer_out_writes", 0, sys::AverageTotal);
        auto gbuffer_act_bank_conflicts = stats.register_uint_t("gbuffer_act_bank_conflicts", 0, sys::AverageTotal);
        auto gbuffer_wgt_bank_conflicts = stats.register_uint_t("gbuffer_wgt_bank_conflicts", 0, sys::AverageTotal);
        auto gbuffer_out_bank_conflicts = stats.register_uint_t("gbuffer_out_bank_conflicts", 0, sys::AverageTotal);

        // Local buffers
        auto on_chip_stall_cycles = stats.register_uint_t("on_chip_stall_cycles", 0, sys::AverageTotal);

        auto act_precision = stats.register_uint_t("activations precision", 0, sys::Average);
        auto wgt_precision = stats.register_uint_t("weights precision", 0, sys::Average);

        // Iterate over the images
        for(auto image = 0; image < images; ++image) {

            // Iterate over the layers
            for (auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

                const base::Layer<T> &layer = network.getLayers()[layer_it];
                bool conv = layer.getType() == "Convolution";
                bool rnn = layer.getType() == "RNN";
                bool fc = layer.getType() == "InnerProduct";

                if (!QUIET) printf("Simulating image: %d/%lu for layer: %s\n", image + 1, images,
                        layer.getName().c_str());

                if (conv) continue;

                auto act = std::make_shared<base::Array<T>>(layer.getActivations());
                arch->dataConversion(*act, layer.getActPrecision());
                if (fc && act->getDimensions() == 4) act->reshape_to_2D();
                if (act->getDimensions() == 2) act->reshape_to_4D();
                act->get_image(image);

                auto wgt = std::make_shared<base::Array<T>>(layer.getWeights());
                arch->dataConversion(*wgt, layer.getWgtPrecision());
                if (wgt->getDimensions() == 2) wgt->reshape_to_4D();

                int padding = layer.getPadding();
                int stride = layer.getStride();

                if (conv) act->zero_pad(padding);

                if (act->getShape()[1] == 3 && stride > 1) {
                    act->reshape_first_layer_act(stride);
                    wgt->reshape_first_layer_wgt(stride);
                    stride = 1;
                }

                const std::vector<size_t> &act_shape = act->getShape();
                const std::vector<size_t> &wgt_shape = wgt->getShape();

                uint64_t Nx, Ny;
                if (rnn) {
                    Nx = 1;
                    Ny = 1;
                } else {
                    Nx = act_shape[2];
                    Ny = act_shape[3];
                }

                auto num_filters = wgt_shape[0];
                auto Kx = wgt_shape[2];
                auto Ky = wgt_shape[3];

                auto Ox = (Nx - Kx) / stride + 1;
                auto Oy = (Ny - Ky) / stride + 1;

                auto act_prec = layer.getActPrecision();
                auto wgt_prec = layer.getWgtPrecision();
                control->configure_layer(act, wgt, act_prec, wgt_prec, fc || rnn, rnn, stride);

                OutputTensor sim_output = OutputTensor(num_filters, std::vector<std::vector<double>>(Ox,
                        std::vector<double>(Oy, 0)));

                *global_cycle = 0;
                auto tiles_data = std::vector<TileData<T>>(arch->getTiles(), TileData<T>());
                do {

                    // Feed off-chip data
                    gbuffer->evict_data(control->getIfEvictAct(), control->getIfEvictWgt());
                    dram->read_data(control->getReadActAddresses(), control->getReadWgtAddresses());

                    bool still_data = control->still_on_chip_data(tiles_data);
                    while(still_data) {

                        // Check if data is on-chip
                        dram->read_request(tiles_data);

                        // Wait for data to be on-chip
                        while (!dram->data_ready(tiles_data)) {
                            dram->cycle();
                            *global_cycle += 1;
                        }

                        // Request data to on-chip global buffer
                        gbuffer->act_read_request(tiles_data, abuffer->getFifoReadyCycle());
                        gbuffer->wgt_read_request(tiles_data, wbuffer->getFifoReadyCycle());

                        // Request data to local buffers
                        abuffer->read_request(gbuffer->getActReadReadyCycle());
                        wbuffer->read_request(gbuffer->getWgtReadReadyCycle());

                        // Wait for:
                        // - global buffer to write before starting new windows (psum registers are empty)
                        // - activation buffer to have the data ready
                        // - weight buffer to have the data ready
                        // - pipeline to be ready
                        while (!gbuffer->write_done() || !abuffer->data_ready() || !wbuffer->data_ready()
                                || !arch->ready()) {
                            dram->cycle();
                            *global_cycle += 1;
                        }

                        arch->process_tiles(tiles_data);

                        abuffer->evict_data();
                        wbuffer->evict_data();

                        dram->cycle();
                        *global_cycle += 1;

                        abuffer->update_fifo();
                        wbuffer->update_fifo();

                        // Check if write the output register back to global buffer
                        if (control->check_if_write_output(tiles_data)) {

                            // Flush pipeline
                            while (!arch->flush()) {
                                dram->cycle();
                                *global_cycle += 1;
                            }

                            gbuffer->write_request(tiles_data);

                        }

                        if (this->CHECK) calculate_output(sim_output, tiles_data);
                        still_data = control->still_on_chip_data(tiles_data);

                    }

                    // Wait for:
                    // - Pipeline is empty
                    // - All write petitions are fulfilled
                    while (!arch->flush() || !gbuffer->write_done()) {
                        dram->cycle();
                        *global_cycle += 1;
                    }

                    // Write values to DRAM (if necessary)

                } while(control->still_off_chip_data());

                if (CHECK) check_result(sim_output, act, wgt, Ox, Oy, stride, rnn);

                // Dump stats
                cycles->value[layer_it][image] = *global_cycle;

                compute_cycles->value[layer_it][image] = arch->getCycles();
                compute_stall_cycles->value[layer_it][image] = arch->getStallCycles();
                scheduled_pe->value[layer_it][image] = arch->getScheduledPe();
                idle_pe->value[layer_it][image] = arch->getIdlePe();

                dram_act_reads->value[layer_it][image] = dram->getActReads();
                dram_wgt_reads->value[layer_it][image] = dram->getWgtReads();
                dram_out_writes->value[layer_it][image] = dram->getOutWrites();
                dram_stall_cycles->value[layer_it][image] = dram->getStallCycles();

                act_precision->value[layer_it][image] = act_prec;
                wgt_precision->value[layer_it][image] = wgt_prec;

            } // Layer

        } // Image

        //Dump statistics
        std::string header = arch->name() + " Number of Cycles for " + network.getName() + "\n";
        header += "Dataflow: " + control->dataflow() + "\n";
        // DRAM header
        // Global buffer header
        header += arch->header();

        stats.dump_csv(network.getName(), network.getLayersName(), header, QUIET);
    }

    /* POTENTIALS */

    template <typename T>
    void Simulator<T>::potentials(const base::Network<T> &network, const std::shared_ptr<Architecture<T>> &arch) {

        if(!QUIET) std::cout << "Starting potentials simulation for architecture " << arch->name() << std::endl;

        // Initialize statistics
        std::string filename = arch->name() + arch->filename_pot() + "_potentials";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getImages(), filename);

        auto work_reduction = stats.register_double_t("work_reduction", 0, sys::Special);
        auto speedup = stats.register_double_t("speedup", 0, sys::Special);
        auto bit_mult = stats.register_uint_t("bit_multiplications", 0, sys::AverageTotal);
        auto max_bit_mult = stats.register_uint_t("max_bit_multiplications", 0, sys::AverageTotal);
        auto max_par_mult = stats.register_double_t("max_parallel_multiplication", 0, sys::AverageTotal);
        auto act_precision = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_precision = stats.register_uint_t("weights_precision", 0, sys::Average);

        auto network_width = network.getNetworkWidth();
        double MAX_BITS = network_width * network_width;
        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool rnn = layer.getType() == "RNN";
            bool fc = layer.getType() == "InnerProduct";

            if (!QUIET) std::cout << "Simulating layer: " << layer.getName() << std::endl;

            base::Array<T> act = layer.getActivations();
            arch->dataConversion(act, layer.getActPrecision());
            if (fc && act.getDimensions() == 4) act.reshape_to_2D();
            if (act.getDimensions() == 2) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            arch->dataConversion(wgt, layer.getWgtPrecision());
            if (wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            uint64_t images, act_channels, Nx, Ny, R;
            images = act_shape[0];
            if (rnn) {
                R = act_shape[1];
                act_channels = act_shape[2];
                Nx = 1;
                Ny = 1;
            } else {
                R = 1;
                act_channels = act_shape[1];
                Nx = act_shape[2];
                Ny = act_shape[3];
            }
            if (FAST_MODE) images = 1;

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

            arch->configure_layer(act_prec, wgt_prec, network_width, fc || rnn, arch->getColumns());

            for(int n = 0; n < images; ++n) {

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
                                            bit_counter += arch->computeBits(act_bits, wgt_bits);
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
                                T act_bits = rnn ? act.get(n, r, k) : act.get(n, k);
                                T wgt_bits = wgt.get(m, k);
                                bit_counter += arch->computeBits(act_bits, wgt_bits);
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

    INITIALISE_DATA_TYPES(Simulator);

}
