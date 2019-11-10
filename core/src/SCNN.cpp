
#include <core/SCNN.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    int SCNN<T>::map_accumulator(uint32_t k, uint32_t x, uint32_t y) {
        return ((((k & 4u) << 2u) ^ ((x & 2u) << 3u) ^ ((y & 2u) << 3u)) + ((x & 1u) << 3u) + ((y & 1u) << 2u) +
                (k & 3u)) % BANKS;
    }

    template <typename T>
    uint16_t SCNN<T>::computeSCNNBitsPE(T act, T wgt, const int network_bits) {

        #ifdef ZERO_COUNT
        if(wgt == 0) return 1;
        else if(act == 0) return 1;
        #else
        if(wgt == 0) return 0;
        else if(act == 0) return 0;
        #endif
        else return (uint16_t)(network_bits*network_bits);
    }

    template <typename T>
    typename SCNN<T>::PE_stats SCNN<T>::computeSCNNPE(uint64_t W, uint64_t H, int stride, const act_idxMap &act,
            const wgt_idxMap &wgt) {

        PE_stats pe_stats;
        pe_stats.cycles = 0;
        pe_stats.mults = 0;
        pe_stats.idle_conflicts = 0;
        pe_stats.accumulator_updates = 0;
        pe_stats.i_loop = 0;
        pe_stats.f_loop = 0;

        for(int i = 0; i < act.size(); i+=I) {
            pe_stats.i_loop += 1;
            for(int f = 0; f < wgt.size(); f+=F) {
                pe_stats.f_loop += 1;
                std::vector<uint8_t> acc(2 * F * I, 0);
                for(int ii = i; ii < std::min(i + (int)I, (int)act.size()); ii++) {
                    for(int ff = f; ff < std::min(f + (int)F, (int)wgt.size()); ff++) {
                        const auto &act_index = act[ii];
                        const auto &wgt_index = wgt[ff];

                        auto x = std::get<0>(act_index);
                        auto y = std::get<1>(act_index);

                        auto k = std::get<0>(wgt_index);
                        auto r = std::get<1>(wgt_index);
                        auto s = std::get<2>(wgt_index);

                        int w = (x - r) / stride;
                        int h = (y - s) / stride;

                        if(w >= 0 && w < W && h >= 0 && h < H) {
                            int acc_idx = map_accumulator(k, w, h);
                            acc[acc_idx] += 1;

                            pe_stats.mults += 1;
                        }
                    }
                }
                auto max_acc = *std::max_element(acc.begin(), acc.end());
                auto warp_time = std::max(max_acc, (uint8_t) 1);
                pe_stats.cycles += warp_time;
                pe_stats.idle_conflicts += (warp_time - 1) * I * F;
                pe_stats.accumulator_updates += accumulate(acc.begin(), acc.end(), 0.0);
            }
        }

        return pe_stats;
    }

    template <typename T>
    typename SCNN<T>::Tile_stats SCNN<T>::computeSCNNTile(int n, int ct, int ck, int kc, int tw, int th, uint64_t X,
            uint64_t Y, int Kc, uint64_t K, uint64_t W, uint64_t H, uint64_t R, uint64_t S, int stride, int padding,
            const base::Array<T> &act, const base::Array<T> &wgt) {

        Tile_stats tile_stats;
        tile_stats.cycles = 0;
        tile_stats.dense_cycles = 0;
        tile_stats.mults = 0;
        tile_stats.idle_bricks = 0;
        tile_stats.idle_conflicts = 0;
        tile_stats.idle_pe = 0;
        tile_stats.weight_buff_reads = 0;
        tile_stats.act_buff_reads = 0;
        tile_stats.accumulator_updates = 0;
        tile_stats.i_loop = 0;
        tile_stats.f_loop = 0;
        tile_stats.offchip_weight_reads = 0;

        std::vector<uint32_t> tile_cycles;
        std::vector<uint32_t> tile_dense_cycles;
        std::vector<uint32_t> tile_i_loop;
        uint32_t wgt_size = 0;

        for(int pex = 0; pex < Wt; pex++) {
            for(int pey = 0; pey < Ht; pey++) {
                int x_begin = pex * tw, y_begin = pey * th, k_begin = kc;
                int x_end = std::min(x_begin + tw, (int)X), y_end = std::min(y_begin + th, (int)Y),
                        k_end = std::min(kc + Kc, (int)K);

                std::vector<std::vector<act_idxMap>> act_queue = std::vector<std::vector<act_idxMap>>((unsigned)stride,
                        std::vector<act_idxMap>((unsigned)stride,act_idxMap()));
                std::vector<std::vector<uint16_t>> dense_act_counter = std::vector<std::vector<uint16_t>>(
                        (unsigned)stride, std::vector<uint16_t>((unsigned)stride,0));
                for(int x = x_begin; x < x_end; x++) {
                    int sx = x % stride;
                    for(int y = y_begin; y < y_end; y++) {
                        int sy = y % stride;
                        auto act_bits = act.get(n,ct+ck,x,y);
                        if(act_bits != 0)
                            act_queue[sx][sy].emplace_back(std::make_tuple(x,y));
                        dense_act_counter[sx][sy] += 1;
                    }
                }

                std::vector<std::vector<wgt_idxMap>> wgt_queue = std::vector<std::vector<wgt_idxMap>>((unsigned)stride,
                        std::vector<wgt_idxMap>((unsigned)stride,wgt_idxMap()));
                std::vector<std::vector<uint32_t>> dense_wgt_counter = std::vector<std::vector<uint32_t>>(
                        (unsigned)stride, std::vector<uint32_t>((unsigned)stride,0));
                for(int r = 0; r < R; r++) {
                    int sx = (r + padding) % stride;
                    for(int s = 0; s < S; s++) {
                        int sy = (s + padding) % stride;
                        for(int k = k_begin; k < k_end; k++) {
                            auto wgt_bits = wgt.get(k,ck,r,s);
                            if(wgt_bits != 0)
                                wgt_queue[sx][sy].emplace_back(std::make_tuple(k,r,s));
                            dense_wgt_counter[sx][sy] += 1;
                        }
                    }
                }

                uint32_t PE_cycles = 0;
                uint32_t PE_dense_cycles = 0;
                uint32_t PE_mults = 0;
                uint32_t PE_idle_conflicts = 0;
                uint32_t PE_accumulator_updates = 0;
                uint32_t PE_i_loop = 0;
                uint32_t PE_f_loop = 0;
                uint32_t PE_wgt_size = 0;

                for(int sx = 0; sx < stride; sx++) {
                    for(int sy = 0; sy < stride; sy++) {

                        const PE_stats &pe_stats = computeSCNNPE(W,H,stride,act_queue[sx][sy],wgt_queue[sx][sy]);

                        auto stride_wgt_size = (uint32_t)(ceil(wgt_queue[sx][sy].size()/(double)F))*F;
                        PE_wgt_size += stride_wgt_size;

                        PE_cycles += pe_stats.cycles;
                        PE_dense_cycles += (uint32_t)(ceil(dense_act_counter[sx][sy]/(double)I) *
                                ceil(dense_wgt_counter[sx][sy]/(double)F));
                        PE_mults += pe_stats.mults;
                        PE_idle_conflicts += pe_stats.idle_conflicts;
                        PE_accumulator_updates += pe_stats.accumulator_updates;
                        PE_i_loop += pe_stats.i_loop;
                        PE_f_loop += pe_stats.f_loop;

                        tile_stats.weight_buff_reads += stride_wgt_size;
                        tile_stats.act_buff_reads += (uint64_t)(ceil(act_queue[sx][sy].size()/(double)I))*I;
                    }
                }
                wgt_size = PE_wgt_size;
                tile_cycles.push_back(PE_cycles);
                tile_dense_cycles.push_back(PE_dense_cycles);
                tile_i_loop.push_back(PE_i_loop);

                tile_stats.idle_bricks += PE_f_loop * I * F - PE_mults;
                tile_stats.mults += PE_mults;
                tile_stats.idle_conflicts += PE_idle_conflicts;
                tile_stats.accumulator_updates += PE_accumulator_updates;
                tile_stats.i_loop += PE_i_loop;
                tile_stats.f_loop += PE_f_loop;
            }
        }

        auto tile_max_cycles = *std::max_element(tile_cycles.begin(), tile_cycles.end());
        uint32_t tile_idle_pe = 0;
        for(const auto &PE_cycles : tile_cycles)
            tile_idle_pe += tile_max_cycles - PE_cycles;
        auto tile_max_i_loop =  *std::max_element(tile_i_loop.begin(), tile_i_loop.end());

        tile_stats.cycles += tile_max_cycles;
        tile_stats.dense_cycles += *std::max_element(tile_dense_cycles.begin(), tile_dense_cycles.end());
        tile_stats.idle_pe += tile_idle_pe * I * F;
        tile_stats.offchip_weight_reads += tile_max_i_loop * wgt_size;

        return tile_stats;

    }

    /* CYCLES */

    template <typename T>
    void SCNN<T>::run(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "SCNN_Wt" + std::to_string(Wt) + "_Ht" + std::to_string(Ht) + "_I" + std::to_string(I) +
                "_F" + std::to_string(F) + "_acc_out" + std::to_string(out_acc_size) + "_B" + std::to_string(BANKS) +
                "_cycles";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto dense_cycles = stats.register_uint_t("dense_cycles", 0, sys::AverageTotal);
        auto mults = stats.register_uint_t("mults", 0, sys::AverageTotal);
        auto idle_bricks = stats.register_uint_t("idle_bricks", 0, sys::AverageTotal);
        auto idle_conflicts = stats.register_uint_t("idle_conflicts", 0, sys::AverageTotal);
        auto idle_pe = stats.register_uint_t("idle_pe", 0, sys::AverageTotal);
        auto idle_halo = stats.register_uint_t("idle_halo", 0, sys::AverageTotal);
        auto total_mult_cycles = stats.register_uint_t("total_mult_cycles", 0, sys::AverageTotal);
        auto halo_transfers = stats.register_uint_t("halo_transfers", 0, sys::AverageTotal);
        auto weight_buff_reads = stats.register_uint_t("weight_buff_reads", 0, sys::AverageTotal);
        auto act_buff_reads = stats.register_uint_t("act_buff_reads", 0, sys::AverageTotal);
        auto accumulator_updates = stats.register_uint_t("accumulator_updates", 0, sys::AverageTotal);
        auto i_loop = stats.register_uint_t("i_loop", 0, sys::AverageTotal);
        auto f_loop = stats.register_uint_t("f_loop", 0, sys::AverageTotal);
        auto offchip_weight_reads = stats.register_uint_t("offchip_weight_reads", 0, sys::AverageTotal);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            if (lstm) continue;

            base::Array<T> act = layer.getActivations();
            base::Array<T> wgt = layer.getWeights();
            if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

            if(fc || act.getDimensions() == 2) {
                if(act.getDimensions() == 4) act.reshape_to_2D();
                act.reshape_to_4D();
                auto C = (int)act.getShape()[1];
                C = (int)(ceil(C/(double)256))*256;
                act.channel_zero_pad(C);
                act.split_4D(C / 256, 16, 16);

                auto Ck = (int)wgt.getShape()[1];
                Ck = (int)(ceil(Ck/(double)256))*256;
                wgt.channel_zero_pad(Ck);
                wgt.split_4D(Ck / 256, 16, 16);
            }

            int padding = layer.getPadding();
            int stride = layer.getStride();

            act.zero_pad(padding);

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            auto N = act_shape[0];
            auto C = act_shape[1];
            auto X = act_shape[2];
            auto Y = act_shape[3];
            if(this->FAST_MODE) N = 1;

            auto K = wgt_shape[0];
            auto Ck = wgt_shape[1];
            auto R = wgt_shape[2];
            auto S = wgt_shape[3];

            auto W = (X - R)/stride + 1;
            auto H = (Y - S)/stride + 1;

            auto groups = C / Ck;
            auto Kg = K / groups;

            auto W_round = (int)(ceil(W/(double)Wt))*Wt;
            auto H_round = (int)(ceil(H/(double)Ht))*Ht;
            auto tw = W_round/Wt;
            auto th = H_round/Ht;
            auto Kc = (int)floor(out_acc_size/(double)(th*tw));

            // Fix for MobileNet
            if(Ck == 1 && C != 1) Kc = 1;

            X = (int)(ceil(X/(double)Wt))*Wt;
            Y = (int)(ceil(Y/(double)Ht))*Ht;
            tw = (uint32_t)X/Wt;
            th = (uint32_t)Y/Wt;

            act.grid_zero_pad(X ,Y);

            for(int n = 0; n < N; n++) {
                for(int kc = 0; kc < K; kc += Kc) {

                    // Two towers alexnet
                    int ct = 0;
                    if(kc >= Kg) ct = (int)Ck;

                    // Fix for MobileNet
                    if(Ck == 1 && C != 1) ct = kc;

                    for(int ck = 0; ck < Ck; ck++) {
                        auto tile_stats = computeSCNNTile(n,ct,ck,kc,tw,th,X,Y,Kc,K,W,H,R,S,stride,padding,act,wgt);

                        cycles->value[layer_it][n] += tile_stats.cycles;
                        dense_cycles->value[layer_it][n] += tile_stats.dense_cycles;
                        mults->value[layer_it][n] += tile_stats.mults;
                        idle_bricks->value[layer_it][n] += tile_stats.idle_bricks;
                        idle_conflicts->value[layer_it][n] += tile_stats.idle_conflicts;
                        idle_pe->value[layer_it][n] += tile_stats.idle_pe;
                        weight_buff_reads->value[layer_it][n] += tile_stats.weight_buff_reads;
                        act_buff_reads->value[layer_it][n] += tile_stats.act_buff_reads;
                        accumulator_updates->value[layer_it][n] += tile_stats.accumulator_updates;
                        i_loop->value[layer_it][n] += tile_stats.i_loop;
                        f_loop->value[layer_it][n] += tile_stats.f_loop;
                        offchip_weight_reads->value[layer_it][n] += tile_stats.offchip_weight_reads;
                    }

                    // resolve halos
                    // compute the areas of the halo regions around a non edge PE
                    // that is, how many psums need to get transferred

                    const int DIM = 3;
                    int x_vec[] = {(int)R - 1 - padding, (int)tw, padding};
                    int y_vec[] = {(int)S - 1 - padding, (int)th, padding};
                    int max_psum = 0;
                    uint32_t batch_halo_transfers = 0;

                    for(int x = 0; x < DIM; x++) {
                        for (int y = 0; y < DIM; y++) {
                            int psum = x_vec[x] * y_vec[y];
                            if(x != 1 || y != 1)  {
                                batch_halo_transfers += psum;
                                if(psum > max_psum)
                                    max_psum = psum;
                            }
                        }
                    }
                    auto max_psums = max_psum * std::min(Kc, (int)K - kc);

                    cycles->value[layer_it][n] += max_psums;
                    dense_cycles->value[layer_it][n] += max_psums;
                    idle_halo->value[layer_it][n] += max_psums * Ht * Wt * I * F;
                    halo_transfers->value[layer_it][n] += batch_halo_transfers;
                }
                total_mult_cycles->value[layer_it][n] = mults->value[layer_it][n] + idle_bricks->value[layer_it][n] +
                        idle_conflicts->value[layer_it][n] + idle_pe->value[layer_it][n] + idle_halo->value[layer_it][n];
            }


        }

        //Dump statistics
        std::string header = "SCNN Number of Cycles for " + network.getName() + "\n";
        header += "Number of PE columns: " + std::to_string(Wt) + "\n";
        header += "Number of PE rows: " + std::to_string(Ht) + "\n";
        header += "Column multipliers per PE: " + std::to_string(I) + "\n";
        header += "Row multipliers per PE: " + std::to_string(F) + "\n";
        header += "Output accumulator size: " + std::to_string(out_acc_size) + "\n";
        header += "Number of banks: " + std::to_string(BANKS) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    /* POTENTIALS */

    template <typename T>
    void SCNN<T>::potentials(const base::Network<T> &network) {

        // Initialize statistics
        std::string filename = "SCNN_potentials";
        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto work_reduction = stats.register_double_t("work_reduction", 0, sys::Average);
        auto speedup = stats.register_double_t("speedup", 0, sys::Average);
        auto par_mult = stats.register_double_t("parallel_multiplication", 0, sys::AverageTotal);
        auto bit_multiplications = stats.register_uint_t("bit_multiplications", 0, sys::AverageTotal);
        auto act_prec = stats.register_uint_t("activations_precision", 0, sys::Average);
        auto wgt_prec = stats.register_uint_t("weights_precision", 0, sys::Average);

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            base::Array<T> act = layer.getActivations();
            if(fc && act.getDimensions() == 4) act.reshape_to_2D();

            base::Array<T> wgt = layer.getWeights();
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

            // Operations
            uint64_t parallel_mult = conv ? num_filters * out_x * out_y * Kx * Ky * wgt_channels :
                    num_filters * wgt_channels * R;

            for(int n = 0; n < batch_size; n++) {
                double MAX_BITS = network_bits * network_bits;
                uint64_t bit_counter = 0;

                if (conv) {

                    for(int m = 0; m<num_filters; m++) {

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
                                            bit_counter += computeSCNNBitsPE(act.get(n, start_group + k, stride * x + i,
                                                    stride * y + j), wgt.get(m, k, i, j), network_bits);
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
                                auto act_bits = lstm ? act.get(r, n, k) : act.get(n, k);
                                bit_counter += computeSCNNBitsPE(act_bits, wgt.get(m, k), network_bits);
                            }
                        }
                    }

                }

                bit_multiplications->value[layer_it][n] = bit_counter;
                work_reduction->value[layer_it][n] = 100 - ((double)bit_counter / (double)parallel_mult / MAX_BITS * 100);
                speedup->value[layer_it][n] = (double)parallel_mult * MAX_BITS / (double)bit_counter;
                par_mult->value[layer_it][n] = parallel_mult;
                act_prec->value[layer_it][n] = layer.getActPrecision();
                wgt_prec->value[layer_it][n] = layer.getWgtPrecision();
            }

        }

        //Dump statistics
        std::string header = "SCNN Potentials/Work Reduction for " + network.getName() + "\n";
        #ifdef ZERO_COUNT
        header += "Zero count as one cycle\n";
        #endif

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    template <typename T>
    void SCNN<T>::on_chip_cycles(const base::Network<T> &network) {

        std::string filename = "SCNN_Wt" + std::to_string(Wt) + "_Ht" + std::to_string(Ht) + "_I" + std::to_string(I) +
                "_F" + std::to_string(F)  + "_AS" + std::to_string(this->memory.getOnChipActSize()) + "_WS" +
                std::to_string(this->memory.getOnChipWgtSize()) + "_B" + std::to_string(BANKS) +
                (BASELINE ? "_baseline" : "") + "_on_chip_cycles";

        sys::Stats stats = sys::Stats(network.getNumLayers(), this->FAST_MODE ? 1 : network.getBatches(), filename);

        auto cycles = stats.register_uint_t("cycles", 0, sys::AverageTotal);
        auto compute_cycles = stats.register_uint_t("compute_cycles", 0, sys::AverageTotal);
        auto memory_cycles = stats.register_uint_t("memory_cycles", 0, sys::AverageTotal);
        auto act_comp_size = stats.register_uint_t("act_comp bits", 0 , sys::AverageTotal);
        auto act_off_chip = stats.register_uint_t("act_off_chip bytes", 0 , sys::AverageTotal);
        auto act_off_chip_bytes = stats.register_uint_t("act_off_chip", 0 , sys::AverageTotal);
        auto wgt_off_chip = stats.register_uint_t("wgt_off_chip bytes", 0 , sys::AverageTotal);
        auto wgt_off_chip_bytes = stats.register_uint_t("wgt_off_chip", 0 , sys::AverageTotal);

        uint64_t act_next_addr = 0;
        uint64_t act_base_addr = 0x40000000;

        uint64_t wgt_next_addr = 0;
        uint64_t wgt_base_addr = 0x00000000;

        auto network_bits = network.getNetwork_bits();
        auto signed_activations = !network.isUnsignedAct();
        auto signed_weights = !network.isUnsignedWgt();
        auto values_block = 64 / network_bits;

        for(auto layer_it = 0; layer_it < network.getNumLayers(); ++layer_it) {

            if (layer_it != 0) signed_activations = false;

            const base::Layer<T> &layer = network.getLayers()[layer_it];
            bool conv = layer.getType() == "Convolution";
            bool lstm = layer.getType() == "LSTM";
            bool fc = layer.getType() == "InnerProduct";

            if (fc || lstm) continue;

            if (!this->QUIET) std::cout << layer.getName() << std::endl;

            base::Array<T> act = layer.getActivations();
            act.sign_magnitude_representation(layer.getActPrecision());
            if (act.getDimensions() == 2) act.reshape_to_4D();

            base::Array<T> wgt = layer.getWeights();
            wgt.sign_magnitude_representation(layer.getWgtPrecision());
            if (wgt.getDimensions() == 2) wgt.reshape_to_4D();

            int padding = layer.getPadding();
            int stride = layer.getStride();

            if (conv) act.zero_pad(padding);

            const std::vector<size_t> &act_shape = act.getShape();
            const std::vector<size_t> &wgt_shape = wgt.getShape();

            auto N = act_shape[0];
            auto C = act_shape[1];
            auto X = act_shape[2];
            auto Y = act_shape[3];
            if (this->FAST_MODE) N = 1;

            auto K = wgt_shape[0];
            auto Ck = wgt_shape[1];
            auto R = wgt_shape[2];
            auto S = wgt_shape[3];

            auto W = (X - R) / stride + 1;
            auto H = (Y - S) / stride + 1;

            auto groups = C / Ck;
            auto Kg = K / groups;

            auto W_round = (int) (ceil(W / (double) Wt)) * Wt;
            auto H_round = (int) (ceil(H / (double) Ht)) * Ht;
            auto tw = W_round / Wt;
            auto th = H_round / Ht;
            auto Kc = (int) floor(out_acc_size / (double) (th * tw));
            Kc = std::min(Kc, (int)Kg);

            X = (int) (ceil(X / (double) Wt)) * Wt;
            Y = (int) (ceil(Y / (double) Ht)) * Ht;
            tw = (uint32_t) X / Wt;
            th = (uint32_t) Y / Wt;

            act.grid_zero_pad(X, Y);

            // Off-chip memory layout
            spat_address_map act_address_map = spat_address_map(N, std::vector<std::vector<uint64_t>>(C,
                    std::vector<uint64_t>(ceil(X * Y /(double) values_block))));

            // Image third
            for (int n = 0; n < N; ++n) {

                // Channel second
                for (int ch = 0; ch < C; ++ch) {

                    // Spatial first
                    for (int xy = 0; xy < (X * Y); ++xy) {
                        act_address_map[n][ch][xy / values_block] = act_base_addr + act_next_addr;
                        act_next_addr += 0x40; // Align to 64 bits
                    }
                }
            }

            spat_address_map wgt_address_map = spat_address_map(K, std::vector<std::vector<uint64_t>>(Ck,
                    std::vector<uint64_t>(ceil(R * S / (double) values_block))));

            // Filter third
            for (int k = 0; k < K; ++k) {

                // Channel second
                for (int ch = 0; ch < Ck; ++ch) {

                    // Spatial first
                    for (int xy = 0; xy < (R * S); ++xy) {
                        wgt_address_map[k][ch][xy / values_block] = wgt_base_addr + wgt_next_addr;
                        wgt_next_addr += 0x40; // Align to 64 bits
                    }
                }
            }

            auto act_layer_prec = layer.getActPrecision();
            auto act_mask = (uint16_t) (1u << (act_layer_prec - 1));

            auto wgt_layer_prec = layer.getWgtPrecision();
            auto wgt_mask = (uint16_t) (1u << (wgt_layer_prec - 1));

            // Pre-allocate wgt queues
            wgt_addr_queue wgt_queue (ceil(K/(double)Kc), std::vector<std::vector<std::vector<wgt_idxAddrMap>>>(Ck,
                    std::vector<std::vector<wgt_idxAddrMap>>(stride, std::vector<wgt_idxAddrMap>(stride,
                    wgt_idxAddrMap()))));

            std::vector<uint64_t> wgt_addr;

            for(int kc = 0; kc < K; kc += Kc) {

                int k_begin = kc;
                int k_end = std::min(kc + Kc, (int)K);

                for(int ck = 0; ck < Ck; ck++) {

                    for(int r = 0; r < R; r++) {
                        int sx = (r + padding) % stride;
                        for(int s = 0; s < S; s++) {
                            int sy = (s + padding) % stride;
                            for(int k = k_begin; k < k_end; k++) {
                                auto wgt_bits = wgt.get(k, ck, r, s);
                                if(wgt_bits != 0) {
                                    auto wgt_addr = wgt_address_map[k][ck][(r * S + s) / values_block];
                                    wgt_queue[kc / Kc][ck][sx][sy].emplace_back(std::make_tuple(k, r, s, wgt_addr));
                                }
                            } // Filters
                        } // Y
                    } // X

                } // Channels
            } // Filter sets

            for(int n = 0; n < N; n++) {

                // Pre-allocate act queues
                act_addr_queue act_queue = act_addr_queue(C,
                        std::vector<std::vector<std::vector<std::vector<act_idxAddrMap>>>>(stride,
                        std::vector<std::vector<std::vector<act_idxAddrMap>>>(stride,
                        std::vector<std::vector<act_idxAddrMap>>(Wt, std::vector<act_idxAddrMap>(Ht,
                        act_idxAddrMap())))));

                std::vector<std::vector<std::vector<std::vector<std::vector<uint64_t>>>>> queue_size (C,
                        std::vector<std::vector<std::vector<std::vector<uint64_t>>>>(stride,
                        std::vector<std::vector<std::vector<uint64_t>>>(stride, std::vector<std::vector<uint64_t>>(Wt,
                        std::vector<uint64_t>(Ht, 0)))));

                for (int c = 0; c < C; c++) {

                    for(int pex = 0; pex < Wt; pex++) {
                        for(int pey = 0; pey < Ht; pey++) {
                            int x_begin = pex * tw, y_begin = pey * th;
                            int x_end = std::min(x_begin + (int)tw, (int)X),
                                    y_end = std::min(y_begin + (int)th, (int)Y);

                            for(int x = x_begin; x < x_end; x++) {
                                int sx = x % stride;
                                for(int y = y_begin; y < y_end; y++) {
                                    int sy = y % stride;
                                    auto act_bits = act.get(n, c, x, y);
                                    if(act_bits != 0) {
                                        auto act_addr = act_address_map[n][c][(x * Y + y) / values_block];
                                        act_queue[c][sx][sy][pex][pey].emplace_back(std::make_tuple(x, y, act_bits, act_addr));
                                        queue_size[c][sx][sy][pex][pey]++;
                                    }
                                } // Y
                            } // X

                        } // PE Y
                    } // PE X

                } // Channels

                // Calculate act sizes
                std::vector<std::vector<std::vector<uint64_t>>> act_queue_bits (C, std::vector<std::vector<uint64_t>>(
                        stride, std::vector<uint64_t>(stride, 0)));

                for (int c = 0; c < C; c++) {

                    for (int sx = 0; sx < stride; ++sx) {
                        for (int sy = 0; sy < stride; ++sy) {
                            for(int pex = 0; pex < Wt; pex++) {
                                for(int pey = 0; pey < Ht; pey++) {

                                    const auto &act_queue_pe = act_queue[c][sx][sy][pex][pey];

                                    for (int i = 0; i < act_queue_pe.size(); i += I) {

                                        uint8_t max_bit = 0;
                                        for(int ii = i; ii < std::min(i + (int)I, (int)act_queue_pe.size()); ii++) {
                                            auto act_bits = std::get<2>(act_queue_pe[ii]);

                                            if (signed_activations) {
                                                if ((act_bits & act_mask) != 0) {
                                                    act_bits = act_bits & ~act_mask;
                                                }
                                            }

                                            const auto &min_max_act_bits = this->minMax(act_bits);
                                            auto max_act_bit = std::get<1>(min_max_act_bits);
                                            if (signed_activations) max_act_bit += 1;

                                            if (max_act_bit > max_bit) max_bit = max_act_bit;
                                        }
                                        uint64_t width = max_bit + 1u;
                                        act_queue_bits[c][sx][sy] += 4; // zero overhead
                                        if (BASELINE) {
                                            act_queue_bits[c][sx][sy] += I * network_bits;
                                        } else {
                                            act_queue_bits[c][sx][sy] += log2(network_bits); // width overhead
                                            act_queue_bits[c][sx][sy] += I * width;
                                        }

                                    }

                                } // PE Y
                            } // PE X
                        } // Stride Y
                    } // Stride X

                } // Channels

                // Act on-chip at the same time
                uint64_t act_size = 0;
                std::vector<std::tuple<int,int,int>> act_set;
                std::vector<std::vector<std::tuple<int,int,int>>> act_sets;
                for (int c = 0; c < C; c++) {
                    for (int sx = 0; sx < stride; ++sx) {
                        for (int sy = 0; sy < stride; ++sy) {

                            uint64_t channel_size = ceil(act_queue_bits[c][sx][sy] / 8.);
                            if ((act_size + channel_size) < this->memory.getOnChipActSize()) {
                                act_set.emplace_back(std::make_tuple(c, sx, sy));
                                act_size += channel_size;
                            } else {
                                assert(act_size != 0);
                                auto tmp = act_set;
                                act_set.clear();
                                act_size = 0;

                                act_sets.emplace_back(tmp);
                                act_set.emplace_back(std::make_tuple(c, sx, sy));
                                act_size += channel_size;
                            }

                            if (c == (C - 1) && sx == (stride - 1) && sy == (stride - 1)) {
                                act_sets.emplace_back(act_set);
                                act_set.clear();
                            }

                        }
                    }
                }

                for(int kc = 0; kc < K; kc += Kc) {

                    // Request weights on-chip

                    // Two towers alexnet
                    int ct = 0;
                    if(kc >= Kg) ct = (int)Ck;

                    int prev_c = 0;
                    std::vector<uint64_t> pe_cycles (Wt * Ht, 0);
                    for (const auto &act_on_chip : act_sets) {

                        // Request activations on-chip

                        for (const auto &act_ch_str : act_on_chip) {

                            int c = std::get<0>(act_ch_str);
                            int sx = std::get<1>(act_ch_str);
                            int sy = std::get<2>(act_ch_str);
                            const auto &act_queue_pes = act_queue[c][sx][sy];

                            if (prev_c != c) {
                                auto tile_cycles = sys::get_max(pe_cycles);
                                compute_cycles->value[layer_it][n] += tile_cycles;
                                pe_cycles = std::vector<uint64_t>(Wt * Ht, 0);
                            }

                            int ck = c;
                            if(kc >= Kg)
                                ck = c % Ck;

                            const auto &wgt_queue_pe = wgt_queue[kc / Kc][ck][sx][sy];
                            auto act_queue_size = sys::get_max(queue_size[ct + ck][sx][sy]);

                            for (int i = 0; i < act_queue_size; i += I) {
                                for (int f = 0; f < wgt_queue_pe.size(); f += F) {

                                    for (int pex = 0; pex < Wt; pex++) {
                                        for (int pey = 0; pey < Ht; pey++) {

                                            const auto &act_queue_pe = act_queue[ct + ck][sx][sy][pex][pey];

                                            std::vector<uint8_t> acc(2 * F * I, 0);
                                            for (int ii = i;
                                                 ii < std::min(i + (int) I, (int) act_queue_pe.size()); ii++) {
                                                for (int ff = f;
                                                     ff < std::min(f + (int) F, (int) wgt_queue_pe.size()); ff++) {
                                                    const auto &act_index = act_queue_pe[ii];
                                                    const auto &wgt_index = wgt_queue_pe[ff];

                                                    auto x = std::get<0>(act_index);
                                                    auto y = std::get<1>(act_index);
                                                    auto act_addr = std::get<3>(act_index);

                                                    auto k = std::get<0>(wgt_index);
                                                    auto r = std::get<1>(wgt_index);
                                                    auto s = std::get<2>(wgt_index);
                                                    auto wgt_addr = std::get<3>(wgt_index);

                                                    int w = (x - r) / stride;
                                                    int h = (y - s) / stride;

                                                    if (w >= 0 && w < W && h >= 0 && h < H) {
                                                        int acc_idx = map_accumulator(k, w, h);
                                                        acc[acc_idx] += 1;
                                                    }
                                                } // Inner prod Wgt
                                            } // Inner prod Act
                                            auto max_acc = *std::max_element(acc.begin(), acc.end());
                                            auto warp_time = std::max(max_acc, (uint8_t) 1);
                                            pe_cycles[pey * Wt + pex] += std::max(warp_time, (uint8_t) 1);

                                        } // PE Y
                                    } // PE X

                                } // Queue Wgt
                            } // Queue Act

                            prev_c = c;
                        }
                    }

                    // resolve halos
                    // compute the areas of the halo regions around a non edge PE
                    // that is, how many psums need to get transferred

                    const int DIM = 3;
                    int x_vec[] = {(int)R - 1 - padding, (int)tw, padding};
                    int y_vec[] = {(int)S - 1 - padding, (int)th, padding};
                    int max_psum = 0;
                    uint32_t batch_halo_transfers = 0;

                    for(int x = 0; x < DIM; x++) {
                        for (int y = 0; y < DIM; y++) {
                            int psum = x_vec[x] * y_vec[y];
                            if(x != 1 || y != 1)  {
                                batch_halo_transfers += psum;
                                if(psum > max_psum)
                                    max_psum = psum;
                            }
                        }
                    }
                    auto max_psums = max_psum * std::min(Kc, (int)K - kc);
                    compute_cycles->value[layer_it][n] += max_psums;
                }
            } // Filter on-chip

        }

        //Dump statistics
        std::string header = "SCNN On-Chip Number of Cycles for " + network.getName() + "\n";
        header += "Number of PE columns: " + std::to_string(Wt) + "\n";
        header += "Number of PE rows: " + std::to_string(Ht) + "\n";
        header += "Column multipliers per PE: " + std::to_string(I) + "\n";
        header += "Row multipliers per PE: " + std::to_string(F) + "\n";
        header += "Output accumulator size: " + std::to_string(out_acc_size) + "\n";
        header += "Number of banks: " + std::to_string(BANKS) + "\n";
        header += "On-chip activations size: " + std::to_string(this->memory.getOnChipActSize()) + "\n";
        header += "On-chip weights size: " + std::to_string(this->memory.getOnChipWgtSize()) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }



    INITIALISE_DATA_TYPES(SCNN);

}