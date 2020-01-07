
#include <core/SCNN.h>

namespace core {

    /* CYCLES */

    template <typename T>
    int SCNN<T>::map_accumulator(uint32_t k, uint32_t x, uint32_t y) {
        return ((((k & 4u) << 2u) ^ ((x & 2u) << 3u) ^ ((y & 2u) << 3u)) + ((x & 1u) << 3u) + ((y & 1u) << 2u) +
                (k & 3u)) % BANKS;
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

                        const PE_stats &pe_stats = computeSCNNPE(W, H, stride, act_queue[sx][sy], wgt_queue[sx][sy]);

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

    template <typename T>
    void SCNN<T>::run(const base::Network<T> &network) {

        if(!QUIET) std::cout << "Starting cycles simulation for architecture SCNN" << std::endl;

        // Initialize statistics
        std::string filename = "SCNN_Wt" + std::to_string(Wt) + "_Ht" + std::to_string(Ht) + "_I" + std::to_string(I) +
                "_F" + std::to_string(F) + "_acc_out" + std::to_string(OUT_ACC_SIZE) + "_B" + std::to_string(BANKS) +
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

            if (!QUIET) std::cout << "Simulating layer: " << layer.getName() << std::endl;

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
            auto Kc = (int)floor(OUT_ACC_SIZE/(double)(th*tw));

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
                        auto tile_stats = computeSCNNTile(n, ct, ck, kc, tw, th, X, Y, Kc, K, W, H, R, S, stride,
                                padding, act, wgt);

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
        header += "Output accumulator size: " + std::to_string(OUT_ACC_SIZE) + "\n";
        header += "Number of banks: " + std::to_string(BANKS) + "\n";

        stats.dump_csv(network.getName(), network.getLayersName(), header, this->QUIET);

    }

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    std::string SCNN<T>::name() {
        return "SCNN";
    }

    /* CYCLES (NOT USED) */

    template <typename T>
    std::string SCNN<T>::filename() {
        throw std::runtime_error("SCNN simulation is not implemented by Inference module");
    }

    template <typename T>
    std::string SCNN<T>::header() {
        throw std::runtime_error("SCNN simulation is not implemented by Inference module");
    }

    template <typename T>
    bool SCNN<T>::diffy() {
        throw std::runtime_error("SCNN simulation is not implemented by Inference module");
    }

    template <typename T>
    bool SCNN<T>::schedule() {
        throw std::runtime_error("SCNN simulation is not implemented by Inference module");
    }

    template <typename T>
    void SCNN<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {
        throw std::runtime_error("SCNN simulation is not implemented by Inference module");
    }

    /* POTENTIALS */

    template <typename T>
    std::string SCNN<T>::filename_pot() {
        return "";
    }

    template <typename T>
    std::string SCNN<T>::header_pot() {
        return "";
    }

    template <typename T>
    uint16_t SCNN<T>::computeBits(T act, T wgt) {
        if(wgt == 0) return 0;
        else if(act == 0) return 0;
        else return this->network_bits * this->network_bits;
    }

    INITIALISE_DATA_TYPES(SCNN);

}