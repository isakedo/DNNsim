
#include <core/SCNN.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    int SCNN<T>::map_accumulator(int k, int x, int y, int banks) {
        return ((((k & 4) << 2) ^ ((x & 2) << 3) ^ ((y & 2) << 3)) + ((x & 1) << 3) + ((y & 1) << 2) + (k & 3)) % banks;
    }

    template <typename T>
    uint16_t SCNN<T>::computeSCNNBitsPE(T act, T wgt) {

        #ifdef ZERO_COUNT
        if(wgt == 0) return 1;
        else if(act == 0) return 1;
        #else
        if(wgt == 0) return 0;
        else if(act == 0) return 0;
        #endif
        else return 256;
    }

    template <typename T>
    typename SCNN<T>::PE_stats SCNN<T>::computeSCNNPE(int W, int H, int stride, int padding, const act_idxMap &act,
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
                for(int ii = i; ii < std::min(i + I, (int)act.size()); ii++) {
                    for(int ff = f; ff < std::min(f + F, (int)wgt.size()); ff++) {
                        const auto &act_index = act[ii];
                        const auto &wgt_index = wgt[ff];

                        auto x = std::get<0>(act_index);
                        auto y = std::get<1>(act_index);

                        auto k = std::get<0>(wgt_index);
                        auto r = std::get<1>(wgt_index);
                        auto s = std::get<2>(wgt_index);

                        int w = (x - r + padding) / stride;
                        int h = (y - s + padding) / stride;

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
    void SCNN<T>::computeSCNNTile(int n, int ct, int ck, int kc, int tw, int th, int X, int Y, int Kc, int K, int W,
            int H, int R, int S, int stride, int padding, const cnpy::Array<T> &act, const cnpy::Array<T> &wgt,
            sys::Statistics::Stats &stats) {

        std::vector<uint32_t> tile_cycles;
        std::vector<uint32_t> tile_dense_cycles;
        std::vector<uint32_t> tile_i_loop;
        uint32_t wgt_size = 0;

        for(int pex = 0; pex < Wt; pex++) {
            for(int pey = 0; pey < Ht; pey++) {
                int x_begin = pex * tw, y_begin = pey * th, k_begin = kc;
                int x_end = std::min(x_begin + tw, X), y_end = std::min(y_begin + th, Y), k_end = std::min(kc + Kc, K);

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

                        const PE_stats &pe_stats = computeSCNNPE(W,H,stride,padding,act_queue[sx][sy],
                                wgt_queue[sx][sy]);

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

                        stats.weight_buff_reads.back()[n] += stride_wgt_size;
                        stats.act_buff_reads.back()[n] += (uint64_t)(ceil(act_queue[sx][sy].size()/(double)I))*I;
                    }
                }
                wgt_size = PE_wgt_size;
                tile_cycles.push_back(PE_cycles);
                tile_dense_cycles.push_back(PE_dense_cycles);
                tile_i_loop.push_back(PE_i_loop);

                stats.idle_bricks.back()[n] += PE_f_loop * I * F - PE_mults;
                stats.mults.back()[n] += PE_mults;
                stats.idle_conflicts.back()[n] += PE_idle_conflicts;
                stats.accumulator_updates.back()[n] += PE_accumulator_updates;
                stats.i_loop.back()[n] += PE_i_loop;
                stats.f_loop.back()[n] += PE_f_loop;
            }
        }

        auto tile_max_cycles = *std::max_element(tile_cycles.begin(), tile_cycles.end());
        uint32_t tile_idle_pe = 0;
        for(const auto &PE_cycles : tile_cycles)
            tile_idle_pe += tile_max_cycles - PE_cycles;
        auto tile_max_i_loop =  *std::max_element(tile_i_loop.begin(), tile_i_loop.end());

        stats.cycles.back()[n] += tile_max_cycles;
        stats.dense_cycles.back()[n] += *std::max_element(tile_dense_cycles.begin(), tile_dense_cycles.end());
        stats.idle_pe.back()[n] += tile_idle_pe * I * F;
        stats.offchip_weight_reads.back()[n] += tile_max_i_loop * wgt_size;

    }

    /* CYCLES */

    template <typename T>
    void SCNN<T>::computeSCNNLayer(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        if(layer.getType() == "InnerProduct" || act.getDimensions() == 2) {
            if(act.getDimensions() == 4) act.reshape_to_2D();
            act.reshape_to_4D();
            auto C = act.getShape()[1];
            C = (int)(ceil(C/(double)256))*256;
            act.channel_zero_pad(C);
            act.split_4D(C / 256, 16, 16);

            auto Ck = wgt.getShape()[1];
            Ck = (int)(ceil(Ck/(double)256))*256;
            wgt.channel_zero_pad(Ck);
            wgt.split_4D(Ck / 256, 16, 16);
        }

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int N = act_shape[0];
        int C = act_shape[1];
        int X = act_shape[2];
        int Y = act_shape[3];
        if(this->FAST_MODE) N = 1;

        int K = wgt_shape[0];
        int Ck = wgt_shape[1];
        int R = wgt_shape[2];
        int S = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        int W = (X - R + 2*padding)/stride + 1;
        int H = (Y - S + 2*padding)/stride + 1;

        auto W_round = (int)(ceil(W/(double)Wt))*Wt;
        auto H_round = (int)(ceil(H/(double)Ht))*Ht;
        auto tw = W_round/Wt;
        auto th = H_round/Ht;
        auto Kc = (int)floor(out_acc_size/(double)(th*tw));

        // Fix for MobileNet
        if(Ck == 1) Kc = 1;

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(N,0));
        stats.dense_cycles.emplace_back(std::vector<uint64_t>(N,0));
        stats.mults.emplace_back(std::vector<uint64_t>(N,0));
        stats.idle_bricks.emplace_back(std::vector<uint64_t>(N,0));
        stats.idle_conflicts.emplace_back(std::vector<uint64_t>(N,0));
        stats.idle_pe.emplace_back(std::vector<uint64_t>(N,0));
        stats.idle_halo.emplace_back(std::vector<uint64_t>(N,0));
        stats.total_mult_cycles.emplace_back(std::vector<uint64_t>(N,0));
        stats.halo_transfers.emplace_back(std::vector<uint64_t>(N,0));
        stats.weight_buff_reads.emplace_back(std::vector<uint64_t>(N,0));
        stats.act_buff_reads.emplace_back(std::vector<uint64_t>(N,0));
        stats.accumulator_updates.emplace_back(std::vector<uint64_t>(N,0));
        stats.i_loop.emplace_back(std::vector<uint64_t>(N,0));
        stats.f_loop.emplace_back(std::vector<uint64_t>(N,0));
        stats.offchip_weight_reads.emplace_back(std::vector<uint64_t>(N,0));

        X = (int)(ceil(X/(double)Wt))*Wt;
        Y = (int)(ceil(Y/(double)Ht))*Ht;
        tw = X/Wt;
        th = Y/Wt;

        act.grid_zero_pad(X ,Y);

        int n;

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n = 0; n < N; n++) {
            for(int kc = 0; kc < K; kc+=Kc) {

                // Fix for MobileNet
                int channel_start = 0;
                int channel_max = C;
                if(Ck == 1) {
                    channel_start = kc;
                    channel_max = kc + 1;
                }

                for(int ct = channel_start; ct < channel_max; ct+=Ck) {
                    for(int ck = 0; ck < Ck; ck++) {
                        computeSCNNTile(n,ct,ck,kc,tw,th,X,Y,Kc,K,W,H,R,S,stride,padding,act,wgt,stats);
                    }
                }

                // resolve halos
                // compute the areas of the halo regions around a non edge PE
                // that is, how many psums need to get transferred

                const int DIM = 3;
                int x_vec[] = {R - 1 - padding, tw, padding};
                int y_vec[] = {S - 1 - padding, th, padding};
                int max_psum = 0;
                uint32_t halo_transfers = 0;

                for(int x = 0; x < DIM; x++) {
                    for (int y = 0; y < DIM; y++) {
                        int psum = x_vec[x] * y_vec[y];
                        if(x != 1 || y != 1)  {
                            halo_transfers += psum;
                            if(psum > max_psum)
                                max_psum = psum;
                        }
                    }
                }
                auto max_psums = max_psum * std::min(Kc, K - kc);

                stats.cycles.back()[n] += max_psums;
                stats.dense_cycles.back()[n] += max_psums;
                stats.idle_halo.back()[n] += max_psums * Ht * Wt * I * F;
                stats.halo_transfers.back()[n] += halo_transfers;
            }
            stats.total_mult_cycles.back()[n] = stats.mults.back()[n] + stats.idle_bricks.back()[n] +
                    stats.idle_conflicts.back()[n] + stats.idle_pe.back()[n] + stats.idle_halo.back()[n];
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void SCNN<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        stats.arch = "SCNN_Wt" + std::to_string(Wt) + "_Ht" + std::to_string(Ht) + "_Kt" + std::to_string(Kt) +
                "_I" + std::to_string(I) + "_F" + std::to_string(F) + "_acc_out" + std::to_string(out_acc_size);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution" || layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                computeSCNNLayer(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* POTENTIALS */

    template <typename T>
    void SCNN<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        int Nx = act_shape[2];
        int Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        int groups = act_channels / wgt_channels;
        int it_per_group = num_filters / groups;

        // Operations
        const auto parallel_mult = (uint64_t)(num_filters * out_x * out_y * Kx * Ky * wgt_channels);
        std::vector<uint64_t> bit_multiplications (batch_size,0);
        std::vector<double> work_reduction (batch_size,0);
        std::vector<double> speedup (batch_size,0);

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n=0; n<batch_size; n++) {

            int current_group = 0, group_m = 0, start_group = 0;
            uint64_t bit_counter = 0;

            for(int m=0; m<num_filters; m++) {
                for(int x=0; x<out_x; x++) {
                    for(int y=0; y<out_y; y++) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = start_group; k < wgt_channels + start_group; k++) {
                                    bit_counter += computeSCNNBitsPE(act.get(n, k, stride * x + i,stride * y + j)
                                            , wgt.get(m, k - start_group, i, j));
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

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.work_reduction.push_back(work_reduction);
        stats.speedup.push_back(speedup);
        stats.parallel_multiplications.push_back(parallel_mult);
        stats.bit_multiplications.push_back(bit_multiplications);

    }

    template <typename T>
    void SCNN<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        cnpy::Array<T> wgt = layer.getWeights();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        if(this->FAST_MODE) batch_size = 1;

        // Operations
        const auto parallel_mult = (uint64_t)num_filters * wgt_channels;
        std::vector<uint64_t> bit_multiplications (batch_size,0);
        std::vector<double> work_reduction (batch_size,0);
        std::vector<double> speedup (batch_size,0);

        int n;

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for (n = 0; n<batch_size; n++) {
            uint64_t bit_counter = 0;
            for (int m = 0; m<num_filters; m++) {
                for (int k = 0; k<wgt_channels; k++) {
                    bit_counter += computeSCNNBitsPE(act.get(n, k), wgt.get(m, k));
                }
            }
            work_reduction[n] = 100 - ((double) bit_counter / (double) parallel_mult / 256. * 100);
            speedup[n] = (double)parallel_mult * 256. / (double)bit_counter;
            bit_multiplications[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.work_reduction.push_back(work_reduction);
        stats.speedup.push_back(speedup);
        stats.parallel_multiplications.push_back(parallel_mult);
        stats.bit_multiplications.push_back(bit_multiplications);

    }


    template <typename T>
    void SCNN<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "SCNN";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(std::get<0>(layer.getWgt_precision()) + std::get<1>(layer.getWgt_precision()));
                computePotentialsConvolution(layer,stats);
            } else if (layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                stats.wgt_prec.push_back(std::get<0>(layer.getWgt_precision()) + std::get<1>(layer.getWgt_precision()));
                computePotentialsInnerProduct(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    INITIALISE_DATA_TYPES(SCNN);

}