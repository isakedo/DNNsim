
#include <core/SCNNe.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint16_t SCNNe<T>::computeSCNNeBitsPE(T act, T wgt) {

        #ifdef ZERO_COUNT
        if(wgt == 0) return 1;
        else if(act == 0) return 1;
        #else
        if(wgt == 0) return 0;
        else if(act == 0) return 0;
        #endif

        uint16_t act_bits = act;

        #ifdef BOOTH_ENCODING
        act_bits = this->booth_encoding(act_bits);
        #endif

        uint8_t act_effectual_bits = this->effectualBits(act_bits);

        uint8_t bit_multiplications = act_effectual_bits * (uint8_t)16;
        #ifdef ZERO_COUNT
        if(bit_multiplications == 0) bit_multiplications = 1;
        #endif

        return bit_multiplications;
    }

    template <typename T>
    typename SCNNe<T>::PE_stats SCNNe<T>::computeSCNNePE(int W, int H, int stride, const act_idxMap &act,
            const wgt_idxMap &wgt) {

        PE_stats pe_stats;
        pe_stats.cycles = 0;
        pe_stats.mults = 0;
        pe_stats.idle_conflicts = 0;
        pe_stats.idle_column_cycles = 0;
        pe_stats.column_stalls = 0;
        pe_stats.accumulator_updates = 0;
        pe_stats.i_loop = 0;
        pe_stats.f_loop = 0;

        for(int i = 0; i < act.size(); i+=this->I) {
            pe_stats.i_loop += 1;
            for(int f = 0; f < wgt.size(); f+=this->F) {
                pe_stats.f_loop += 1;
                uint8_t accumulator_updates = 0;
                std::vector<uint8_t> acc_cycles;
                std::unordered_set<uint8_t> uniq_acc_cycles;
                std::vector<std::vector<uint8_t>> acc ((unsigned)this->I,
                        std::vector<uint8_t>(2 * this->F * this->I, 0));

                for(int ii = i; ii < std::min(i + this->I, (int)act.size()); ii++) {

                    const auto &act_index = act[ii];
                    auto x = std::get<0>(act_index);
                    auto y = std::get<1>(act_index);
                    auto act_cycles = std::get<2>(act_index);

                    uint8_t column_acc_updates = 0;
                    for(int ff = f; ff < std::min(f + this->F, (int)wgt.size()); ff++) {
                        const auto &wgt_index = wgt[ff];

                        auto k = std::get<0>(wgt_index);
                        auto r = std::get<1>(wgt_index);
                        auto s = std::get<2>(wgt_index);

                        int w = (x - r) / stride;
                        int h = (y - s) / stride;

                        if(w >= 0 && w < W && h >= 0 && h < H) {
                            int acc_idx = this->map_accumulator(k, w, h);
                            acc[ii % this->I][acc_idx] += 1;
                            column_acc_updates++;
                            pe_stats.mults += 1;
                        }
                    }

                    if(column_acc_updates != 0) {
                        uniq_acc_cycles.emplace(act_cycles);
                        acc_cycles.push_back(act_cycles);
                    }
                    accumulator_updates += column_acc_updates;
                }
                std::vector<uint8_t> acc_maxs = std::vector<uint8_t>(this->I,0);
                for(int a = 0; a < this->I; a++) {
                    auto acc_max = *std::max_element(acc[a].begin(), acc[a].end());
                    acc_maxs[a] = std::max(acc_max - 1, 0);
                }
                auto conflicts_stalls = std::accumulate(acc_maxs.begin(), acc_maxs.end(), 0);
                auto column_stalls = acc_cycles.size() - uniq_acc_cycles.size();

                if(accumulator_updates != 0) {
                    auto max_cycles = *std::max_element(uniq_acc_cycles.begin(), uniq_acc_cycles.end());

                    for(const auto &column_cycles : acc_cycles) {
                        pe_stats.idle_column_cycles += max_cycles - column_cycles;
                    }

                    pe_stats.cycles += max_cycles + conflicts_stalls + column_stalls;
                } else
                    pe_stats.cycles += 1;
                pe_stats.idle_conflicts += conflicts_stalls * this->I * this->F;
                pe_stats.column_stalls += column_stalls;
                pe_stats.accumulator_updates += accumulator_updates;
            }
        }

        return pe_stats;
    }

    template <typename T>
    void SCNNe<T>::computeSCNNeTile(int n, int ct, int ck, int kc, int tw, int th, int X, int Y, int Kc, int K, int W,
            int H, int R, int S, int stride, int padding, const cnpy::Array<T> &act, const cnpy::Array<T> &wgt,
            sys::Statistics::Stats &stats) {

        std::vector<uint32_t> tile_cycles;
        std::vector<uint32_t> tile_dense_cycles;
        std::vector<uint32_t> tile_i_loop;
        uint32_t wgt_size = 0;

        for(int pex = 0; pex < this->Wt; pex++) {
            for(int pey = 0; pey < this->Ht; pey++) {
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
                        if(act_bits != 0) {
                            auto act_cycles = this->effectualBits(act_bits);
                            act_cycles = (uint8_t)ceil(act_cycles/(double)PE_SERIAL_BITS);
                            act_queue[sx][sy].emplace_back(std::make_tuple(x, y, act_cycles));
                        }
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
                uint32_t PE_idle_column_cycles = 0;
                uint32_t PE_column_stalls = 0;
                uint32_t PE_accumulator_updates = 0;
                uint32_t PE_i_loop = 0;
                uint32_t PE_f_loop = 0;
                uint32_t PE_wgt_size = 0;

                for(int sx = 0; sx < stride; sx++) {
                    for(int sy = 0; sy < stride; sy++) {

                        const PE_stats &pe_stats = computeSCNNePE(W,H,stride,act_queue[sx][sy],wgt_queue[sx][sy]);

                        auto stride_wgt_size = (uint32_t)(ceil(wgt_queue[sx][sy].size()/(double)this->F))*this->F;
                        PE_wgt_size += stride_wgt_size;

                        PE_cycles += pe_stats.cycles;
                        PE_dense_cycles += (uint32_t)(ceil(dense_act_counter[sx][sy]/(double)this->I) *
                                ceil(dense_wgt_counter[sx][sy]/(double)this->F));
                        PE_mults += pe_stats.mults;
                        PE_idle_conflicts += pe_stats.idle_conflicts;
                        PE_idle_column_cycles += pe_stats.idle_column_cycles;
                        PE_column_stalls += pe_stats.column_stalls;
                        PE_accumulator_updates += pe_stats.accumulator_updates;
                        PE_i_loop += pe_stats.i_loop;
                        PE_f_loop += pe_stats.f_loop;

                        stats.weight_buff_reads.back()[n] += stride_wgt_size;
                        stats.act_buff_reads.back()[n] += (uint64_t)(ceil(act_queue[sx][sy].size() /
                                (double)this->I)) *this->I;
                    }
                }
                wgt_size = PE_wgt_size;
                tile_cycles.push_back(PE_cycles);
                tile_dense_cycles.push_back(PE_dense_cycles);
                tile_i_loop.push_back(PE_i_loop);

                stats.idle_bricks.back()[n] += PE_f_loop * this->I * this->F - PE_mults;
                stats.mults.back()[n] += PE_mults;
                stats.idle_conflicts.back()[n] += PE_idle_conflicts;
                stats.idle_column_cycles.back()[n] += PE_idle_column_cycles;
                stats.column_stalls.back()[n] += PE_column_stalls;
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
        stats.idle_pe.back()[n] += tile_idle_pe * this->I * this->F;
        stats.offchip_weight_reads.back()[n] += tile_max_i_loop * wgt_size;

    }

    /* CYCLES */

    template <typename T>
    void SCNNe<T>::computeSCNNeLayer(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
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

        int padding = layer.getPadding();
        int stride = layer.getStride();

        act.zero_pad(padding);

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

        int W = (X - R)/stride + 1;
        int H = (Y - S)/stride + 1;

        int groups = C / Ck;
        int Kg = K / groups;

        auto W_round = (int)(ceil(W/(double)this->Wt))*this->Wt;
        auto H_round = (int)(ceil(H/(double)this->Ht))*this->Ht;
        auto tw = W_round/this->Wt;
        auto th = H_round/this->Ht;
        auto Kc = (int)floor(this->out_acc_size/(double)(th*tw));

        // Fix for MobileNet
        if(Ck == 1) Kc = 1;

        // Stats
        stats.cycles.emplace_back(std::vector<uint64_t>(N,0));
        stats.dense_cycles.emplace_back(std::vector<uint64_t>(N,0));
        stats.mults.emplace_back(std::vector<uint64_t>(N,0));
        stats.idle_bricks.emplace_back(std::vector<uint64_t>(N,0));
        stats.idle_conflicts.emplace_back(std::vector<uint64_t>(N,0));
        stats.idle_column_cycles.emplace_back(std::vector<uint64_t>(N,0));
        stats.column_stalls.emplace_back(std::vector<uint64_t>(N,0));
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

        X = (int)(ceil(X/(double)this->Wt))*this->Wt;
        Y = (int)(ceil(Y/(double)this->Ht))*this->Ht;
        tw = X/this->Wt;
        th = Y/this->Wt;

        act.grid_zero_pad(X ,Y);

        int n;

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n = 0; n < N; n++) {
            for(int kc = 0; kc < K; kc+=Kc) {

                // Two towers alexnet
                int ct = 0;
                if(kc >= Kg) ct = Ck;

                // Fix for MobileNet
                if(Ck == 1) ct = kc;

                for(int ck = 0; ck < Ck; ck++) {
                    computeSCNNeTile(n,ct,ck,kc,tw,th,X,Y,Kc,K,W,H,R,S,stride,padding,act,wgt,stats);
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
                stats.idle_halo.back()[n] += max_psums * this->Ht * this->Wt * this->I * this->F;
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
    void SCNNe<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        stats.arch = "SCNNe_Wt" + std::to_string(this->Wt) + "_Ht" + std::to_string(this->Ht) + "_I" +
                std::to_string(this->I) + "_F" + std::to_string(this->F) + "_acc_out" +
                std::to_string(this->out_acc_size) + "_B" + std::to_string(this->BANKS) + "_PSB" +
                std::to_string(PE_SERIAL_BITS);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution" || layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                computeSCNNeLayer(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    /* POTENTIALS */

    template <typename T>
    void SCNNe<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

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
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));

        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n)
        #endif
        for(n=0; n<batch_size; n++) {
            uint64_t bit_counter = 0;
            for(int m=0; m<num_filters; m++) {
                int start_group = 0;
                if(m >= it_per_group)
                    start_group = wgt_channels;
                for(int x=0; x<out_x; x++) {
                    for(int y=0; y<out_y; y++) {
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = 0; k < wgt_channels; k++) {
                                    bit_counter += computeSCNNeBitsPE(act.get(n, start_group + k, stride * x + i,
                                            stride * y + j), wgt.get(m, k, i, j));
                                }
                            }
                        }
                    }
                }
            }
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / 256. * 100);
            stats.speedup.back()[n] = (double)parallel_mult * 256. / (double)bit_counter;
            stats.bit_multiplications.back()[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.parallel_multiplications.push_back(parallel_mult);

    }

    template <typename T>
    void SCNNe<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

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
        stats.bit_multiplications.emplace_back(std::vector<uint64_t>(batch_size,0));
        stats.work_reduction.emplace_back(std::vector<double>(batch_size,0));
        stats.speedup.emplace_back(std::vector<double>(batch_size,0));

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
                    bit_counter += computeSCNNeBitsPE(act.get(n, k), wgt.get(m, k));
                }
            }
            stats.work_reduction.back()[n] = 100 - ((double)bit_counter / (double)parallel_mult / 256. * 100);
            stats.speedup.back()[n] = (double)parallel_mult * 256. / (double)bit_counter;
            stats.bit_multiplications.back()[n] = bit_counter;
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.parallel_multiplications.push_back(parallel_mult);

    }


    template <typename T>
    void SCNNe<T>::potentials(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "potentials";
        stats.net_name = network.getName();
        stats.arch = "SCNNe";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(0);
                computePotentialsConvolution(layer,stats);
            } else if (layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(layer.getAct_precision());
                stats.wgt_prec.push_back(0);
                computePotentialsInnerProduct(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class SCNNe<uint16_t>;

}