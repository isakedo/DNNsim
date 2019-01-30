
#include <core/SCNN.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

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

    /* CYCLES */

    template <typename T>
    void SCNN<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        const cnpy::Array<T> &wgt = layer.getWeights();

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

        long W = (X - R + 2*padding)/stride + 1;
        long H = (Y - S + 2*padding)/stride + 1;

        auto W_round = (int)(ceil(W/(double)Wt))*Wt;
        auto H_round = (int)(ceil(H/(double)Ht))*Ht;
        auto tw = W_round/Wt;
        auto th = H_round/Ht;
        auto Kc = (int)floor(out_acc_size/(double)(th*tw));

        // Stats
        auto index = stats.cycles.size();
        stats.cycles.emplace_back(std::vector<uint32_t>(N,0));
        stats.dense_cycles.emplace_back(std::vector<uint32_t>(N,0));
        stats.mults.emplace_back(std::vector<uint32_t>(N,0));
        stats.idle_bricks.emplace_back(std::vector<uint32_t>(N,0));
        stats.idle_conflicts.emplace_back(std::vector<uint32_t>(N,0));
        stats.idle_pe.emplace_back(std::vector<uint32_t>(N,0));
        stats.idle_halo.emplace_back(std::vector<uint32_t>(N,0));
        stats.halo_transfers.emplace_back(std::vector<uint32_t>(N,0));
        stats.weight_buff_reads.emplace_back(std::vector<uint32_t>(N,0));
        stats.act_buff_reads.emplace_back(std::vector<uint32_t>(N,0));
        stats.accumulator_updates.emplace_back(std::vector<uint32_t>(N,0));
        stats.i_loop.emplace_back(std::vector<uint32_t>(N,0));
        stats.f_loop.emplace_back(std::vector<uint32_t>(N,0));
        stats.offchip_weight_reads.emplace_back(std::vector<uint32_t>(N,0));

        X = (int)(ceil(X/(double)Wt))*Wt;
        Y = (int)(ceil(Y/(double)Ht))*Ht;
        tw = X/Wt;
        th = Y/Wt;

        act.grid_zero_pad(X ,Y);
        const auto &act_idx = this->generate_idxMap(act);

        int n;

        // Convolution
        for(n = 0; n < N; n++) {
            for(int kc = 0; kc < K; kc+=Kc) {
                for(int ct = 0; ct < C; ct+=Ck) {
                    for(int ck = 0; ck < Ck; ck++) {

                    }
                }
            }
        }

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);

    }

    template <typename T>
    void SCNN<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

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
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                computeConvolution(layer, stats);
            } /*else if(layer.getType() == "InnerProduct") {
                stats.layers.push_back(layer.getName());
                computeInnerProduct(layer, stats);
            }*/
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
        uint64_t bit_counter = 0;

        int current_group = 0, group_m = 0, start_group = 0;
        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n,current_group,group_m,start_group,bit_counter)
        #endif
        for(n=0; n<batch_size; n++) {
            current_group = 0; group_m =0; start_group = 0; bit_counter = 0;
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
        uint64_t bit_counter = 0;

        int n;

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n,bit_counter)
        #endif
        for (n = 0; n<batch_size; n++) {
            bit_counter = 0;
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