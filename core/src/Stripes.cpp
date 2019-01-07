
#include <core/Stripes.h>

namespace core {

    /* AUXILIARY FUNCTIONS */


    /* CYCLES */

    template <typename T>
    void Stripes<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        cnpy::Array<T> act = layer.getActivations();
        act.powers_of_two_representation();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();

        std::vector<size_t> act_shape = act.getShape();
        std::vector<size_t> wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        int Nx = act_shape[2];
        int Ny = act_shape[3];

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        int groups = act_channels / wgt_channels;
        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS/groups);

        // Stats
        std::vector<uint32_t> cycles (batch_size,0);
        uint32_t batch_cycles;

        std::vector<int> list_x, list_y;
        int n;

        // Get layer precision
        auto layer_prec = std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision());

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(max_threads);
        #pragma omp parallel for private(n,batch_cycles,list_x,list_y)
        #endif
        for(n=0; n<batch_size; n++) {
            batch_cycles = 0;
            while(this->iterateWindows(out_x,out_y,list_x,list_y,N_COLUMNS)) {
                for (int i = 0; i < Kx; i++) {
                    for (int j = 0; j < Ky; j++) {
                        for (int k = 0; k < act_channels; k += 16) {
                            batch_cycles += layer_prec;
                        }
                    }
                }
            }
            cycles[n] = batch_cycles*num_filters_sets;
        }

        auto base_cycles = (uint32_t)(out_x * out_y * act_channels * Kx * Ky * num_filters_sets / 16);
        auto avg_cycles = accumulate(cycles.begin(), cycles.end(), 0.0)/cycles.size();

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.time.push_back(time_span);
        stats.STR_cycles.push_back(cycles);
        stats.STR_baseline_cycles.push_back(base_cycles);
        stats.STR_avg_cycles.push_back((uint32_t)avg_cycles);
    }

    template <typename T>
    void Stripes<T>::computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

    }

    template <typename T>
    void Stripes<T>::run(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "cycles";
        stats.net_name = network.getName();
        stats.arch = "Stripes_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                stats.act_prec.push_back(std::get<0>(layer.getAct_precision()) + std::get<1>(layer.getAct_precision()));
                computeConvolution(layer, stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);

    }

    /* POTENTIALS */

    template <typename T>
    void Stripes<T>::computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

    }

    template <typename T>
    void Stripes<T>::computePotentialsInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) {

    }

    template <typename T>
    void Stripes<T>::potentials(const Network<T> &network) {

    }

    /* MEMORY ACCESSES */

    template <typename T>
    void Stripes<T>::computeMemAccessesConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {
        const cnpy::Array<T> &wgt = layer.getWeights();
        const std::vector<size_t> &wgt_shape = wgt.getShape();
        const cnpy::Array<T> &act = layer.getActivations();
        const std::vector<size_t> &act_shape = act.getShape();

        int act_channels = act_shape[1];
        int Nx = act_shape[2];
        int Ny = act_shape[3];

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        //Memory stats - 16 bits
        int groups = act_channels / wgt_channels;
        auto num_filters_sets = (uint32_t)ceil(num_filters/(double)N_ROWS)/groups;
        auto num_activations_sets = (uint32_t)ceil(out_x*out_y/(double)N_COLUMNS);
        auto num_channel_sets = (uint32_t)ceil(wgt_channels/16.);

        stats.on_chip_accesses_filters.push_back(num_filters_sets*num_activations_sets*num_channel_sets*Kx*Ky);
        stats.on_chip_accesses_activations.push_back(num_filters_sets*num_activations_sets*num_channel_sets*Kx*Ky);
        stats.off_chip_accesses_filters_sch3.push_back(1); // All filters per layer
        stats.off_chip_accesses_filters_sch4.push_back(num_filters_sets); // Working set of filters
        stats.off_chip_accesses_activations.push_back((uint32_t)out_y); // One row of activations
        stats.num_bytes_filters_sche3.push_back((uint32_t)(num_filters*wgt_channels*Kx*Ky*16)/8);
        stats.num_bytes_filters_sche4.push_back((uint32_t)(16*wgt_channels*Kx*Ky*16)/8);
        stats.num_bytes_one_row_activations.push_back((uint32_t)(Nx*Ky*act_channels*16)/8);
        stats.num_computations.push_back(num_filters_sets*num_activations_sets*num_channel_sets*Kx*Ky);
    }


    template <typename T>
    void Stripes<T>::memoryAccesses(const Network<T> &network) {
        // Initialize statistics
        sys::Statistics::Stats stats;
        sys::Statistics::initialize(stats);

        stats.task_name = "mem_accesses";
        stats.net_name = network.getName();
        stats.arch = "Stripes_C" + std::to_string(N_COLUMNS) + "_R" + std::to_string(N_ROWS);

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                stats.layers.push_back(layer.getName());
                computeMemAccessesConvolution(layer,stats);
            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template class Stripes<uint16_t>;

}
