
#include <core/BitPragmatic.h>

namespace core {

    template <typename T>
    static inline
    void computePragmaticFunctionalUnit(int n, int m, int x, int y, int i, int j, int k, int stride, int start_group,
                                          const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int max_k) {
        for(int channels = 0; channels < std::min(16,max_k); channels++) { // Process 16 synapses in a filter and window
            const T &activation = padded_act.get(n, k + channels, stride * x + i, stride * y + j);
            const T &wheight = wgt.get(m, k + channels - start_group, i, j);
            // Calculate cycles
        }
    }

    template <typename T>
    static inline
    void computePragmaticTile(int n, int m, std::vector<int> &list_x, std::vector<int> &list_y, int i, int j, int k,
            int stride, int start_group, const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int max_k) {

        for(int window = 0; window < list_x.size(); window++)   // Process 16 windows
            for(int filter = 0; filter < 16; filter++)  // Process 16 filters
            computePragmaticFunctionalUnit<T>(n,m + filter,list_x[window],list_y[window],i,j,k,stride,
                    start_group,padded_act,wgt,max_k); // 256 computations
    }

    static inline
    bool getWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y) {
        static int x = 0;
        static int y = 0;
        list_x.clear();
        list_y.clear();
        const int max_windows = 16;
        int current_windows = 0;
        while(x < out_x) {
            while(y < out_y) {
                list_x.push_back(x);
                list_y.push_back(y);
                current_windows++;
                y++;
                if(current_windows >= max_windows)
                    return true;
            }
            y = 0;
            x++;
        }
        if(current_windows > 0)
            return true;

        x = 0;
        return false;
    }

    template <typename T>
    void BitPragmatic<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {
        // Simplify names getting their pointers
        const cnpy::Array<T> &wgt = layer.getWeights();
        const std::vector<size_t> &wgt_shape = wgt.getShape();
        const cnpy::Array<T> &act = layer.getActivations();
        const std::vector<size_t> &act_shape = act.getShape();

        int padding = layer.getPadding();
        int stride = layer.getStride();
        int Kx = layer.getKx();
        int Ky = layer.getKy();

        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (act_shape[2] - wgt_shape[2] + 2*padding)/stride + 1;
        long out_y = (act_shape[3] - wgt_shape[3] + 2*padding)/stride + 1;

        // Set filter grouping
        int groups = (int)act.getShape()[1] / (int)wgt_shape[1];
        int it_per_group = (int)wgt_shape[0] / groups;
        int current_group = 0, group_m =0, start_group = 0;

        // Convolution
        for(int n=0; n<act_shape[0]; n++) {
            for(int m=0; m<wgt_shape[0]; m += 16) { // Sixteen filters each time
                std::vector<int> list_x;
                std::vector<int> list_y;
                while(getWindows(out_x,out_y,list_x,list_y)) { // Sixteen windows each time
                    // Compute in parallel
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            // Sixteen values depthwise, sixteen channels
                            for (int k = start_group; k < wgt_shape[1] + start_group; k += 16) {
                                computePragmaticTile<T>(n, m, list_x, list_y, i, j, k, stride, start_group,
                                        padded_act, wgt, (int)act.getShape()[1]);
                            }
                        }
                    }
                }
                group_m++;
                if(group_m >= it_per_group) {
                    group_m = 0;
                    current_group++;
                    start_group = (int)wgt_shape[1]*current_group;
                }
            }
        }


    }

    template <typename T>
    void BitPragmatic<T>::memoryAccesses(const Network<T> &network) {
        sys::Statistics::Stats stats;

        stats.net_name = network.getName();
        stats.arch = "BitPragmatic";

        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {

                // Simplify names getting their pointers
                const cnpy::Array<T> &wgt = layer.getWeights();
                const std::vector<size_t> &wgt_shape = wgt.getShape();
                const cnpy::Array<T> &act = layer.getActivations();
                const std::vector<size_t> &act_shape = act.getShape();

                int padding = layer.getPadding();
                int stride = layer.getStride();
                int Kx = layer.getKx();
                int Ky = layer.getKy();

                long out_x = (act_shape[2] - wgt_shape[2] + 2*padding)/stride + 1;
                long out_y = (act_shape[3] - wgt_shape[3] + 2*padding)/stride + 1;

                //Memory stats - 16 bits
                auto num_weights_sets = (uint32_t)ceil(wgt_shape[0]/16.); // Groups of 16 weights
                auto num_activations_sets = (uint32_t)ceil(out_x*out_y/16.); // Groups of 16 windows
                auto num_channel_sets = (uint32_t)ceil(wgt_shape[1]/16.); // Groups of 16 channels

                stats.layers.push_back(layer.getName());
                stats.on_chip_weights.push_back(num_weights_sets*num_activations_sets*num_channel_sets*Kx*Ky);
                stats.on_chip_activations.push_back(num_weights_sets*num_activations_sets*num_channel_sets*Kx*Ky);
                stats.off_chip_weights_sch3.push_back(1); // Filters per layer
                stats.bits_weights.push_back((uint32_t)(wgt_shape[0]*wgt_shape[1]*wgt_shape[2]*wgt_shape[3]*16));
                stats.off_chip_weights_sch4.push_back(num_weights_sets); // Working set of filters
                stats.bits_working_weights.push_back((uint32_t)(16*wgt_shape[1]*wgt_shape[2]*wgt_shape[3]*16));
                stats.off_chip_activations.push_back((uint32_t)out_y); // One row of activations
                stats.bits_one_activation_row.push_back((uint32_t)(act_shape[2]*Ky*act_shape[1]*16));
                stats.computations.push_back(num_weights_sets*num_activations_sets*num_channel_sets*Kx*Ky);

            }
        }

        // Set statistics to write
        sys::Statistics::addStats(stats);
    }

    template <typename T>
    void BitPragmatic<T>::run(const Network<T> &network) {
        sys::Statistics::Stats stats;
        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                computeConvolution(layer, stats);
            }
        }
    }

    template class BitPragmatic<uint16_t>;

}