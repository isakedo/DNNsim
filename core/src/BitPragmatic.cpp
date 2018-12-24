
#include <core/BitPragmatic.h>

#define N_COLUMNS 16
#define ZERO_COUNT
#define BOOTH_ENCODING

namespace core {

    template <typename T>
    uint8_t BitPragmatic<T>::computePragmaticColumn(int batch, int act_x, int act_y, int kernel_x, int kernel_y,
            int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel) {

        // WRONG
        //Get the maximum number of cycles in this column
        std::vector<uint8_t> cycles;
        for(int channel = 0; channel < std::min(16,max_channel); channel++) {

            uint16_t act_bits = padded_act.get(batch, init_channel + channel, stride * act_x + kernel_x,
                    stride * act_y + kernel_y);
            #ifdef BOOTH_ENCODING
                act_bits = this->booth_encoding(act_bits);
            #endif

            uint8_t PE_cycles = 0;
            while (act_bits) {
                PE_cycles += act_bits & 1;
                act_bits >>= 1;
            }

            #ifdef ZERO_COUNT
            if(PE_cycles == 0) PE_cycles = 1;
            #endif

            cycles.push_back(PE_cycles);

        }

        return *std::max_element(cycles.begin(), cycles.end());

    }

    template <typename T>
    uint8_t BitPragmatic<T>::computePragmaticTile(int batch, std::vector<int> &list_act_x, std::vector<int> &list_act_y,
            int kernel_x, int kernel_y, int init_channel, int stride, const cnpy::Array<T> &padded_act,
            int max_channel) {

        //Get the number of cycles in the slowest column
        std::vector<uint8_t> cycles;
        for(int window = 0; window < list_act_x.size(); window++) {  // Process 16 windows
            uint8_t column_cycles = computePragmaticColumn(batch, list_act_x[window], list_act_y[window], kernel_x,
                    kernel_y, init_channel, stride, padded_act, max_channel);
            cycles.push_back(column_cycles);
        }
        return *std::max_element(cycles.begin(), cycles.end());

    }

    template <typename T>
    void BitPragmatic<T>::computeConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) {

        cnpy::Array<T> act = layer.getActivations();
        act.two_exponents_representation();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = layer.getWeights().getShape();

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

        // Convolution
        for(int n=0; n<act_shape[0]; n++) {
            int current_group = 0, group_m =0, start_group = 0;
            uint32_t cycles = 0;
            for(int m=0; m<wgt_shape[0]; m += 16) { // Sixteen filters each time
                std::vector<int> list_x;
                std::vector<int> list_y;
                while(this->iterateWindows(out_x,out_y,list_x,list_y,N_COLUMNS)) { // Sixteen windows each time
                    // Compute in parallel
                    for (int i = 0; i < Kx; i++) {
                        for (int j = 0; j < Ky; j++) {
                            // Sixteen values depthwise, sixteen channels
                            for (int k = start_group; k < wgt_shape[1] + start_group; k += 16) {
                                cycles += computePragmaticTile(n, list_x, list_y, i, j, k, stride, padded_act,
                                        (int)act.getShape()[1]);
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

        stats.task_name = "mem_accesses";
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