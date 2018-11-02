
#include <core/BitPragmatic.h>

namespace core {

    template <typename T>
    void BitPragmatic<T>::computeConvolution(const core::Layer<T> &layer) {
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

        // Set filter batching
        int batches = (int)act.getShape()[1] / (int)wgt_shape[1];
        int it_per_batch = (int)wgt_shape[0] / batches;
        int current_batch = 0, batch_m =0, start_batch = 0;


    }

    template <typename T>
    void BitPragmatic<T>::run(const Network<T> &network) {
        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                computeConvolution(layer);
            }
        }
    }

    template class BitPragmatic<uint16_t>;

}