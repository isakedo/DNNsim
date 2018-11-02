
#include <core/Laconic.h>

namespace core {

    template <typename T>
    void Laconic<T>::computeConvolution(const core::Layer<T> &layer) {

    }

    template <typename T>
    void Laconic<T>::run(const Network<T> &network) {

    }

    template <typename T>
    void Laconic<T>::computeWorkReductionConvolution(const core::Layer<T> &layer) {
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

    }

    template <typename T>
    void Laconic<T>::workReduction(const Network<T> &network) {
        for(const Layer<T> &layer : network.getLayers()) {
            if(layer.getType() == "Convolution") {
                computeWorkReductionConvolution(layer);
            }
        }
    }

    template class Laconic<uint16_t >;

}