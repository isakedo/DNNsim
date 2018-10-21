
#include <core/Simulator.h>

namespace core {

    float Simulator::ReLu(const float &value) {

    }

    void Simulator::computeConvolution(const core::Layer &layer, cnpy::Array &result, bool ReLu) {

    }

    void Simulator::computePooling(const core::Layer &layer, cnpy::Array &result) {

    }

    void Simulator::computeFullyConnected(const Layer &layer, cnpy::Array &result, bool ReLu) {
        layer.getActivations().get(0,0,0,0);
        layer.getActivations().getDimensions();
        layer.getActivations().getShape()[0];
        std::vector<size_t> output_shape;
        std::vector<float> output_activations;
        output_activations.push_back(.7);
        result.set_values(output_activations,output_shape);
    }

}