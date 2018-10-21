
#include <core/Simulator.h>

namespace core {

    static inline float ReLU(const float &value) {

    }

    void Simulator::computeInferenceTestConvolution(const core::Layer &layer, cnpy::Array &result, bool ReLu) {

    }

    void Simulator::computeInferenceTestPooling(const core::Layer &layer, cnpy::Array &result) {

    }

    void Simulator::computeInferenceTestFullyConnected(const Layer &layer, cnpy::Array &result, bool ReLu) {
        layer.getActivations().get(0,0,0,0);
        layer.getActivations().getDimensions();
        layer.getActivations().getShape()[0];
        std::vector<size_t> output_shape;
        std::vector<float> output_activations;
        output_activations.push_back(.7);
        result.set_values(output_activations,output_shape);
    }

    void Simulator::inferenceTestSimulation(const Network &network) {

    }

}