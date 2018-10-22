#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include "Layer.h"
#include "Network.h"

namespace core {

    class Simulator {

    private:

        /* Compute the output activations for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param result    Output array where output activations are stored
         * @param has_ReLU  Bool to know if the layer has ReLU
         */
        void computeConvolution(const Layer &layer, cnpy::Array &result, bool has_ReLU);

        /* Compute the output activations for a fully connected layers
         * @param layer     Layer for which we want to calculate the outputs
         * @param result    Output array where output activations are stored
         * @param has_ReLU  Bool to know if the layer has ReLU
         */
        void computeInnerProduct(const Layer &layer, cnpy::Array &result, bool has_ReLU);

    public:

        /* Compute the output activations fot all the layers in the network, and check their correctness
         * @param network   Initialised network for the computation
         */
        void run(const Network &network);
        
    };
}

#endif //DNNSIM_SIMULATOR_H
