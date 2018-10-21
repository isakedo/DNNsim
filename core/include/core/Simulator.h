#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include "Layer.h"
#include "Network.h"

namespace core {

    class Simulator {

    private:

        void computeInferenceTestConvolution(const Layer &layer, cnpy::Array &result, bool ReLu);

        void computeInferenceTestPooling(const Layer &layer, cnpy::Array &result);

        void computeInferenceTestFullyConnected(const Layer &layer, cnpy::Array &result, bool ReLu);


    public:

        void inferenceTestSimulation(const Network &network);
        
    };
}

#endif //DNNSIM_SIMULATOR_H
