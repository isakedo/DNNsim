#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include "Layer.h"
#include "Network.h"

namespace core {

    class Simulator {

    private:

        void computeConvolution(const Layer &layer, cnpy::Array &result, bool ReLu);

        void computePooling(const Layer &layer, cnpy::Array &result);

        void computeInnerProduct(const Layer &layer, cnpy::Array &result, bool ReLu);


    public:

        void run(const Network &network);
        
    };
}

#endif //DNNSIM_SIMULATOR_H
