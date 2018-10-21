#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

#include "Layer.h"

namespace core {

    class Simulator {

    protected:

        static inline float ReLu(const float &value);

        void computeConvolution(const Layer &layer, cnpy::Array &result, bool ReLu);

        void computePooling(const Layer &layer, cnpy::Array &result);

        void computeFullyConnected(const Layer &layer, cnpy::Array &result, bool ReLu);
        
    };
}

#endif //DNNSIM_SIMULATOR_H
