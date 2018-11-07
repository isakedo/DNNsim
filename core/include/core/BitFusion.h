#ifndef DNNSIM_BITFUSION_H
#define DNNSIM_BITFUSION_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class BitFusion : public Simulator<T> {

    private:

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats);

    public:

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

    };

}

#endif //DNNSIM_BITFUSION_H
