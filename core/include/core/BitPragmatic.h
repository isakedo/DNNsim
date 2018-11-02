#ifndef DNNSIM_BITPRAGMATIC_H
#define DNNSIM_BITPRAGMATIC_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class BitPragmatic : public Simulator<T> {

    private:

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         */
        void computeConvolution(const Layer<T> &layer);

    public:

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

    };

}

#endif //DNNSIM_BITPRAGMATIC_H
