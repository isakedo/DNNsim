#ifndef DNNSIM_LACONIC_H
#define DNNSIM_LACONIC_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class Laconic : public Simulator<T> {

    private:

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the work reduction for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeWorkReductionConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats);


    public:

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

        /* Calculate work reduction for the given network
         * @param network   Network we want to calculate work reduction
         */
        void workReduction(const Network<T> &network);

    };

}

#endif //DNNSIM_LACONIC_H
