#ifndef DNNSIM_LACONIC_H
#define DNNSIM_LACONIC_H

#include "TimingSimulator.h"

namespace core {

    template <typename T>
    class Laconic : public TimingSimulator<T> {

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
