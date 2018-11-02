#ifndef DNNSIM_BITPRAGMATIC_H
#define DNNSIM_BITPRAGMATIC_H

#include "TimingSimulator.h"

namespace core {

    template <typename T>
    class BitPragmatic : public TimingSimulator<T> {

    public:

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

    };

}

#endif //DNNSIM_BITPRAGMATIC_H
