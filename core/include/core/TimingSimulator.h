#ifndef DNNSIM_TIMINGSIMULATOR_H
#define DNNSIM_TIMINGSIMULATOR_H


#include <sys/common.h>
#include <sys/Statistics.h>
#include <cnpy/Array.h>
#include "Layer.h"
#include "Network.h"

namespace core {

    template <typename T>
    class TimingSimulator {

    private:

        /* Return a vector zero padded
         * @param array     Array we want to apply padding
         * @param padding   Padding of the layer
         * @return          Array zero padded
         */
        cnpy::Array<T> adjustPadding(const cnpy::Array<T> &array, int padding);

    public:

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        virtual void run(const Network<T> &network) = 0;

    };

}

#endif //DNNSIM_TIMINGSIMULATOR_H
