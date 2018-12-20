#ifndef DNNSIM_TIMINGSIMULATOR_H
#define DNNSIM_TIMINGSIMULATOR_H


#include <sys/common.h>
#include <sys/Statistics.h>
#include <cnpy/Array.h>
#include "Layer.h"
#include "Network.h"

#ifdef OPENMP
#include <omp.h>
#endif

namespace core {

    template <typename T>
    class Simulator {

    protected:

        /* Return a vector zero padded
         * @param array     Array we want to apply padding
         * @param padding   Padding of the layer
         * @return          Array zero padded
         */
        cnpy::Array<T> adjustPadding(const cnpy::Array<T> &array, int padding);

        /* Return the optimal encoding for the given value
         * @param value     Value we want to encode WITHOUT the sign
         * @return          Value with the optimal encoding
         */
        uint16_t booth_encoding(uint16_t value);

    };

}

#endif //DNNSIM_TIMINGSIMULATOR_H
