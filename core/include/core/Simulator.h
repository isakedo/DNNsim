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

        /* Iterate set of windows in groups
         * @param out_x         Output activations X size
         * @param out_y         Output activations Y size
         * @param list_x        X position for the set of input windows (overwritten)
         * @param list_y        Y position for the set of input windows (overwritten)
         * @param max_windows   Maximum number of windows (Number of columns in the accelerator)
         * @return              Return false when all input windows are read
         */
        bool iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
                int max_windows = 16);

        /* Return the optimal encoding for the given value
         * @param value     Value we want to encode WITHOUT the sign
         * @return          Value with the optimal encoding
         */
        uint16_t booth_encoding(uint16_t value);

    };

}

#endif //DNNSIM_TIMINGSIMULATOR_H
