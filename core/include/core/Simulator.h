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
         * @param list_x        X position for the set of input windows (Overwritten)
         * @param list_y        Y position for the set of input windows (Overwritten)
         * @param max_windows   Maximum number of windows (Number of columns in the accelerator)
         * @return              Return false when all input windows are read
         */
        bool iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
                int max_windows = 16);

        /* Calculate the 3D mapping of each activation value in their corresponding dispatcher row
         * @param padded_Nx     Input activations X size including padding
         * @param padded_Ny     Input activations Y size including padding
         * @param act_channels  Number of activations channels
         * @param NM_WIDTH      Width of the neuron memory row in bits
         */
        std::vector<std::vector<std::vector<int>>> generate_rowMap(int padded_Nx, int padded_Ny, int act_channels,
                int NM_WIDTH = 256);

        /* Return the optimal encoding for the given value
         * @param value     Value we want to encode WITHOUT the sign
         * @return          Value with the optimal encoding
         */
        uint16_t booth_encoding(uint16_t value);

        /* Return true if all the queues of activation bits are empty
         * @param offsets   Collection of activations with their explicit one positions in a queue
         * @return          True if empty
         */
        bool check_act_bits(const std::vector<std::queue<uint8_t>> &offsets);

    };

}

#endif //DNNSIM_TIMINGSIMULATOR_H
