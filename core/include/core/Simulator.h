#ifndef DNNSIM_SIMULATOR_H
#define DNNSIM_SIMULATOR_H

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

        /* Number of parallel cores */
        const int N_THREADS;

        /* Enable fast mode: only one image */
        const bool FAST_MODE = false;

        /* Iterate set of windows in groups
         * @param out_x         Output activations X size
         * @param out_y         Output activations Y size
         * @param list_x        X position for the set of input windows (Overwritten)
         * @param list_y        Y position for the set of input windows (Overwritten)
         * @param x_counter     X input window counter (Overwritten)
         * @param y_counter     Y input window counter (Overwritten)
         * @param max_windows   Maximum number of windows (Number of columns in the accelerator)
         * @return              Return false when all input windows are read
         */
        bool iterateWindows(long out_x, long out_y, std::vector<int> &list_x, std::vector<int> &list_y,
                int &x_counter, int &y_counter, int max_windows = 16);

        /* Return the optimal encoding for the given value
         * @param value     Value to encode WITHOUT the sign
         * @return          Value with the optimal encoding
         */
        uint16_t booth_encoding(uint16_t value);

        /* Return the minimum and maximum index position for a given value
         * @param value     Value to get the indexes
         * @return          Minimum and maximum indexes
         */
        std::tuple<uint8_t,uint8_t> minMax(uint16_t value);

        /* Return the number of effectual bits for a given value
         * @param value     Value to get the effectual bits
         * @return          Number of effectual bits
         */
        uint8_t effectualBits(uint16_t value);

        /* Return true if all the queues of activation bits are empty
         * @param offsets   Collection of activations with their explicit one positions in a queue
         * @return          True if empty
         */
        bool check_act_bits(const std::vector<std::queue<uint8_t>> &offsets);

        /* Return value into sign-magnitude representation
         * @param two_comp  Signed value in two complement
         * @param mask      Mask with one bit for the bit position
         * @return          Value in sign-maginutde
         */
        uint16_t sign_magnitude(short two_comp, uint16_t mask);

    public:

        /* Constructor
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        Simulator(uint8_t _N_THREADS, bool _FAST_MODE) : N_THREADS(_N_THREADS), FAST_MODE(_FAST_MODE) {}

        /* Calculate the sparsity in the network
         * @param network   Network we want to check
         */
        void sparsity(const Network<T> &network);

        /* Calculate the bit-sparsity in the network. Assumes two-complement
         * @param network   Network we want to check
         */
        void bit_sparsity(const Network<T> &network);

    };

}

#endif //DNNSIM_SIMULATOR_H
