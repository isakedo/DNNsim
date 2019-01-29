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

typedef std::vector<std::vector<std::vector<std::vector<std::tuple<int,int,int,int>>>>> idxMap;
typedef std::vector<std::vector<std::vector<int>>> rowIdxMap;

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

        /* Calculate the 3D mapping of each activation value in their corresponding dispatcher row
         * @param padded_Nx     Input activations X size including padding
         * @param padded_Ny     Input activations Y size including padding
         * @param act_channels  Number of activations channels
         * @param NM_WIDTH      Width of the neuron memory row in bits
         */
        rowIdxMap generate_rowMap(int padded_Nx, int padded_Ny, int act_channels, int NM_WIDTH = 256);

        /* Calculate and store the index mapping of each value in the data array
         * @param data   Array that we want to generate the idx map
         */
        idxMap generate_idxMap(const cnpy::Array<T> &data);

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

        /* Constructor
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        Simulator(uint8_t _N_THREADS, bool _FAST_MODE) : N_THREADS(_N_THREADS), FAST_MODE(_FAST_MODE) {}

        /* Virtual destructor, force class to be abstract */
        virtual ~Simulator() = default;

    };

}

#endif //DNNSIM_SIMULATOR_H
