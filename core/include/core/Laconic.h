#ifndef DNNSIM_LACONIC_H
#define DNNSIM_LACONIC_H

#include "Simulator.h"

#define ZERO_COUNT // Count zeroes as 1 cycle
#define BOOTH_ENCODING // Activate booth-like encoding

namespace core {

    /**
     * Laconic simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class Laconic : public Simulator<T> {

    private:

        /** Number of concurrent multiplications per PE */
        const int N_LANES;

        /** Number of columns */
        const int N_COLUMNS;

        /** Number of rows */
        const int N_ROWS;

        /** Number of tiles */
        const uint32_t N_TILES;

        /** Compute number of one bit multiplications given a weights and an activation
         * @param act       Activation
         * @param wgt       Weight
         * @return          Number of one bit multiplications
         */
        uint8_t computeLaconicPE(uint16_t act, uint16_t wgt);

        /** Compute cycles for one column of laconic
         * @param batch         Current number of batch
         * @param recursion     Current recursion for LSTM
         * @param act_x         X position for the input window
         * @param act_y         Y position for the input window
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param init_filter   Starting index for the filter
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param wgt           Set of weights
         * @param max_channel   Maximum number of channels
         * @param max_filter    Maximum number of filters
         * @param lstm          True if it is LSTM layer
         * @param conv2D        True if 2D convolution
         * @return              Number of cycles
         */
        uint8_t computeLaconicColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
                int init_channel, int init_filter, int stride, const base::Array<T> &padded_act,
                const base::Array<T> &wgt, int start_group, int max_channel, int max_filter, bool lstm, bool conv2D);

        /** Compute cycles for laconic tile
         * @param batch         Current number of batch
         * @param list_act_x    X position for the set of input windows
         * @param list_act_y    Y position for the set of input windows
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param init_filter   Starting index for the filter
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param wgt           Set of weights
         * @param start_group   Starting channel of the group
         * @param max_channel   Maximum number of channels
         * @param max_filter    Maximum number of filters
         * @param conv2D        True if 2D convolution
         * @param stall_cycles  Stall cycles stat (Overwritten)
         * @return              Number of cycles
         */
        uint8_t computeLaconicTile(int batch, const std::vector<int> &list_act_x, const std::vector<int> &list_act_y,
                int kernel_x, int kernel_y, int init_channel, int init_filter, int stride,
                const base::Array<T> &padded_act, const base::Array<T> &wgt, int start_group, int max_channel,
                int max_filter, bool conv2D, uint64_t &stall_cycles);

    public:

        /** Constructor
         * @param _N_LANES      Number of concurrent multiplications per PE
         * @param _N_COLUMNS    Number of columns
         * @param _N_ROWS       Number of rows
         * @param _N_TILES      Number of tiles
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         * @param _QUIET        Avoid std::out messages
         * @param _CHECK        Check the correctness of the simulations
         */
        Laconic(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES, uint8_t _N_THREADS,
                bool _FAST_MODE, bool _QUIET, bool _CHECK) : Simulator<T>(_N_THREADS,_FAST_MODE,_QUIET,_CHECK),
                N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), N_TILES(_N_TILES) {}

        /** Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const base::Network<T> &network);

        /** Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const base::Network<T> &network);

    };

}

#endif //DNNSIM_LACONIC_H
