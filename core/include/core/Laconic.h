#ifndef DNNSIM_LACONIC_H
#define DNNSIM_LACONIC_H

#include "Simulator.h"

#define ZERO_COUNT // Count zeroes as 1 cycle
#define BOOTH_ENCODING // Activate booth-like encoding
#define FC_MULTIPLEX_COLUMNS // Execute each mult-add in a different column

namespace core {

    template <typename T>
    class Laconic : public Simulator<T> {

    private:

        /* Number of concurrent multiplications per PE */
        const int N_LANES;

        /* Number of columns */
        const int N_COLUMNS;

        /* Number of rows */
        const int N_ROWS;

        /* Compute number of one bit multiplications given a weights and an activation
         * @param act       Activation
         * @param wgt       Weight
         * @return          Number of one bit multiplications
         */
        uint8_t computeLaconicPE(uint16_t act, uint16_t wgt);

        /* Compute cycles for one column of laconic
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
                int init_channel, int init_filter, int stride, const cnpy::Array<T> &padded_act,
                const cnpy::Array<T> &wgt, int start_group, int max_channel, int max_filter, bool lstm, bool conv2D);

        /* Compute cycles for laconic tile
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
         * @param stats         Statistics to fill
         * @return              Number of cycles
         */
        uint8_t computeLaconicTile(int batch, const std::vector<int> &list_act_x, const std::vector<int> &list_act_y,
                int kernel_x, int kernel_y, int init_channel, int init_filter, int stride,
                const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int start_group, int max_channel,
                int max_filter, bool conv2D, sys::Statistics::Stats &stats);

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the timing for a fully-connected layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the potentials for a convolutional layer
         * @param layer         Layer for which we want to calculate potentials
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,int network_bits);

        /* Compute the potentials for a inner product layer
         * @param layer         Layer for which we want to calculate potentials
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats,int network_bits);

    public:

        /* Constructor
         * @param _N_LANES      Number of concurrent multiplications per PE
         * @param _N_COLUMNS    Number of columns
         * @param _N_ROWS       Number of rows
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        Laconic(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint8_t _N_THREADS, bool _FAST_MODE) :
                Simulator<T>(_N_THREADS,_FAST_MODE), N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

        /* Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const Network<T> &network);

    };

}

#endif //DNNSIM_LACONIC_H
