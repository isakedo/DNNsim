#ifndef DNNSIM_BITPRAGMATIC_H
#define DNNSIM_BITPRAGMATIC_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class BitPragmatic : public Simulator<T> {

    private:

        /* Number of columns */
        const int N_COLUMNS;

        /* Number of rows */
        const int N_ROWS;

        /* Bits of the first stage in the two stages shifting */
        const int BITS_FIRST_STAGE;

        /* Compute number of one bit multiplications given an activation
         * @param act       Activation
         * @return          Number of one bit multiplications
         */
        uint8_t computePragmaticBitsPE(uint16_t act);

        /* Compute number of cycles for a two stage shift pragmatic PE
         * @param offsets   Explicit position for the ones for all the values
         * @return          Number of cycles
         */
        uint8_t computePragmaticPE(const std::vector<std::queue<uint8_t>> &offsets);

        /* Compute cycles for one column of pragmatic
         * @param batch         Current number of batch
         * @param act_x         X position in the input activations
         * @param act_y         Y position in the input activations
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param max_channel   Maximum number of channels
         * @return              Number of cycles
         */
        uint8_t computePragmaticColumn(int batch, int act_x, int act_y, int kernel_x, int kernel_y, int init_channel,
                int stride, const cnpy::Array<T> &padded_act, int max_channel);

        /* Compute cycles for pragmatic tile
         * @param batch                 Current number of batch
         * @param list_act_x            X position for the set of input windows
         * @param list_act_y            Y position for the set of input windows
         * @param kernel_x              X position in the kernel window
         * @param kernel_y              Y position in the kernel window
         * @param init_channel          Starting index for the channel
         * @param stride                Stride of the current layer
         * @param padded_act            Set of padded input activations
         * @param max_channel           Maximum number of channels
         * @param cycles_per_col        Number of cycles per column. Updated
         * @param end_previous_pallet   Cycle when the previous pallet finishes. Updated
         */
        void computePragmaticTile(int batch, std::vector<int> &list_act_x, std::vector<int> &list_act_y, int kernel_x,
                int kernel_y, int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel,
                std::vector<uint32_t> &cycles_per_col, uint32_t &end_previous_pallet);

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
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the potentials for a inner product layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the potentials for a convolutional layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computeMemAccessesConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

    public:

        /* Constructor
         * @param _N_COLUMNS            Number of columns
         * @param _N_ROWS               Number of rows
         * @param _BITS_FIRST_STAGE     Bits of the first stage in the two stages shifting
         */
        BitPragmatic(int _N_COLUMNS, int _N_ROWS, int _BITS_FIRST_STAGE) : N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS),
            BITS_FIRST_STAGE(_BITS_FIRST_STAGE) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

        /* Calculate work reduction for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const Network<T> &network);

        /* Calculate the number of memory accesses
         * @param network   Network we want to simulate
         */
        void memoryAccesses(const Network<T> &network);

    };

}

#endif //DNNSIM_BITPRAGMATIC_H
