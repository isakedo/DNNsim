#ifndef DNNSIM_BITPRAGMATIC_H
#define DNNSIM_BITPRAGMATIC_H

#include "Simulator.h"

#define ZERO_COUNT // Count zeroes as 1 cycle
#define BOOTH_ENCODING // Activate booth-like encoding
#define FC_MULTIPLEX_COLUMNS // Execute each mult-add in a different column

namespace core {

    template <typename T>
    class BitPragmatic : public Simulator<T> {

    private:

        /* Number of concurrent multiplications per PE */
        const uint32_t N_LANES;

        /* Number of columns */
        const uint32_t N_COLUMNS;

        /* Number of rows */
        const uint32_t N_ROWS;

        /* Bits of the first stage in the two stages shifting */
        const uint32_t BITS_FIRST_STAGE;

        /* Number of registers per SIP */
        const uint32_t COLUMN_REGISTERS;

        /* Diffy simulation */
        const bool DIFFY;

        /* Compute number of one bit multiplications given an activation
         * @param act           Activation
         * @param network_bits  Max bits network
         * @return              Number of one bit multiplications
         */
        uint8_t computePragmaticBitsPE(uint16_t act, int network_bits);

        /* Compute number of cycles for a two stage shift pragmatic PE
         * @param offsets   Explicit position for the ones for all the values
         * @return          Number of cycles
         */
        uint8_t computePragmaticPE(const std::vector<std::queue<uint8_t>> &offsets);

        /* Compute cycles for one column of pragmatic
         * @param batch         Current number of batch
         * @param recursion     Current recursion for LSTM
         * @param act_x         X position in the input activations
         * @param act_y         Y position in the input activations
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param max_channel   Maximum number of channels
         * @param lstm          True if it is LSTM layer
         * @return              Number of cycles
         */
        uint8_t computePragmaticColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
                int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel, bool lstm);

        /* Compute cycles for pragmatic tile
         * @param batch                 Current number of batch
         * @param list_act_x            X position for the set of input windows
         * @param list_act_y            Y position for the set of input windows
         * @param kernel_x              X position in the kernel window
         * @param kernel_y              Y position in the kernel window
         * @param init_channel          Starting index for the channel
         * @param stride                Stride of the current layer
         * @param padded_act            Set of padded input activations
         * @param act_mask              Position of the activations sign bit
         * @param max_channel           Maximum number of channels
         * @param cycles_per_col        Number of cycles per column (Overwritten)
         * @param end_previous_pallet   Cycle when the previous pallet finishes (Overwritten)
         * @param stats                 Statistics to fill 
         */
        void computePragmaticTile(int batch, const std::vector<int> &list_act_x, const std::vector<int> &list_act_y,
                int kernel_x, int kernel_y, int init_channel, int stride, const cnpy::Array<T> &padded_act,
                int act_max, int max_channel, std::vector<uint32_t> &cycles_per_col,
                std::vector<uint32_t> &end_previous_pallet, sys::Statistics::Stats &stats);

        /* Compute cycles for laconic tile
         * @param batch                 Current number of batch
         * @param list_act_x            X position for the set of input windows
         * @param list_act_y            Y position for the set of input windows
         * @param kernel_x              X position in the kernel window
         * @param kernel_y              Y position in the kernel window
         * @param init_filter           Starting index for the filter
         * @param stride                Stride of the current layer
         * @param padded_act            Set of padded input activations
         * @param wgt                   Set of weights
         * @param cycles_per_col        Number of cycles per column (Overwritten)
         * @param end_previous_pallet   Cycle when the previous pallet finishes (Overwritten)
         * @param stats                 Statistics to fill
         */
        void computePragmatic2DTile(int batch, const std::vector<int> &list_act_x,const std::vector<int> &list_act_y,
                int kernel_x, int kernel_y, int init_filter, int stride, const cnpy::Array<T> &padded_act,
                const cnpy::Array<T> &wgt, int max_filter, std::vector<uint32_t> &cycles_per_col,
                std::vector<uint32_t> &end_previous_pallet, sys::Statistics::Stats &stats);

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the timing for a 2D convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution2D(const Layer<T> &layer, sys::Statistics::Stats &stats);

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
         * @param _N_LANES              Number of concurrent multiplications per PE
         * @param _N_COLUMNS            Number of columns
         * @param _N_ROWS               Number of rows
         * @param _BITS_FIRST_STAGE     Bits of the first stage in the two stages shifting
         * @param _COLUMN_REGISTERS     Number of registers per SIP
         * @param _DIFFY                Enable Diffy
         * @param _N_THREADS            Number of parallel threads for multi-threading execution
         * @param _FAST_MODE            Enable fast mode to simulate only one image
         */
        BitPragmatic(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _BITS_FIRST_STAGE,
                uint32_t _COLUMN_REGISTERS, bool _DIFFY, uint8_t _N_THREADS, bool _FAST_MODE) :
                Simulator<T>(_N_THREADS,_FAST_MODE), N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS),
                BITS_FIRST_STAGE(_BITS_FIRST_STAGE), COLUMN_REGISTERS(_COLUMN_REGISTERS), DIFFY(_DIFFY) {}

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

#endif //DNNSIM_BITPRAGMATIC_H
