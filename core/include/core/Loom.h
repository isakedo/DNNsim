#ifndef DNNSIM_LOOM_H
#define DNNSIM_LOOM_H

#include "Simulator.h"

#define FC_MULTIPLEX_COLUMNS // Execute each mult-add in a different column
#define WEIGHT_LANES 16 // Number of weight lanes

namespace core {

    template <typename T>
    class Loom : public Simulator<T> {

    private:

        /* Number of columns */
        const int N_COLUMNS;

        /* Number of rows */
        const int N_ROWS;

        /* Number of activations per group */
        const int PRECISION_GRANULARITY;

        /* Number of bits in series that the PE process */
        const int PE_SERIAL_BITS;

        /* Calculate also the minor bit for dynamic precisions */
        const bool MINOR_BIT;

        /* Compute number of one bit multiplications given a weights and an activation
         * @param act_prec  Activation layer precision
         * @param wgt_prec  Weight layer precision
         * @return          Number of one bit multiplications
         */
        uint8_t computeLoomBitsPE(uint8_t act_prec, uint8_t wgt_prec);

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
         * @return              Number of cycles
         */
        uint8_t computeLoomColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
                int init_channel, int init_filter, int stride, const cnpy::Array<T> &padded_act,
                const cnpy::Array<T> &wgt, int start_group, int max_channel, int max_filter, int act_mask, int wgt_mask,
                bool lstm);

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
         * @return              Number of cycles
         */
        uint8_t computeLoomTile(int batch, const std::vector<int> &list_act_x, const std::vector<int> &list_act_y,
                int kernel_x, int kernel_y, int init_channel, int init_filter, int stride,
                const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int start_group, int max_channel,
                int max_filter, int act_mask, int wgt_mask);

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

    public:

        /* Constructor
         * @param _N_COLUMNS                Number of columns
         * @param _N_ROWS                   Number of rows
         * @param _PRECISION_GRANULARITY    Granularity for dynamic precisions
         * @param _PE_SERIAL_BITS           Number of bits in series that the PE process
         * @param _MINOR_BIT                Calculate also the minor bit for dynamic precisions
         * @param _N_THREADS                Number of parallel threads for multi-threading execution
         * @param _FAST_MODE                Enable fast mode to simulate only one image
         */
        Loom(int _N_COLUMNS, int _N_ROWS, int _PRECISION_GRANULARITY, int _PE_SERIAL_BITS, bool _MINOR_BIT,
                uint8_t _N_THREADS, bool _FAST_MODE) : Simulator<T>(_N_THREADS,_FAST_MODE), N_COLUMNS(_N_COLUMNS),
                N_ROWS(_N_ROWS), PRECISION_GRANULARITY(_PRECISION_GRANULARITY), PE_SERIAL_BITS(_PE_SERIAL_BITS),
                MINOR_BIT(_MINOR_BIT) {}

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

#endif //DNNSIM_LOOM_H
