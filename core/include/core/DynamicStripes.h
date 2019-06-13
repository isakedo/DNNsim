#ifndef DNNSIM_DYNAMICSTRIPES_H
#define DNNSIM_DYNAMICSTRIPES_H

#include "Simulator.h"

#define FC_MULTIPLEX_COLUMNS // Execute each mult-add in a different column
#define WEIGHT_LANES 16 // Number of weight lanes

namespace core {

    template <typename T>
    class DynamicStripes : public Simulator<T> {

    private:

        /* Number of columns */
        const int N_COLUMNS;

        /* Number of rows */
        const int N_ROWS;

        /* Number of activations per group */
        const int PRECISION_GRANULARITY;

        /* Number of registers per SIP */
        const int COLUMN_REGISTERS;

        /* Bits per PE */
        const int BITS_PE;

        /* Calculate only the leading bit for dynamic precisions */
        const bool LEADING_BIT;

        /* Diffy simulation */
        const bool DIFFY;

        /* Compute number of one bit multiplications
         * @param layer_prec    Layer precision
         * @param network_bits  Max bits network
         * @return              Number of one bit multiplications
         */
        uint16_t computeDynamicStripesBitsPE(uint8_t layer_prec, int network_bits);

        /* Compute cycles for dynamic stripes column
         * @param batch         Current number of batch
         * @param recursion     Current recursion for LSTM
         * @param act_x         X position in the input activations
         * @param act_y         Y position in the input activations
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param layer_prec    Activations precision per layer
         * @param init_channel  Starting index for the channel
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param act_mask      Position of the activations sign bit
         * @param max_channel   Maximum number of channels
         * @param lstm          True if it is LSTM layer
         * @return              Number of cycles
         */
        uint8_t computeDynamicStripesColumn(int batch, int recursion, int act_x, int act_y, int kernel_x, int kernel_y,
                int init_channel, int stride, const cnpy::Array<T> &padded_act, int act_mask, int max_channel,
                bool lstm);

        /* Compute cycles for dynamic stripes tile
         * @param batch                 Current number of batch
         * @param list_act_x            X position for the set of input windows
         * @param list_act_y            Y position for the set of input windows
         * @param kernel_x              X position in the kernel window
         * @param kernel_y              Y position in the kernel window
         * @param layer_prec            Activations precision per layer
         * @param init_channel          Starting index for the channel
         * @param stride                Stride of the current layer
         * @param padded_act            Set of padded input activations
         * @param act_mask              Position of the activations sign bit
         * @param max_channel           Maximum number of channels
         * @param cycles_per_group      Number of cycles per column (Overwritten)
         * @param end_previous_pallet   Cycle when the previous pallet finishes (Overwritten)
         * @param stats                 Statistics to fill
         */
        void computeDynamicStripesTile(int batch, const std::vector<int> &list_act_x, const std::vector<int>
                &list_act_y, int kernel_x, int kernel_y, int init_channel, int stride, const cnpy::Array<T> &padded_act,
                int act_mask, int max_channel, std::vector<uint32_t> &cycles_per_group,
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
         * @param act_mask              Position of the activations sign bit
         * @param cycles_per_group      Number of cycles per column (Overwritten)
         * @param end_previous_pallet   Cycle when the previous pallet finishes (Overwritten)
         * @param stats                 Statistics to fill
         */
        void computeDynamicStripes2DTile(int batch, const std::vector<int> &list_act_x,
                const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_filter, int stride,
                const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt, int act_mask, int max_filter,
                std::vector<uint32_t> &cycles_per_group, std::vector<uint32_t> &end_previous_pallet,
                sys::Statistics::Stats &stats);

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

        /* Compute average width for activations for laconic tile
         * @param batch         Current number of batch
         * @param recursion     Current recursion for LSTM
         * @param list_act_x    X position for the set of input windows
         * @param list_act_y    Y position for the set of input windows
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param start_group   Starting channel of the group
         * @param max_channel   Maximum number of channels
         * @param act_mask      Position of the activations sign bit
         * @param lstm          True if it is LSTM layer
         * @return              Average width per group
         */
        std::vector<double> computeAvgWidthDynamicStripesActTile(int batch, int recursion,
                const std::vector<int> &list_act_x, const std::vector<int> &list_act_y, int kernel_x, int kernel_y,
                int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel, int act_mask,
                bool lstm);

        /* Compute average width for weights for laconic tile
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param init_filter   Starting index for the filter
         * @param wgt           Set of weights
         * @param max_channel   Maximum number of channels
         * @param max_filter    Maximum number of filters
         * @param wgt_mask      Position of the weights sign bit
         * @return              Average width per group
         */
        std::vector<double> computeAvgWidthDynamicStripesWgtTile(int kernel_x, int kernel_y, int init_channel,
                int init_filter, const cnpy::Array<T> &wgt, int max_channel, int max_filter, int wgt_mask);

        /* Compute the average width for a layer
         * @param layer         Layer for which we want to calculate the outputs
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        void computeAvgWidthLayer(const Layer<T> &layer, sys::Statistics::Stats &stats, int network_bits);

    public:

        /* Constructor
         * @param _N_COLUMNS                Number of columns
         * @param _N_ROWS                   Number of rows
         * @param _PRECISION_GRANULARITY    Granularity for dynamic precisions
         * @param _COLUMN_REGISTERS         Number of registers per SIP
         * @param _BITS_PE                  Number of bits per PE
         * @param _LEADING_BIT              Calculate only the leading bit for dynamic precisions
         * @param _DIFFY                    Enable Diffy
         * @param _N_THREADS                Number of parallel threads for multi-threading execution
         * @param _FAST_MODE                Enable fast mode to simulate only one image
         */
        DynamicStripes(int _N_COLUMNS, int _N_ROWS, const int &_PRECISION_GRANULARITY, int _COLUMN_REGISTERS,
                int _BITS_PE, bool _LEADING_BIT, bool _DIFFY, uint8_t _N_THREADS, bool _FAST_MODE) :
                Simulator<T>(_N_THREADS,_FAST_MODE), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS),
                PRECISION_GRANULARITY(_PRECISION_GRANULARITY), COLUMN_REGISTERS(_COLUMN_REGISTERS), BITS_PE(_BITS_PE),
                LEADING_BIT(_LEADING_BIT), DIFFY(_DIFFY) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

        /* Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const Network<T> &network);

        /* Calculate the average width in the network transformed to sign-magnitude
         * @param network   Network we want to check
         */
        void average_width(const Network<T> &network);

    };

}

#endif //DNNSIM_DYNAMICSTRIPES_H
