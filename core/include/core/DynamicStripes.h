#ifndef DNNSIM_DYNAMICSTRIPES_H
#define DNNSIM_DYNAMICSTRIPES_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class DynamicStripes : public Simulator<T> {

    private:

        /** Number of concurrent multiplications per PE */
        const uint32_t N_LANES;

        /** Number of columns */
        const uint32_t N_COLUMNS;

        /** Number of rows */
        const uint32_t N_ROWS;

        /** Number of rows */
        const uint32_t N_TILES;

        /** Number of activations per group */
        const uint32_t PRECISION_GRANULARITY;

        /** Number of registers per SIP */
        const uint32_t COLUMN_REGISTERS;

        /** Bits per PE */
        const uint32_t BITS_PE;

        /** Calculate only the leading bit for dynamic precisions */
        const bool LEADING_BIT;

        /** Diffy simulation */
        const bool DIFFY;

        /** Compute number of one bit multiplications
         * @param layer_prec    Layer precision
         * @param network_bits  Max bits network
         * @return              Number of one bit multiplications
         */
        uint16_t computeDynamicStripesBitsPE(uint8_t layer_prec, int network_bits);

        /** Compute cycles for dynamic stripes column
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
                int init_channel, int stride, const base::Array<T> &padded_act, uint16_t act_mask, int max_channel,
                bool lstm);

        /** Compute cycles for dynamic stripes tile
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
         * @param stall_cycles          Stall cycles stat (Overwritten)
         */
        void computeDynamicStripesTile(int batch, const std::vector<int> &list_act_x, const std::vector<int>
                &list_act_y, int kernel_x, int kernel_y, int init_channel, int stride, const base::Array<T> &padded_act,
                uint16_t act_mask, int max_channel, std::vector<uint32_t> &cycles_per_group,
                std::vector<uint32_t> &end_previous_pallet, uint64_t &stall_cycles);


        /** Compute cycles for laconic tile
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
         * @param stall_cycles          Stall cycles stat (Overwritten)
         */
        void computeDynamicStripes2DTile(int batch, const std::vector<int> &list_act_x,
                const std::vector<int> &list_act_y, int kernel_x, int kernel_y, int init_filter, int stride,
                const base::Array<T> &padded_act, const base::Array<T> &wgt, uint16_t act_mask, int max_filter,
                std::vector<uint32_t> &cycles_per_group, std::vector<uint32_t> &end_previous_pallet,
                uint64_t &stall_cycles);

        /** Compute average width for activations for laconic tile
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
                int init_channel, int stride, const base::Array<T> &padded_act, int max_channel, uint16_t act_mask,
                bool lstm);

        /** Compute average width for weights for laconic tile
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
                int init_filter, const base::Array<T> &wgt, int max_channel, int max_filter, uint16_t wgt_mask);

    public:

        /** Constructor
         * @param _N_LANES                  Number of concurrent multiplications per PE
         * @param _N_COLUMNS                Number of columns
         * @param _N_ROWS                   Number of rows
         * @param _N_TILES                  Number of tiles
         * @param _PRECISION_GRANULARITY    Granularity for dynamic precisions
         * @param _COLUMN_REGISTERS         Number of registers per SIP
         * @param _BITS_PE                  Number of bits per PE
         * @param _LEADING_BIT              Calculate only the leading bit for dynamic precisions
         * @param _DIFFY                    Enable Diffy
         * @param _N_THREADS                Number of parallel threads for multi-threading execution
         * @param _FAST_MODE                Enable fast mode to simulate only one image
         * @param _QUIET                    Avoid std::out messages
         * @param _CHECK                    Check the correctness of the simulations
         */
        DynamicStripes(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES,
                uint32_t _PRECISION_GRANULARITY, uint32_t _COLUMN_REGISTERS, uint32_t _BITS_PE, bool _LEADING_BIT,
                bool _DIFFY, uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) : Simulator<T>(_N_THREADS,
                _FAST_MODE,_QUIET,_CHECK), N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), N_TILES(_N_TILES),
                PRECISION_GRANULARITY(_PRECISION_GRANULARITY), COLUMN_REGISTERS(_COLUMN_REGISTERS), BITS_PE(_BITS_PE),
                LEADING_BIT(_LEADING_BIT), DIFFY(_DIFFY) {}

        /** Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const base::Network<T> &network);

        /** Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const base::Network<T> &network);

        /** Calculate the average width in the network transformed to sign-magnitude
         * @param network   Network we want to check
         */
        void average_width(const base::Network<T> &network);

        /** Simulate on-chip memory dynamic width storage
         * @param network   Network we want to check
         */
        void on_chip(const base::Network<T> &network);

    };

}

#endif //DNNSIM_DYNAMICSTRIPES_H
