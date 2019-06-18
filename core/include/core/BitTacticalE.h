#ifndef DNNSIM_BITTACTICAL_E_H
#define DNNSIM_BITTACTICAL_E_H

#include "BitTactical.h"

#define ZERO_COUNT // Count zeroes as 1 cycle
#define BOOTH_ENCODING // Activate booth-like encoding
#define FC_MULTIPLEX_COLUMNS // Execute each mult-add in a different column

namespace core {

    template <typename T>
    class BitTacticalE : public BitTactical<T> {

    private:

        /* Bits of the first stage in the two stages shifting */
        const int BITS_FIRST_STAGE;

        /* Compute number of one bit multiplications given a weights and an activation
         * @param act               Activation
         * @param wgt               Weight
         * @param network_bits      Max bits network
         * @return                  Number of one bit multiplications
         */
        uint8_t computeTacticalEBitsPE(uint16_t act, uint16_t wgt, int network_bits);

        /* Compute number of cycles for a two stage shift pragmatic PE
         * @param offsets   Explicit position for the ones for all the values
         * @return          Number of cycles
         */
        uint8_t computeTacticalEPE(const std::vector<std::queue<uint8_t>> &offsets);

        /* Compute cycles for BitTacticalE column
         * @param batch             Current number of batch
         * @param recursion         Current recursion for LSTM
         * @param act_x             X position for the input window
         * @param act_y             Y position for the input window
         * @param stride            Stride of the current layer
         * @param padded_act        Set of padded input activations
         * @param dense_schedule    Data structure containing the weights
         * @param schedule_time     Time index for the scheduler
         * @param lstm              True if it is LSTM layer
         * @return                  Number of cycles
         */
        uint8_t computeTacticalEColumn(int batch, int recursion, int act_x, int act_y, int stride,
                const cnpy::Array<T> &padded_act, const schedule &dense_schedule, int schedule_time, bool lstm);

        /* Compute cycles for BitTacticalE tile
         * @param batch                 Current number of batch
         * @param list_act_x            X position for the set of input windows
         * @param list_act_y            Y position for the set of input windows
         * @param stride                Stride of the current layer
         * @param padded_act            Set of padded input activations
         * @param dense_schedule        Data structure containing the weights
         * @param schedule_time         Time index for the scheduler
         * @param cycles_per_col        Number of cycles per column (Overwritten)
         * @param end_previous_pallet   Cycle when the previous pallet finishes (Overwritten)
         * @param stats                 Statistics to fill
         */
        void computeTacticalETile(int batch, const std::vector<int> &list_act_x, const std::vector<int> &list_act_y,
                int stride, const cnpy::Array<T> &padded_act, const schedule &dense_schedule, int schedule_time,
                std::vector<uint32_t> &cycles_per_col, std::vector<uint32_t> &end_previous_pallet,
                sys::Statistics::Stats &stats);

        /* Compute the timing for a convolutional layer
         * @param layer                 Layer for which we want to calculate the outputs
         * @param stats                 Statistics to fill
         * @param proto_dense_schedule  Schedule read from protobuf file
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats,
                const schedule &proto_dense_schedule) override;

        /* Compute the timing for a fully-connected layer
         * @param layer                 Layer for which we want to calculate the outputs
         * @param stats                 Statistics to fill
         * @param proto_dense_schedule  Schedule read from protobuf file
         */
        void computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats,
                const schedule &proto_dense_schedule) override;

        /* Compute the potentials for a convolutional layer
         * @param layer         Layer for which we want to calculate potentials
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,int network_bits)
            override;

        /* Compute the potentials for a inner product layer
         * @param layer         Layer for which we want to calculate potentials
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats,int network_bits)
            override;

    public:

        /* Constructor
         * @param _N_LANES              Number of concurrent multiplications per PE
         * @param _N_COLUMNS            Number of columns
         * @param _N_ROWS               Number of rows
         * @param _BITS_FIRST_STAGE     Bits of the first stage in the two stages shifting
         * @param _COLUMN_REGISTERS     Number of registers per SIP
         * @param _LOOKAHEAD_D          Value for scheduler lookahead
         * @param _LOOKASIDE_H          Value for scheduler lookaside
         * @param _SEARCH_SHAPE         Type of search
         * @param _N_THREADS            Number of parallel threads for multi-threading execution
         * @param _FAST_MODE            Enable fast mode to simulate only one image
         */
        BitTacticalE(int _N_LANES, int _N_COLUMNS, int _N_ROWS, int _BITS_FIRST_STAGE, int _COLUMN_REGISTERS,
                int _LOOKAHEAD_H, int _LOOKASIDE_D, const char _SEARCH_SHAPE, uint8_t _N_THREADS, bool _FAST_MODE) :
                BitTactical<T>(_N_LANES,_N_COLUMNS,_N_ROWS,_COLUMN_REGISTERS,_LOOKAHEAD_H, _LOOKASIDE_D,_SEARCH_SHAPE,
                _N_THREADS,_FAST_MODE), BITS_FIRST_STAGE(_BITS_FIRST_STAGE) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network) override;

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         * @param schedules Dense schedules for the layer we want to simulate
         */
        void run(const Network<T> &network, const std::vector<schedule> &schedules) override;

        /* Calculate work reduction for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const Network<T> &network) override;

    };

}

#endif //DNNSIM_BITTACTICAL_E_H
