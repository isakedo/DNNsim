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

        /* Number of activations per group: Tile, SIP */
        std::string PRECISION_GRANULARITY;

        /* Number of registers per SIP */
        const int COLUMN_REGISTERS;

        /* Compute number of one bit multiplications
         * @param layer_prec    Layer precision
         * @return              Number of one bit multiplications
         */
        static inline uint16_t computeDynamicStripesBitsPE(uint8_t layer_prec);

        /* Compute cycles for dynamic stripes column
         * @param batch         Current number of batch
         * @param act_x         X position in the input activations
         * @param act_y         Y position in the input activations
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param layer_prec    Activations precision per layer
         * @param init_channel  Starting index for the channel
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param max_channel   Maximum number of channels
         * @return              Number of cycles
         */
        uint8_t computeDynamicStripesColumn(int batch, int act_x, int act_y, int kernel_x, int kernel_y,
                int init_channel, int stride, const cnpy::Array<T> &padded_act, int max_channel);

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
         * @param max_channel           Maximum number of channels
         * @param cycles_per_col        Number of cycles per column (Overwritten)
         * @param end_previous_pallet   Cycle when the previous pallet finishes (Overwritten)
         */
        void computeDynamicStripesTile(int batch, const std::vector<int> &list_act_x, const std::vector<int>
                &list_act_y, int kernel_x, int kernel_y, int init_channel, int stride, const cnpy::Array<T> &padded_act,
                int max_channel, std::vector<uint32_t> &cycles_per_col, std::vector<uint32_t> &end_previous_pallet);

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
         * @param _COLUMN_REGISTERS         Number of registers per SIP
         * @param _N_THREADS                Number of parallel threads for multi-threading execution
         * @param _FAST_MODE                Enable fast mode to simulate only one image
         */
        DynamicStripes(int _N_COLUMNS, int _N_ROWS, const std::string &_PRECISION_GRANULARITY, int _COLUMN_REGISTERS,
                uint8_t _N_THREADS, bool _FAST_MODE) : Simulator<T>(_N_THREADS,_FAST_MODE), N_COLUMNS(_N_COLUMNS),
                N_ROWS(_N_ROWS), COLUMN_REGISTERS(_COLUMN_REGISTERS) {
           PRECISION_GRANULARITY = _PRECISION_GRANULARITY;
        }

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

#endif //DNNSIM_DYNAMICSTRIPES_H
