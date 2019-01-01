#ifndef DNNSIM_LACONIC_H
#define DNNSIM_LACONIC_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class Laconic : public Simulator<T> {

    private:

        /* Compute number of one bit multiplications given a weights and an activation
         * @param act       Activation
         * @param wgt       Weight
         * @return          Number of one bit multiplications
         */
        uint8_t computeLaconicPE(uint16_t act, uint16_t wgt);

        /* Compute cycles for pragmatic tile
         * @param batch         Current number of batch
         * @param act_x         X position for the input window
         * @param act_y         Y position for the input window
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param init_filter   Starting index for the filter
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param padded_act    Set of weights
         * @param max_channel   Maximum number of channels
         * @param max_filter    Maximum number of filters
         * @return              Number of cycles
         */
        uint8_t computeLaconicColumn(int batch, int act_x, int act_y, int kernel_x, int kernel_y, int init_channel,
                int init_filter, int stride, const cnpy::Array<T> &padded_act, const cnpy::Array<T> &wgt,
                int start_group, int max_channel, int max_filter);

        /* Compute cycles for pragmatic tile
         * @param batch         Current number of batch
         * @param list_act_x    X position for the set of input windows
         * @param list_act_y    Y position for the set of input windows
         * @param kernel_x      X position in the kernel window
         * @param kernel_y      Y position in the kernel window
         * @param init_channel  Starting index for the channel
         * @param init_filter   Starting index for the filter
         * @param stride        Stride of the current layer
         * @param padded_act    Set of padded input activations
         * @param padded_act    Set of weights
         * @param max_channel   Maximum number of channels
         * @param max_filter    Maximum number of filters
         * @return              Number of cycles
         */
        uint8_t computeLaconicTile(int batch, std::vector<int> &list_act_x, std::vector<int> &list_act_y, int kernel_x,
                int kernel_y, int init_channel, int init_filter, int stride, const cnpy::Array<T> &padded_act,
                const cnpy::Array<T> &wgt, int start_group, int max_channel, int max_filter);

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

        /* Compute the work reduction for a convolutional layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the work reduction for a inner product layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats);


    public:

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

        /* Calculate work reduction for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const Network<T> &network);

    };

}

#endif //DNNSIM_LACONIC_H
