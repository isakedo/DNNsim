#ifndef DNNSIM_DYNAMICSTRIPESFP_H
#define DNNSIM_DYNAMICSTRIPESFP_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class DynamicStripesFP : public Simulator<T> {

    private:

        /* Compute the average width for a layer
         * @param network       Network we want to check
         * @param layer_it      Index pointing the current layer in the network
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         * @param epoch		    Current epoch
		 * @param epochs        Number of epochs
         */
        void computeAvgWidthLayer(const Network<T> &network, int layer_it, sys::Statistics::Stats &stats,
                int network_bits, int epoch, int epochs);

    public:

        /* Constructor
         * @param _N_THREADS                Number of parallel threads for multi-threading execution
         * @param _FAST_MODE                Enable fast mode to simulate only one image
         */
        DynamicStripesFP(uint8_t _N_THREADS, bool _FAST_MODE) :Simulator<T>(_N_THREADS,_FAST_MODE) {}

        /* Calculate the average width in the network transformed to sign-magnitude
         * @param network   Network we want to check
		 * @param stats		Shared stats for the epochs
         * @param epoch		Current epoch
		 * @param epochs    Number of epochs
         */
        void average_width(const Network<T> &network, sys::Statistics::Stats &stats, int epoch, int epochs);

    };

}

#endif //DNNSIM_DYNAMICSTRIPESFP_H
