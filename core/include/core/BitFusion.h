#ifndef DNNSIM_BITFUSION_H
#define DNNSIM_BITFUSION_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class BitFusion : public Simulator<T> {

    private:

        /* Number of PEs */
        const int NUM_PE;

        /* Compute number of one bit multiplications
         * @param act_rounded_precision Rounded precision for the activations to the next power of two
         * @param wgt_rounded_precision Rounded precision for the weights to the next power of two
         * @return                      Number of one bit multiplications
         */
        static inline uint16_t computeBitFusionBitsPE(uint8_t act_rounded_precision, uint8_t wgt_rounded_precision);

        /* Compute the timing for a layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeLayer(const Layer<T> &layer, sys::Statistics::Stats &stats);

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
         * @param _NUM_PE       Number of PEs
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        BitFusion(int _NUM_PE, uint8_t _N_THREADS, bool _FAST_MODE) : Simulator<T>(_N_THREADS,_FAST_MODE),
                NUM_PE(_NUM_PE) {}

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

#endif //DNNSIM_BITFUSION_H
