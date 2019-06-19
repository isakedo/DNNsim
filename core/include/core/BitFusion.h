#ifndef DNNSIM_BITFUSION_H
#define DNNSIM_BITFUSION_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class BitFusion : public Simulator<T> {

    private:

        /* Systolic array width */
        const uint32_t M;

        /* Systolic array height */
        const uint32_t N;

        /* Maximum precision */
        const uint32_t PMAX;

        /* Maximum precision */
        const uint32_t PMIN;

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
         * @param layer         Layer for which we want to calculate potentials
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,int network_bits);

        /* Compute the potentials for a inner product layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         * @param network_bits  Max bits network
         */
        void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats,int network_bits);

    public:

        /* Constructor
         * @param _M            Matrix width
         * @param _N            Matrix height
         * @param _PMAX         Maximum precision
         * @param _PMIN         Minimum precision
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        BitFusion(uint32_t _M, uint32_t _N, uint32_t _PMAX, uint32_t _PMIN, uint8_t _N_THREADS, bool _FAST_MODE) :
            Simulator<T>(_N_THREADS,_FAST_MODE), M(_M), N(_N), PMAX(_PMAX), PMIN(_PMIN) {}

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
