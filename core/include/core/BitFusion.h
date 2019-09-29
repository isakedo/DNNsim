#ifndef DNNSIM_BITFUSION_H
#define DNNSIM_BITFUSION_H

#include "Simulator.h"

namespace core {

    /**
     * BitFusion simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class BitFusion : public Simulator<T> {

    private:

        /** Systolic array width */
        const uint32_t M;

        /** Systolic array height */
        const uint32_t N;

        /** Maximum precision */
        const uint32_t PMAX;

        /** Maximum precision */
        const uint32_t PMIN;

        /** Compute number of one bit multiplications
         * @param act_rounded_precision Rounded precision for the activations to the next power of two
         * @param wgt_rounded_precision Rounded precision for the weights to the next power of two
         * @return                      Number of one bit multiplications
         */
        static inline uint16_t computeBitFusionBitsPE(uint8_t act_rounded_precision, uint8_t wgt_rounded_precision);

    public:

        /** Constructor
         * @param _M            Matrix width
         * @param _N            Matrix height
         * @param _PMAX         Maximum precision
         * @param _PMIN         Minimum precision
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         * @param _QUIET        Avoid std::out messages
         * @param _CHECK        Check the correctness of the simulations
         */
        BitFusion(uint32_t _M, uint32_t _N, uint32_t _PMAX, uint32_t _PMIN, uint8_t _N_THREADS, bool _FAST_MODE,
                bool _QUIET, bool _CHECK) : Simulator<T>(_N_THREADS,_FAST_MODE,_QUIET,_CHECK), M(_M), N(_N),
                PMAX(_PMAX), PMIN(_PMIN) {}

        /** Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const base::Network<T> &network);

        /** Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const base::Network<T> &network);

    };

}

#endif //DNNSIM_BITFUSION_H
