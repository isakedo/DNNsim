#ifndef DNNSIM_PARALLEL_H
#define DNNSIM_PARALLEL_H

#include "Architecture.h"

#define ZERO_COUNT // Count zeroes as 1 cycle

namespace core {

    /**
     * Parallel simulator
     * @tparam T 16 bits fixed point or 32b float-point
     */
    template <typename T>
    class Parallel : public Architecture<T> {

    private:

        /** BitTactical simulation */
        const bool TCT;

        /**
         * Convert the data representation to the one need it.
         * @param data          Array of values
         * @param data_prec     Activation layer precision
         */
        void dataConversion(base::Array<T> &data, uint8_t data_prec) {}

    public:

        /** Constructor
         * @param _TCT                  Enable BitTactical simulation
         */
        explicit Parallel(bool _TCT) : TCT(_TCT) {}

        /** Compute number of one bit multiplications given a weights and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @param act_prec      Activation layer precision
         * @param wgt_prec      Weight layer precision
         * @param network_bits  Maximum number of bits in the network
         * @return              Number of one bit multiplications
         */
        uint16_t computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits);

        /**
         * Return stats filename for the architecture in the potentials function
         * @return Filename
         */
        std::string filename_pot();

        /**
         * Return stats header for the architecture in the potentials function
         * @return Header
         */
        std::string header_pot(const std::string &name);

    };

}

#endif //DNNSIM_PARALLEL_H
