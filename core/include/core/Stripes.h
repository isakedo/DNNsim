#ifndef DNNSIM_STRIPES_H
#define DNNSIM_STRIPES_H

#include "Architecture.h"

namespace core {

    /**
     * Stripes simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class Stripes : public Architecture<T> {

    private:

        /* AUXILIARY FUNCTIONS */

        /**
         * Return name of the class
         * @return Name
         */
        std::string name();

        /**
         * Convert the data representation to the one need it.
         * @param data          Array of values
         * @param data_prec     Activation layer precision
         */
        void dataConversion(base::Array<T> &data, uint8_t data_prec) {}

        /* CYCLES */

        /**
         * Return stats filename for the architecture in the cycles function
         * @return Filename
         */
        std::string filename();

        /**
         * Return stats header for the architecture in the cycles function
         * @return Header
         */
        std::string header();

        /**
         * Return if schedule the weight buffer
         * @return True if weight buffer to schedule, False if not
         */
        bool schedule();

        /* POTENTIALS */

        /**
         * Return stats filename for the architecture in the potentials function
         * @return Filename
         */
        std::string filename_pot();

        /**
         * Return stats header for the architecture in the potentials function
         * @return Header
         */
        std::string header_pot();

        /** Compute number of one bit multiplications given a weights and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @param act_prec      Activation layer precision
         * @param wgt_prec      Weight layer precision
         * @param network_bits  Maximum number of bits in the network
         * @return              Number of one bit multiplications
         */
        uint16_t computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits);

    };

}

#endif //DNNSIM_STRIPES_H
