#ifndef DNNSIM_BITPRAGMATIC_H
#define DNNSIM_BITPRAGMATIC_H

#include "Architecture.h"

#define ZERO_COUNT // Count zeroes as 1 cycle
#define BOOTH_ENCODING // Activate booth-like encoding

namespace core {

    /**
     * BitPragmatic simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class BitPragmatic : public Architecture<T> {

    private:

        /** Bits of the first stage in the two stages shifting */
        const uint32_t BITS_FIRST_STAGE;

        /** Number of registers per SIP */
        const uint32_t COLUMN_REGISTERS;

        /** Diffy simulation */
        const bool DIFFY;

        /**
         * Convert the data representation to the one need it.
         * @param data          Array of values
         * @param data_prec     Activation layer precision
         */
        void dataConversion(base::Array<T> &data, uint8_t data_prec);

    public:

        /** Constructor
         * @param _BITS_FIRST_STAGE     Bits of the first stage in the two stages shifting
         * @param _COLUMN_REGISTERS     Number of registers per SIP
         * @param _DIFFY                Enable Diffy
         */
        BitPragmatic(uint32_t _BITS_FIRST_STAGE, uint32_t _COLUMN_REGISTERS, bool _DIFFY) :
                BITS_FIRST_STAGE(_BITS_FIRST_STAGE), COLUMN_REGISTERS(_COLUMN_REGISTERS), DIFFY(_DIFFY) {}

        /** Compute number of one bit multiplications given a weights and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @param act_prec      Activation layer precision
         * @param wgt_prec      Weight layer precision
         * @param network_bits  Maximum number of bits in the network
         * @return              Number of one bit multiplications
         */
        uint8_t computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits);

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

#endif //DNNSIM_BITPRAGMATIC_H
