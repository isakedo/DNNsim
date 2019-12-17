#ifndef DNNSIM_BITPRAGMATIC_H
#define DNNSIM_BITPRAGMATIC_H

#include "Architecture.h"

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

        /** Activate booth-like encoding */
        const bool BOOTH_ENCODING;

        /** Diffy simulation */
        const bool DIFFY;

        /** BitTactical simulation */
        const bool TCT;

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
         * @param _BOOTH_ENCODING       Activate booth-like encoding
         * @param _DIFFY                Enable Diffy
         * @param _TCT                  Enable BitTactical simulation
         */
        BitPragmatic(uint32_t _BITS_FIRST_STAGE, uint32_t _COLUMN_REGISTERS, bool _BOOTH_ENCODING, bool _DIFFY,
                bool _TCT) : BITS_FIRST_STAGE(_BITS_FIRST_STAGE), COLUMN_REGISTERS(_COLUMN_REGISTERS),
                BOOTH_ENCODING(_BOOTH_ENCODING), DIFFY(_DIFFY), TCT(_TCT) {}

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

#endif //DNNSIM_BITPRAGMATIC_H
