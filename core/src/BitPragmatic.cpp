
#include <core/BitPragmatic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void BitPragmatic<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.powers_of_two_representation(data_prec);
    }

    /* CYCLES */


    /* POTENTIALS */

    template <typename T>
    uint8_t BitPragmatic<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        uint16_t act_bits = act;

        #ifdef BOOTH_ENCODING
        act_bits = booth_encoding(act_bits);
        #endif

        uint8_t act_effectual_bits = effectualBits(act_bits);

        uint8_t bit_multiplications = act_effectual_bits * (uint8_t)network_bits;
        #ifdef ZERO_COUNT
        if(bit_multiplications == 0) bit_multiplications = 1;
        #endif

        return bit_multiplications;
    }

    template <typename T>
    std::string BitPragmatic<T>::filename_pot() {
        return "BitPragmatic_potentials";
    }

    template <typename T>
    std::string BitPragmatic<T>::header_pot(const std::string &name) {
        std::string header = "Bit-Pragmatic Potentials/Work Reduction for " + name + "\n";
        #ifdef BOOTH_ENCODING
        header += "Booth-like Encoding\n";
        #endif
        #ifdef ZERO_COUNT
        header += "Zero count as one cycle\n";
        #endif
        return header;
    }

    template class BitPragmatic<uint16_t>;

}