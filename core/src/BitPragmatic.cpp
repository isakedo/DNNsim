
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
    uint16_t BitPragmatic<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        if (TCT) {
            if(wgt == 0) return 0;
        }

        uint16_t act_bits = act;
        act_bits = BOOTH_ENCODING ? booth_encoding(act_bits) : act_bits;
        return effectualBits(act_bits) * network_bits;
    }

    template <typename T>
    std::string BitPragmatic<T>::filename_pot() {
        std::string arch = TCT ? "BitTacticalE" : "BitPragmatic";
        return arch + (BOOTH_ENCODING ? "_booth" : "") + "_potentials";
    }

    template <typename T>
    std::string BitPragmatic<T>::header_pot(const std::string &name) {
        std::string arch = TCT ? "Bit-Tactical E" : "BitPragmatic";
        std::string header = arch + " Potentials/Work Reduction for " + name + "\n";
        if (BOOTH_ENCODING) header += "Booth-like Encoding\n";
        return header;
    }

    template class BitPragmatic<uint16_t>;

}