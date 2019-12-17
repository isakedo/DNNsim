
#include <core/Laconic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void Laconic<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.powers_of_two_representation(data_prec);
    }

    /* CYCLES */

    /* POTENTIALS */

    template <typename T>
    uint16_t Laconic<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        uint16_t act_bits = act;
        uint16_t wgt_bits = wgt;
        act_bits = BOOTH_ENCODING ? booth_encoding(act_bits) : act_bits;
        wgt_bits = BOOTH_ENCODING ? booth_encoding(wgt_bits) : wgt_bits;
        return effectualBits(act_bits) * effectualBits(wgt_bits);
    }

    template <typename T>
    std::string Laconic<T>::filename_pot() {
        return "Laconic_potentials";
    }

    template <typename T>
    std::string Laconic<T>::header_pot(const std::string &name) {
        std::string header = "Loom Potentials/Work Reduction for " + name + "\n";
        #ifdef BOOTH_ENCODING
        header += "Booth-like Encoding\n";
        #endif
        #ifdef ZERO_COUNT
        header += "Zero count as one cycle\n";
        #endif
        return header;
    }

    template class Laconic<uint16_t>;

}