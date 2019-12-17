
#include <core/Laconic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    std::string Laconic<T>::name() {
        return "Laconic";
    }

    template <typename T>
    void Laconic<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.powers_of_two_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string Laconic<T>::filename() {
        return "";
    }

    template <typename T>
    std::string Laconic<T>::header() {
        return BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
    }

    template <typename T>
    bool Laconic<T>::schedule() {
        return false;
    }

    /* POTENTIALS */

    template <typename T>
    std::string Laconic<T>::filename_pot() {
        return BOOTH_ENCODING ? "_booth" : "";
    }

    template <typename T>
    std::string Laconic<T>::header_pot() {
        return BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
    }

    template <typename T>
    uint16_t Laconic<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        uint16_t act_bits = act;
        uint16_t wgt_bits = wgt;
        act_bits = BOOTH_ENCODING ? booth_encoding(act_bits) : act_bits;
        wgt_bits = BOOTH_ENCODING ? booth_encoding(wgt_bits) : wgt_bits;
        return effectualBits(act_bits) * effectualBits(wgt_bits);
    }


    template class Laconic<uint16_t>;

}