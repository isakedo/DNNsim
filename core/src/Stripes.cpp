
#include <core/Stripes.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    std::string Stripes<T>::name() {
        return "Stripes";
    }

    /* CYCLES */

    template <typename T>
    std::string Stripes<T>::filename() {
        return "";
    }

    template <typename T>
    std::string Stripes<T>::header() {
        return "";
    }

    template <typename T>
    bool Stripes<T>::schedule() {
        return false;
    }

    /* POTENTIALS */

    template <typename T>
    std::string Stripes<T>::filename_pot() {
        return "";
    }

    template <typename T>
    std::string Stripes<T>::header_pot() {
        return "";
    }

    template <typename T>
    uint16_t Stripes<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {
        return act_prec * network_bits;
    }

    template class Stripes<uint16_t>;

}
