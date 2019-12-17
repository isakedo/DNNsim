
#include <core/Stripes.h>

namespace core {

    /* CYCLES */

    /* POTENTIALS */

    template <typename T>
    uint16_t Stripes<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {
        return act_prec * network_bits;
    }

    template <typename T>
    std::string Stripes<T>::filename_pot() {
        return "Stripes_potentials";
    }

    template <typename T>
    std::string Stripes<T>::header_pot(const std::string &name) {
        return "Stripes Potentials/Work Reduction for " + name + "\n";
    }

    template class Stripes<uint16_t>;

}
