
#include <core/Loom.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void Loom<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.sign_magnitude_representation(data_prec);
    }

    /* CYCLES */

    /* POTENTIALS */

    template <typename T>
    uint8_t Loom<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {
        return act_prec * wgt_prec;
    }

    template <typename T>
    std::string Loom<T>::filename_pot() {
        return "Loom_potentials";
    }

    template <typename T>
    std::string Loom<T>::header_pot(const std::string &name) {
        return "Loom Potentials/Work Reduction for " + name + "\n";
    }

    template class Loom<uint16_t>;

}