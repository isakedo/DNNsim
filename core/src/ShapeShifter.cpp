
#include <core/ShapeShifter.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void ShapeShifter<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.sign_magnitude_representation(data_prec);
    }

    /* CYCLES */

    /* POTENTIALS */

    template <typename T>
    uint8_t ShapeShifter<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {
        return act_prec * network_bits;
    }

    template <typename T>
    std::string ShapeShifter<T>::filename_pot() {
        return "ShapeShifter_potentials";
    }

    template <typename T>
    std::string ShapeShifter<T>::header_pot(const std::string &name) {
        return "ShapeShifter Potentials/Work Reduction for " + name + "\n";
    }

    template class ShapeShifter<uint16_t>;

}
