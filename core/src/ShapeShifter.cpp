
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
    uint16_t ShapeShifter<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        static auto act_mask = (uint16_t)(1u << (act_prec - 1u));

        if (TCT) {
            if(wgt == 0) return 0;
        }

        bool neg = false;
        if((act & act_mask) != 0) {
            act = act & ~act_mask;
            neg = true;
        }

        const auto &min_max_act_bits = minMax(act);
        auto min_act_bit = std::get<0>(min_max_act_bits);
        auto max_act_bit = std::get<1>(min_max_act_bits);
        if (neg) max_act_bit++;

        uint8_t act_width;
        if (MINOR_BIT) act_width = min_act_bit > max_act_bit ? 0 : max_act_bit - min_act_bit + 1u;
        else act_width = max_act_bit + 1u;

        return act_width * network_bits;
    }

    template <typename T>
    std::string ShapeShifter<T>::filename_pot() {
        std::string arch = TCT ? "BitTacticalP" : "ShapeShifter";
        return arch + (MINOR_BIT ? "_minor" : "") + "_potentials";
    }

    template <typename T>
    std::string ShapeShifter<T>::header_pot(const std::string &name) {
        std::string arch = TCT ? "Bit-Tactical P" : "ShapeShifter";
        std::string header = arch + " Potentials/Work Reduction for " + name + "\n";
        if (MINOR_BIT) header += "Trim bits from the bottom\n";
        return header;
    }

    template class ShapeShifter<uint16_t>;

}
