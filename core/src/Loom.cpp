
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
    uint16_t Loom<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        static auto act_mask = (uint16_t)(1u << (act_prec - 1u));
        static auto wgt_mask = (uint16_t)(1u << (wgt_prec - 1u));

        bool neg_act = false;
        if((act & act_mask) != 0) {
            act = act & ~act_mask;
            neg_act = true;
        }

        const auto &min_max_act_bits = minMax(act);
        auto min_act_bit = std::get<0>(min_max_act_bits);
        auto max_act_bit = std::get<1>(min_max_act_bits);
        if (neg_act) max_act_bit++;

        uint8_t act_width;
        if (MINOR_BIT) act_width = min_act_bit > max_act_bit ? 0 : max_act_bit - min_act_bit + 1u;
        else act_width = max_act_bit + 1u;

        if (!DYNAMIC_WEIGHTS) {
            return act_width * wgt_prec;
        } else {
            bool neg_wgt = false;
            if((wgt & wgt_mask) != 0) {
                wgt = wgt & ~wgt_mask;
                neg_wgt = true;
            }

            const auto &min_max_wgt_bits = minMax(wgt);
            auto min_wgt_bit = std::get<0>(min_max_wgt_bits);
            auto max_wgt_bit = std::get<1>(min_max_wgt_bits);
            if (neg_wgt) max_wgt_bit++;

            uint8_t wgt_width;
            if (MINOR_BIT) wgt_width = min_wgt_bit > max_wgt_bit ? 0 : max_wgt_bit - min_wgt_bit + 1u;
            else wgt_width = max_wgt_bit + 1u;

            return act_width * wgt_width;
        }

    }

    template <typename T>
    std::string Loom<T>::filename_pot() {
        std::string arch = "Loom";
        return arch + (MINOR_BIT ? "_minor" : "") + (DYNAMIC_WEIGHTS ? "_dyn_wgt" : "") + "_potentials";
    }

    template <typename T>
    std::string Loom<T>::header_pot(const std::string &name) {
        std::string header = "Loom Potentials/Work Reduction for " + name + "\n";
        if (MINOR_BIT) header += "Trim bits from the bottom\n";
        if (DYNAMIC_WEIGHTS) header += "Dynamic width adaptation for weights\n";
        return header;
    }

    template class Loom<uint16_t>;

}