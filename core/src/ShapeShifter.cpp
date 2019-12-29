
#include <core/ShapeShifter.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint64_t ShapeShifter<T>::getCycles() const {
        return *std::max_element(this->column_cycles.begin(), this->column_cycles.end());
    }

    template <typename T>
    std::string ShapeShifter<T>::name() {
        return TCT ? "BitTacticalP" : DIFFY ? "ShapeShifterDiffy" : "ShapeShifter";
    }

    template <typename T>
    void ShapeShifter<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        data.sign_magnitude_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string ShapeShifter<T>::filename() {
        return "_PG" + std::to_string(PRECISION_GRANULARITY) + "_CR" + std::to_string(COLUMN_REGISTERS) +
               (MINOR_BIT ? "_MB" : "");
    }

    template <typename T>
    std::string ShapeShifter<T>::header() {
        std::string header = "Number of values per group: " + std::to_string(PRECISION_GRANULARITY) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(COLUMN_REGISTERS) + "\n";
        header +=  MINOR_BIT ? "Trim bits from the bottom\n" : "";
        return header;
    }

    template <typename T>
    bool ShapeShifter<T>::schedule() {
        return TCT;
    }

    template <typename T>
    void ShapeShifter<T>::process_tiles(const std::vector<TileData<T>> &tiles_data, int act_prec, int wgt_prec) {

    }

    /* POTENTIALS */

    template <typename T>
    std::string ShapeShifter<T>::filename_pot() {
        return MINOR_BIT ? "_minor" : "";
    }

    template <typename T>
    std::string ShapeShifter<T>::header_pot() {
        return MINOR_BIT ? "Trim bits from the bottom\n" : "";
    }

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

    template class ShapeShifter<uint16_t>;

}
