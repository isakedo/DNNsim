
#include <core/ShapeShifter.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    void ShapeShifter<T>::initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear) {
        Architecture<T>::initialise_layer(_act_prec, _wgt_prec, _network_bits, _linear);
        act_mask = (uint16_t)(1u << (_act_prec - 1u));
    }

    template <typename T>
    void ShapeShifter<T>::initialise_batch(uint64_t COLUMNS, uint64_t TILES) {
        Architecture<T>::initialise_batch(COLUMNS, TILES);

        auto GROUPS = ceil(COLUMNS / (double)GROUP_SIZE);
        this->column_cycles = std::vector<std::vector<uint64_t>>(TILES, std::vector<uint64_t>(GROUPS, 0));
    }

    template <typename T>
    uint64_t ShapeShifter<T>::getCycles() const {
        return sys::get_max(this->column_cycles);
    }

    template <typename T>
    std::string ShapeShifter<T>::name() {
        return TCT ? "BitTacticalP" : DIFFY ? "ShapeShifterDiffy" : "ShapeShifter";
    }

    template <typename T>
    void ShapeShifter<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        if (!DIFFY) data.sign_magnitude_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string ShapeShifter<T>::filename() {
        return "_GS" + std::to_string(GROUP_SIZE) + "_CR" + std::to_string(COLUMN_REGISTERS) +
               (MINOR_BIT ? "_MB" : "");
    }

    template <typename T>
    std::string ShapeShifter<T>::header() {
        std::string header = "Number of columns per group: " + std::to_string(GROUP_SIZE) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(COLUMN_REGISTERS) + "\n";
        header +=  MINOR_BIT ? "Trim bits from the bottom\n" : "";
        return header;
    }

    template <typename T>
    bool ShapeShifter<T>::diffy() {
        return DIFFY;
    }

    template <typename T>
    bool ShapeShifter<T>::schedule() {
        return TCT;
    }

    template <typename T>
    void ShapeShifter<T>::process_linear(const std::vector<core::TileData<T>> &tiles_data) {

    }

    template <typename T>
    void ShapeShifter<T>::process_convolution(const std::vector<core::TileData<T>> &tiles_data) {

    }

    template <typename T>
    void ShapeShifter<T>::process_tiles(const std::vector<TileData<T>> &tiles_data) {
        if (this->linear) process_linear(tiles_data);
        else process_convolution(tiles_data);
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
    uint16_t ShapeShifter<T>::computeBits(T act, T wgt) {

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

        return act_width * this->network_bits;
    }

    template class ShapeShifter<uint16_t>;

}
