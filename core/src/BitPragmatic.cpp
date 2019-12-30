
#include <core/BitPragmatic.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    uint64_t BitPragmatic<T>::getCycles() const {
        return sys::get_max(this->column_cycles);
    }

    template <typename T>
    std::string BitPragmatic<T>::name() {
        return TCT ? "BitTacticalE" : DIFFY ? "BitPragmaticDiffy" : "BitPragmatic";
    }

    template <typename T>
    void BitPragmatic<T>::dataConversion(base::Array<T> &data, uint8_t data_prec) {
        if (!DIFFY) data.powers_of_two_representation(data_prec);
    }

    /* CYCLES */

    template <typename T>
    std::string BitPragmatic<T>::filename() {
        return "_B" + std::to_string(BITS_FIRST_STAGE) + "_CR" + std::to_string(COLUMN_REGISTERS) +
               (BOOTH_ENCODING ? "_booth" : "");
    }

    template <typename T>
    std::string BitPragmatic<T>::header() {
        std::string header = "Number of bits for first stage shifter: " + std::to_string(BITS_FIRST_STAGE) + "\n";
        header += "Number of run-ahead input registers per column: " + std::to_string(COLUMN_REGISTERS) + "\n";
        header +=  BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
        return header;
    }

    template <typename T>
    bool BitPragmatic<T>::schedule() {
        return TCT;
    }

    template <typename T>
    void BitPragmatic<T>::process_tiles(const std::vector<TileData<T>> &tiles_data, int act_prec, int wgt_prec) {

    }

    /* POTENTIALS */

    template <typename T>
    std::string BitPragmatic<T>::filename_pot() {
        return BOOTH_ENCODING ? "_booth" : "";
    }

    template <typename T>
    std::string BitPragmatic<T>::header_pot() {
        return BOOTH_ENCODING ? "Booth-like Encoding\n" : "";
    }

    template <typename T>
    uint16_t BitPragmatic<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        if (TCT) {
            if(wgt == 0) return 0;
        }

        uint16_t act_bits = act;
        act_bits = BOOTH_ENCODING ? booth_encoding(act_bits) : act_bits;
        return effectualBits(act_bits) * network_bits;
    }

    template class BitPragmatic<uint16_t>;

}