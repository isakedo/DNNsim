
#include <core/Parallel.h>

namespace core {

    /* AUXILIARY FUNCTIONS */

    template <typename T>
    std::string Parallel<T>::name() {
        return TCT ? "BitTactical" : "Parallel";
    }

    /* CYCLES */

    template <typename T>
    std::string Parallel<T>::filename() {
        return "";
    }

    template <typename T>
    std::string Parallel<T>::header() {
        return "";
    }

    template <typename T>
    bool Parallel<T>::schedule() {
        return TCT;
    }

    template <typename T>
    void Parallel<T>::process_tiles(const std::vector<TileData<T>> &tiles_data, int act_prec, int wgt_prec) {
        this->cycles++;
    }

    /* POTENTIALS */

    template <typename T>
    std::string Parallel<T>::filename_pot() {
        return "";
    }

    template <typename T>
    std::string Parallel<T>::header_pot() {
        return "";
    }

    template <typename T>
    uint16_t Parallel<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        if (TCT) {
            if(wgt == 0) return 0;
        }

        return network_bits * network_bits;
    }

    INITIALISE_DATA_TYPES(Parallel);

}