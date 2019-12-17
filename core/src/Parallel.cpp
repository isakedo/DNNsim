
#include <core/Parallel.h>

namespace core {

    /* CYCLES */


    /* POTENTIALS */

    template <typename T>
    uint16_t Parallel<T>::computeBits(T act, T wgt, uint8_t act_prec, uint8_t wgt_prec, uint8_t network_bits) {

        if (TCT) {
            if(wgt == 0) return 0;
        }

        return network_bits * network_bits;
    }

    template <typename T>
    std::string Parallel<T>::filename_pot() {
        std::string arch = TCT ? "BitTactical" : "Parallel";
        return arch + "_potentials";
    }

    template <typename T>
    std::string Parallel<T>::header_pot(const std::string &name) {
        std::string arch = TCT ? "Bit-Tactical" : "Parallel";
        std::string header = arch + " Potentials/Work Reduction for " + name + "\n";
        #ifdef ZERO_COUNT
        if (TCT) header += "Zero count as one cycle\n";
        #endif
        return header;
    }

    INITIALISE_DATA_TYPES(Parallel);

}