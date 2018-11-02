
#include <core/Simulator.h>

namespace core {

    template <typename T>
    cnpy::Array<T> Simulator<T>::adjustPadding(const cnpy::Array<T> &array, int padding) {
        cnpy::Array<T> padded_array;
        return padded_array;
    }

    template class Simulator<uint16_t >;

}
