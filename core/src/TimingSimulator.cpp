
#include <core/TimingSimulator.h>

namespace core {

    template <typename T>
    cnpy::Array<T> TimingSimulator<T>::adjustPadding(const cnpy::Array<T> &array, int padding) {
        cnpy::Array<T> padded_array;
        return padded_array;
    }

    template class TimingSimulator<uint16_t >;

}
