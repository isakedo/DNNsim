
#include <cnpy/NumpyArray.h>

namespace cnpy {

    void NumpyArray::set_values(const std::string &path) {
        cnpy::npy_load(path, this->data, this->shape);
    }

    float NumpyArray::get (int i, int j, int k, int l) const {
        unsigned long long index = shape[1]*shape[2]*shape[3]*i + shape[2]*shape[3]*j + shape[3]*k + l;
        return this->data.data<float>()[index];
    }

    /* Getters */
    const std::vector<size_t> &NumpyArray::getShape() const { return shape; }

}
