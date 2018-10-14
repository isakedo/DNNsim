
#include <cnpy/NumpyArray.h>

namespace cnpy {

    void NumpyArray::set_values(const std::string &path) {
        cnpy::npy_load(path, this->data, this->shape);
        this->max_index = 1;
        for(size_t length : shape)
            this->max_index *= (int)length;
    }

    float NumpyArray::get (int i, int j, int k, int l) const {
        unsigned long long index = shape[1]*shape[2]*shape[3]*i + shape[2]*shape[3]*j + shape[3]*k + l;
        if(index >= this->max_index)
            exit(1);
        return this->data.data<float>()[index];
    }

    float NumpyArray::get (int i, int j) const {
        unsigned long long index = shape[1]*i + j;
        if(index >= this->max_index)
            exit(1);
        return this->data.data<float>()[index];
    }

    float NumpyArray::get(unsigned long long index) const {
        if(index >= this->max_index)
            exit(1);
        return this->data.data<float>()[index];
    }

    unsigned long NumpyArray::getDimensions() const {
        return shape.size();
    }

    /* Getters */
    const std::vector<size_t> &NumpyArray::getShape() const { return shape; }
    unsigned long long int NumpyArray::getMax_index() const { return max_index; }

}
