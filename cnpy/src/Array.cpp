
#include <cnpy/Array.h>

namespace cnpy {

    template <typename T>
    void Array<T>::set_values(const std::string &path) {
        cnpy::NpyArray data_npy;
        cnpy::npy_load(path, data_npy, this->shape);
        unsigned long long max_index = 1;
        for(size_t length : shape)
            max_index *= (int)length;
        for(unsigned long long i = 0; i < max_index; i++)
            this->data.push_back(data_npy.data<float>()[i]);
    }

    template <typename T>
    void Array<T>::set_values(const std::vector<T> &_data, const std::vector<size_t> &_shape) {
        Array::data = _data;
        Array::shape = _shape;
    }


    template <typename T>
    T Array<T>::get (int i, int j, int k, int l) const {
        unsigned long long index = shape[1]*shape[2]*shape[3]*i + shape[2]*shape[3]*j + shape[3]*k + l;

        #ifdef DEBUG
        if(index >= this->data.size()) {
            throw std::runtime_error("Array out of index");
        }
        #endif

        return this->data[index];
    }

    template <typename T>
    T Array<T>::get (unsigned long i, unsigned long j) const {
        unsigned long long index = shape[1]*i + j;

        #ifdef DEBUG
        if(index >= this->data.size()) {
            throw std::runtime_error("Array out of index");
        }
        #endif

        return this->data[index];
    }

    template <typename T>
    T Array<T>::get(unsigned long long index) const {

        #ifdef DEBUG
        if(index >= this->data.size()) {
            throw std::runtime_error("Array out of index");
        }
        #endif

        return this->data[index];
    }

    template <typename T>
    unsigned long Array<T>::getDimensions() const {
        return shape.size();
    }

    /* Getters */
    template <typename T> const std::vector<size_t> &Array<T>::getShape() const { return shape; }
    template <typename T> unsigned long long int Array<T>::getMax_index() const { return data.size(); }

    INITIALISE_DATA_TYPES(Array);

}
