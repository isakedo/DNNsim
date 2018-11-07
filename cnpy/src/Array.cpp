
#include <cnpy/Array.h>

namespace cnpy {

    template <typename T>
    void Array<T>::set_values(const std::string &path) {
        cnpy::NpyArray data_npy;
        cnpy::npy_load(path, data_npy, this->shape);
        this->data = data_npy.as_vec<T>();
        this->coef1 = shape[1]*shape[2]*shape[3];
        this->coef2 = shape[2]*shape[3];
    }

    template <typename T>
    void Array<T>::set_values(const std::vector<T> &_data, const std::vector<size_t> &_shape) {
        Array::data = _data;
        Array::shape = _shape;
        this->coef1 = shape[1]*shape[2]*shape[3];
        this->coef2 = shape[2]*shape[3];
    }


    template <typename T>
    T Array<T>::get (int i, int j, int k, int l) const {
        unsigned long long index = this->coef1*i + this->coef2*j + shape[3]*k + l;

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
         if(shape.size() == 4 && shape[2] == 1 && shape[3] == 1) return 2;
         else return shape.size();
    }

    /* Getters */
    template <typename T> const std::vector<size_t> &Array<T>::getShape() const { return shape; }
    template <typename T> unsigned long long int Array<T>::getMax_index() const { return data.size(); }

    INITIALISE_DATA_TYPES(Array);

}
