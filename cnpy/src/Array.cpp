
#include <cnpy/Array.h>

namespace cnpy {

    template <typename T>
    void Array<T>::set_values(const std::string &path) {
        cnpy::NpyArray data_npy;
        cnpy::npy_load(path, data_npy, this->shape);
        std::vector<T> flat_array = data_npy.as_vec<T>();
        if (this->getDimensions() == 1) this->data1D = flat_array;
        else if(this->getDimensions() == 2){
            for(int i = 0; i < this->shape[0]; i++) {
                std::vector<T> second_dim;
                second_dim.reserve(this->shape[1]* sizeof(T));
                for(int j = 0; j < this->shape[1]; j++)
                    second_dim.push_back(flat_array[this->shape[1]*i + j]);
                this->data2D.push_back(second_dim);
            }

        } else if (this->getDimensions() == 4) {
            unsigned long coef1 = shape[1]*shape[2]*shape[3];
            unsigned long coef2 = shape[2]*shape[3];
            for(int i = 0; i < this->shape[0]; i++) {
                std::vector<std::vector<std::vector<T>>> second_dim;
                for(int j = 0; j < this->shape[1]; j++) {
                    std::vector<std::vector<T>> third_dim;
                    for(int k = 0; k < this->shape[2]; k++) {
                        std::vector<T> fourth_dim;
                        fourth_dim.reserve(this->shape[3]* sizeof(T));
                        for(int l = 0; l < this->shape[3]; l++)
                            fourth_dim.push_back(flat_array[coef1*i + coef2*j + shape[3]*k + l]);
                        third_dim.push_back(fourth_dim);
                    }
                    second_dim.push_back(third_dim);
                }
                this->data4D.push_back(second_dim);
            }
        } else throw std::runtime_error("Array dimensions error");
    }

    template <typename T>
    void Array<T>::set_values(const std::vector<T> &_data, const std::vector<size_t> &_shape) {
        Array::shape = _shape;
        if (this->getDimensions() == 1) this->data1D = _data;
        else if(this->getDimensions() == 2){
            for(int i = 0; i < this->shape[0]; i++) {
                std::vector<T> second_dim;
                second_dim.reserve(this->shape[1]* sizeof(T));
                for(int j = 0; j < this->shape[1]; j++)
                    second_dim.push_back(_data[this->shape[1]*i + j]);
                this->data2D.push_back(second_dim);
            }

        } else if (this->getDimensions() == 4) {
            auto coef1 = shape[1]*shape[2]*shape[3];
            auto coef2 = shape[2]*shape[3];
            for(int i = 0; i < this->shape[0]; i++) {
                std::vector<std::vector<std::vector<T>>> second_dim;
                for(int j = 0; j < this->shape[1]; j++) {
                    std::vector<std::vector<T>> third_dim;
                    for(int k = 0; k < this->shape[2]; k++) {
                        std::vector<T> fourth_dim;
                        fourth_dim.reserve(this->shape[3]* sizeof(T));
                        for(int l = 0; l < this->shape[3]; l++)
                            fourth_dim.push_back(_data[coef1*i + coef2*j + shape[3]*k + l]);
                        third_dim.push_back(fourth_dim);
                    }
                    second_dim.push_back(third_dim);
                }
                this->data4D.push_back(second_dim);
            }
        } else throw std::runtime_error("Array dimensions error");
    }


    template <typename T>
    T Array<T>::get (int i, int j, int k, int l) const {
        #ifdef DEBUG
        if(getDimensions() != 4)
            throw std::runtime_error("Array dimensions error");
        #endif
        return this->data4D[i][j][k][l];
    }

    template <typename T>
    T Array<T>::get (int i, int j) const {
        #ifdef DEBUG
        if(getDimensions() != 2)
            throw std::runtime_error("Array dimensions error");
        #endif
        return this->data2D[i][j];
    }

    template <typename T>
    T Array<T>::get(unsigned long long index) const {
        if(this->getDimensions() == 4) {
            auto i = index / (this->shape[1]*this->shape[2]*this->shape[3]);
            auto rem = index % (this->shape[1]*this->shape[2]*this->shape[3]);
            auto j = rem / (this->shape[2]*this->shape[3]);
            rem %= (this->shape[2]*this->shape[3]);
            auto k = rem / this->shape[3];
            auto l = rem % this->shape[3];
            return this->data4D[i][j][k][l];
        } else if (this->getDimensions() == 2) {
            auto i = index / this->shape[1];
            auto j = index % this->shape[1];
            return this->data2D[i][j];
        } else if (this->getDimensions() == 1) return this->data1D[index];
        else throw std::runtime_error("Array dimensions error");
    }

    template <typename T>
    unsigned long Array<T>::getDimensions() const {
         if(shape.size() == 4 && shape[2] == 1 && shape[3] == 1) return 2;
         else return shape.size();
    }

    /* Getters */
    template <typename T> const std::vector<size_t> &Array<T>::getShape() const { return shape; }
    template <typename T> unsigned long long Array<T>::getMax_index() const {
        if(this->getDimensions() == 2) return this->shape[0]*this->shape[1];
        else if (this->getDimensions() == 4) return this->shape[0]*this->shape[1]*this->shape[2]*this->shape[3];
        else throw std::runtime_error("Array dimensions error");
    }


    INITIALISE_DATA_TYPES(Array);

}
