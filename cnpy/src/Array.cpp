
#include <cnpy/Array.h>

namespace cnpy {

    template <typename T>
    void Array<T>::set_values(const std::string &path) {
        cnpy::NpyArray data_npy;
        cnpy::npy_load(path, data_npy, this->shape);
        std::vector<T> flat_array = data_npy.as_vec<T>();
        if (this->getDimensions() == 1) this->data1D = flat_array;
        else if(this->getDimensions() == 2){
            this->data2D = std::vector<std::vector<T>>(this->shape[0],std::vector<T>(this->shape[1]));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++)
                    this->data2D[i][j] = flat_array[this->shape[1]*i + j];
            }

        } else if (this->getDimensions() == 4) {
            unsigned long coef1 = shape[1]*shape[2]*shape[3];
            unsigned long coef2 = shape[2]*shape[3];
            this->data4D = std::vector<std::vector<std::vector<std::vector<T>>>>(this->shape[0],
                    std::vector<std::vector<std::vector<T>>>(this->shape[1],std::vector<std::vector<T>>(this->shape[2],
                    std::vector<T>(this->shape[3]))));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++)
                            this->data4D[i][j][k][l] = flat_array[coef1*i + coef2*j + shape[3]*k + l];
                    }
                }
            }
        } else throw std::runtime_error("Array dimensions error");
    }

    template <typename T>
    void Array<T>::set_values(const std::vector<T> &_data, const std::vector<size_t> &_shape) {
        Array::shape = _shape;
        if (this->getDimensions() == 1) this->data1D = _data;
        else if(this->getDimensions() == 2){
            this->data2D = std::vector<std::vector<T>>(this->shape[0],std::vector<T>(this->shape[1]));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++)
                    this->data2D[i][j] = _data[this->shape[1]*i + j];
            }
        } else if (this->getDimensions() == 4) {
            auto coef1 = shape[1]*shape[2]*shape[3];
            auto coef2 = shape[2]*shape[3];
            this->data4D = std::vector<std::vector<std::vector<std::vector<T>>>>(this->shape[0],
                    std::vector<std::vector<std::vector<T>>>(this->shape[1],std::vector<std::vector<T>>(this->shape[2],
                    std::vector<T>(this->shape[3]))));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++)
                            this->data4D[i][j][k][l] = _data[coef1*i + coef2*j + shape[3]*k + l];
                    }
                }
            }
        } else throw std::runtime_error("Array dimensions error");
    }

    template <typename T>
    void Array<T>::sign_magnitude_representation(int mag, int prec) {
        double intmax = (1 << (mag + prec - 1)) - 1;
        auto mask = (uint16_t)intmax + 1;
        if (this->getDimensions() == 1) {
            for(int i = 0; i < this->shape[0]; i++) {
                auto two_comp = (int)this->data1D[i];
                auto abs_value = (uint16_t)abs(two_comp);
                auto sign_mag = abs_value | (two_comp & mask);
                this->data1D[i] = sign_mag;
            }
        } else if(this->getDimensions() == 2){
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    auto two_comp = (int)this->data2D[i][j];
                    auto abs_value = (uint16_t)abs(two_comp);
                    auto sign_mag = abs_value | (two_comp & mask);
                    this->data2D[i][j] = sign_mag;
                }
            }
        } else if (this->getDimensions() == 4) {
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++) {
                            auto two_comp = (int)this->data4D[i][j][k][l];
                            auto abs_value = (uint16_t)abs(two_comp);
                            auto sign_mag = abs_value | (two_comp & mask);
                            this->data4D[i][j][k][l] = sign_mag;
                        }
                    }
                }
            }
        } else throw std::runtime_error("Array dimensions error");
    }

    template <typename T>
    void Array<T>::powers_of_two_representation() {
        if (this->getDimensions() == 1) {
            for(int i = 0; i < this->shape[0]; i++) {
                auto two_comp = (short)this->data1D[i];
                auto powers_of_two = (uint16_t)abs(two_comp);
                this->data1D[i] = powers_of_two;
            }
        } else if(this->getDimensions() == 2){
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    auto two_comp = (short)this->data2D[i][j];
                    auto powers_of_two = (uint16_t)abs(two_comp);
                    this->data2D[i][j] = powers_of_two;
                }
            }
        } else if (this->getDimensions() == 4) {
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++) {
                            auto two_comp = (short)this->data4D[i][j][k][l];
                            auto powers_of_two = (uint16_t)abs(two_comp);
                            this->data4D[i][j][k][l] = powers_of_two;
                        }
                    }
                }
            }
        } else throw std::runtime_error("Array dimensions error");
    }

    template <typename T>
    void Array<T>::reshape_to_4D() {
        //if(getDimensions() == 4 || (this->shape[2] == 1 && this->shape[3] == 1)) return;
        this->data4D.clear();
        this->data4D = std::vector<std::vector<std::vector<std::vector<T>>>>(this->shape[0],
                std::vector<std::vector<std::vector<T>>>(this->shape[1],std::vector<std::vector<T>>(1,
                std::vector<T>(1))));
        for(int i = 0; i < this->shape[0]; i++) {
            for(int j = 0; j < this->shape[1]; j++) {
                this->data4D[i][j][0][0] = this->data2D[i][j];
            }
        }
        this->shape.push_back(1);
        this->shape.push_back(1);
        this->force4D = true;
    }

    template <typename T>
    void Array<T>::reshape_to_2D() {
        this->data2D.clear();
        for(int i = 0; i < this->shape[0]; i++) {
            std::vector<T> second_dim;
            for(int j = 0; j < this->shape[1]; j++) {
                for(int k = 0; k < this->shape[2]; k++) {
                    for(int l = 0; l < this->shape[3]; l++) {
                        second_dim.push_back(this->data4D[i][j][k][l]);
                    }
                }
            }
            this->data2D.push_back(second_dim);
        }
        this->shape[1] = this->shape[1]*this->shape[2]*this->shape[3];
        this->shape.pop_back();
        this->shape.pop_back();
    }

    template <typename T>
    void Array<T>::reshape_first_layer_act(uint16_t stride) {
        if(getDimensions() != 4 || this->shape[1] != 3) return;
        auto batch_size = this->shape[0];
        auto act_channels = this->shape[1];
        auto Nx = this->shape[2];
        auto Ny = this->shape[3];

        auto new_act_channels = (uint16_t)act_channels*stride*stride;
        auto new_Nx = (uint16_t)ceil(Nx/(double)stride);
        auto new_Ny = (uint16_t)ceil(Nx/(double)stride);

        auto tmp_data4D = std::vector<std::vector<std::vector<std::vector<T>>>>(batch_size,
                std::vector<std::vector<std::vector<T>>>(new_act_channels,std::vector<std::vector<T>>(new_Nx,
                std::vector<T>(new_Ny,0))));

        for(int n = 0; n < batch_size; n++)
            for(int k = 0; k < act_channels; k++)
                for(int i = 0; i < Nx; i++)
                    for(int j = 0; j < Ny; j++) {
                        auto new_i = i/stride;
                        auto new_j = j/stride;
                        auto new_k = (j%stride)*stride*act_channels + act_channels*(i%stride) + k;
                        tmp_data4D[n][new_k][new_i][new_j] = this->data4D[n][k][i][j];
                    }

        this->data4D.clear();
        this->data4D = tmp_data4D;
        this->shape.clear();
        this->shape.push_back(batch_size);
        this->shape.push_back(new_act_channels);
        this->shape.push_back(new_Nx);
        this->shape.push_back(new_Ny);
    }

    template <typename T>
    void Array<T>::reshape_first_layer_wgt(uint16_t stride) {
        if(getDimensions() != 4 || this->shape[1] != 3) return;
        auto num_filters = this->shape[0];
        auto wgt_channels = this->shape[1];
        auto Kx = this->shape[2];
        auto Ky = this->shape[3];

        auto new_wgt_channels = (uint16_t)(uint16_t)wgt_channels*stride*stride;
        auto new_Kx = (uint16_t)ceil(Kx/(double)stride);
        auto new_Ky = (uint16_t)ceil(Ky/(double)stride);

        auto tmp_data4D = std::vector<std::vector<std::vector<std::vector<T>>>>(num_filters,
                std::vector<std::vector<std::vector<T>>>(new_wgt_channels,std::vector<std::vector<T>>(new_Kx,
                std::vector<T>(new_Ky,0))));

        for(int m = 0; m < num_filters; m++)
            for(int k = 0; k < wgt_channels; k++)
                for(int i = 0; i < Kx; i++)
                    for(int j = 0; j < Ky; j++) {
                        auto new_i = i/stride;
                        auto new_j = j/stride;
                        auto new_k = (j%stride)*stride*wgt_channels + wgt_channels*(i%stride) + k;
                        tmp_data4D[m][new_k][new_i][new_j] = this->data4D[m][k][i][j];
                    }

        this->data4D.clear();
        this->data4D = tmp_data4D;
        this->shape.clear();
        this->shape.push_back(num_filters);
        this->shape.push_back(new_wgt_channels);
        this->shape.push_back(new_Kx);
        this->shape.push_back(new_Ky);
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
        if(this->force4D) return 4;
        else if(shape.size() == 4 && shape[2] == 1 && shape[3] == 1) return 2;
        else return shape.size();
    }

    /* Getters */
    template <typename T> const std::vector<size_t> &Array<T>::getShape() const { return shape; }
    template <typename T> unsigned long long Array<T>::getMax_index() const {
        if (this->getDimensions() == 1) return this->shape[0];
        else if (this->getDimensions() == 2) return this->shape[0]*this->shape[1];
        else if (this->getDimensions() == 4) return this->shape[0]*this->shape[1]*this->shape[2]*this->shape[3];
        else throw std::runtime_error("Array dimensions error");
    }


    INITIALISE_DATA_TYPES(Array);

}
