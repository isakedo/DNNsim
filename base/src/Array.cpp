
#include <base/Array.h>

namespace base {

    /* SETTERS */

    template <typename T>
    void Array<T>::set_values(const std::string &path) {
        base::NpyArray data_npy;
        base::npy_load(path, data_npy, this->shape);
        std::vector<T> flat_array = data_npy.as_vec<T>();
        if (this->getDimensions() == 1) this->data1D = flat_array;
        else if(this->getDimensions() == 2){
            this->data2D = Array2D(this->shape[0],Array1D(this->shape[1]));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++)
                    this->data2D[i][j] = flat_array[this->shape[1]*i + j];
            }

        } else if (this->getDimensions() == 3) {
            unsigned long coef1 = shape[1]*shape[2];
            this->data3D = Array3D(this->shape[0],Array2D(this->shape[1],Array1D(this->shape[2])));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++)
                        this->data3D[i][j][k] = flat_array[coef1*i + shape[2]*j + k];
                }
            }
        } else if (this->getDimensions() == 4) {
            unsigned long coef1 = shape[1]*shape[2]*shape[3];
            unsigned long coef2 = shape[2]*shape[3];
            this->data4D = Array4D(this->shape[0],Array3D(this->shape[1],Array2D(this->shape[2],
                    Array1D(this->shape[3]))));
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
            this->data2D = Array2D(this->shape[0],Array1D(this->shape[1]));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++)
                    this->data2D[i][j] = _data[this->shape[1]*i + j];
            }
        } else if (this->getDimensions() == 3) {
            unsigned long coef1 = shape[1]*shape[2];
            this->data3D = Array3D(this->shape[0],Array2D(this->shape[1],Array1D(this->shape[2])));
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++)
                        this->data3D[i][j][k] = _data[coef1*i + shape[2]*j + k];
                }
            }
        } else if (this->getDimensions() == 4) {
            auto coef1 = shape[1]*shape[2]*shape[3];
            auto coef2 = shape[2]*shape[3];
            this->data4D = Array4D(this->shape[0],Array3D(this->shape[1],Array2D(this->shape[2],
                    Array1D(this->shape[3]))));
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

    /* GETTERS */

    template <typename T>
    T Array<T>::get (int i, int j, int k, int l) const {
        #ifdef DEBUG
        if(getDimensions() != 4)
            throw std::runtime_error("4D Array dimensions error");
        #endif
        return this->data4D[i][j][k][l];
    }

    template <typename T>
    T Array<T>::get (int i, int j, int k) const {
        #ifdef DEBUG
        if(getDimensions() != 3)
            throw std::runtime_error("3D Array dimensions error");
        #endif
        return this->data3D[i][j][k];
    }

    template <typename T>
    T Array<T>::get (int i, int j) const {
        #ifdef DEBUG
        if(getDimensions() != 2)
            throw std::runtime_error("2D Array dimensions error");
        #endif
        return this->data2D[i][j];
    }

    template <typename T>
    T max_1D(const std::vector<T> &vector) {
        return *std::max_element(vector.begin(), vector.end());
    }

    template <typename T>
    T max_2D(const std::vector<std::vector<T>> &vector) {
        std::vector<T> maximums = std::vector<T>(vector.size(),0);
        for(int i = 0; i < vector.size(); i++) {
            maximums[i] = max_1D(vector[i]);
        }
        return max_1D(maximums);
    }

    template <typename T>
    T max_3D(const std::vector<std::vector<std::vector<T>>> &vector) {
        std::vector<T> maximums = std::vector<T>(vector.size(),0);
        for(int i = 0; i < vector.size(); i++) {
            maximums[i] = max_2D(vector[i]);
        }
        return max_1D(maximums);
    }

    template <typename T>
    T max_4D(const std::vector<std::vector<std::vector<std::vector<T>>>> &vector) {
        std::vector<T> maximums = std::vector<T>(vector.size(),0);
        for(int i = 0; i < vector.size(); i++) {
            maximums[i] = max_3D(vector[i]);
        }
        return max_1D(maximums);
    }

    template <typename T>
    T min_1D(const std::vector<T> &vector) {
        return *std::min_element(vector.begin(), vector.end());
    }

    template <typename T>
    T min_2D(const std::vector<std::vector<T>> &vector) {
        std::vector<T> minimums = std::vector<T>(vector.size(),0);
        for(int i = 0; i < vector.size(); i++) {
            minimums[i] = min_1D(vector[i]);
        }
        return min_1D(minimums);
    }

    template <typename T>
    T min_3D(const std::vector<std::vector<std::vector<T>>> &vector) {
        std::vector<T> minimums = std::vector<T>(vector.size(),0);
        for(int i = 0; i < vector.size(); i++) {
            minimums[i] = min_2D(vector[i]);
        }
        return min_1D(minimums);
    }

    template <typename T>
    T min_4D(const std::vector<std::vector<std::vector<std::vector<T>>>> &vector) {
        std::vector<T> minimums = std::vector<T>(vector.size(),0);
        for(int i = 0; i < vector.size(); i++) {
            minimums[i] = min_3D(vector[i]);
        }
        return min_1D(minimums);
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
        } else if(this->getDimensions() == 3) {
            auto i = index / (this->shape[1]*this->shape[2]);
            auto rem = index % (this->shape[1]*this->shape[2]);
            auto j = rem / this->shape[2];
            auto k = rem % this->shape[2];
            return this->data3D[i][j][k];
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

    template <typename T>
    const std::vector<size_t> &Array<T>::getShape() const {
        return shape;
    }

    template <typename T>
    unsigned long long Array<T>::getMax_index() const {
        if (this->getDimensions() == 1) return this->shape[0];
        else if (this->getDimensions() == 2) return this->shape[0]*this->shape[1];
        else if (this->getDimensions() == 3) return this->shape[0]*this->shape[1]*this->shape[2];
        else if (this->getDimensions() == 4) return this->shape[0]*this->shape[1]*this->shape[2]*this->shape[3];
        else throw std::runtime_error("Array dimensions error");
    }

    /* DATA TRANSFORMATION */

    /* Return value in two complement */
    static inline
    uint16_t profiled_value(float num, int mag, int frac) {
        double scale = pow(2.,(double)frac);
        double intmax = (1u << (mag + frac)) - 1;
        double intmin = -1 * intmax;
        double ds = num * scale;
        if (ds > intmax) ds = intmax;
        if (ds < intmin) ds = intmin;
        auto two_comp = (int)round(ds);
        return (uint16_t)two_comp;
    }

    template <typename T>
    Array<uint16_t> Array<T>::profiled_fixed_point(int mag, int frac) const {
        std::vector<uint16_t> fixed_point_vector;
        if (this->getDimensions() == 1) {
            for(int i = 0; i < this->shape[0]; i++) {
                auto float_value = this->data1D[i];
                fixed_point_vector.push_back(profiled_value(float_value,mag,frac));
            }
        } else if(this->getDimensions() == 2){
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    auto float_value = this->data2D[i][j];
                    fixed_point_vector.push_back(profiled_value(float_value,mag,frac));
                }
            }
        } else if (this->getDimensions() == 3) {
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        auto float_value = this->data3D[i][j][k];
                        fixed_point_vector.push_back(profiled_value(float_value,mag,frac));
                    }
                }
            }
        } else if (this->getDimensions() == 4) {
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++) {
                            auto float_value = this->data4D[i][j][k][l];
                            fixed_point_vector.push_back(profiled_value(float_value,mag,frac));
                        }
                    }
                }
            }
        } else throw std::runtime_error("Array dimensions error");

        Array<uint16_t> fixed_point_array;
        fixed_point_array.set_values(fixed_point_vector,this->shape);
        return fixed_point_array;
    }

    /*static inline
    uint16_t tensorflow_value(float num, double scale, float min_value, int max_fixed, int min_fixed) {
        auto sign_mag = (int)(round(num * scale) - round(min_value * scale) + min_fixed);
        sign_mag = std::max(sign_mag, min_fixed);
        sign_mag = std::min(sign_mag, max_fixed);
        return (uint16_t)sign_mag;
    }*/

    static inline
    uint16_t tensorflow_value(float num, double scale, int max_fixed, int min_fixed) {
    auto sign_mag = (int)round(num * scale);
    sign_mag = std::max(sign_mag, min_fixed);
    sign_mag = std::min(sign_mag, max_fixed);
    return (uint16_t)sign_mag;
}

    template <typename T>
    Array<uint16_t> Array<T>::tensorflow_fixed_point() const {
        //const int NUM_BITS = 8;
        int max_fixed = 127;
        int min_fixed = -127;
        //const int num_discrete_values = 1u << NUM_BITS;
        //const auto range_adjust = num_discrete_values / (num_discrete_values - 1.0);

        std::vector<uint16_t> fixed_point_vector;
        if (this->getDimensions() == 1) {

            auto min_value = min_1D(this->data1D);
            auto max_value = max_1D(this->data1D);
            auto m = std::max(abs(max_value), abs(min_value));
            float scale;
            if (min_value == 0) {
                min_fixed = 0;
                max_fixed = 255;
                scale = (max_fixed - min_fixed) / m;
            } else {
                scale = (max_fixed - min_fixed) / (2 * m);
            }
            //auto range = (max_value - min_value) * range_adjust;
            //auto scale = num_discrete_values / range;

            for(int i = 0; i < this->shape[0]; i++) {
                auto float_value = this->data1D[i];
                //fixed_point_vector.push_back(tensorflow_value(float_value,scale,min_value,max_fixed,min_fixed));
                fixed_point_vector.push_back(tensorflow_value(float_value,scale,max_fixed,min_fixed));
            }
        } else if(this->getDimensions() == 2){

            auto min_value = min_2D(this->data2D);
            auto max_value = max_2D(this->data2D);
            auto m = std::max(abs(max_value), abs(min_value));
            float scale;
            if (min_value == 0) {
                min_fixed = 0;
                max_fixed = 255;
                scale = (max_fixed - min_fixed) / m;
            } else {
                scale = (max_fixed - min_fixed) / (2 * m);
            }
            //auto range = (max_value - min_value) * range_adjust;
            //auto scale = num_discrete_values / range;

            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    auto float_value = this->data2D[i][j];
                    //fixed_point_vector.push_back(tensorflow_value(float_value,scale,min_value,max_fixed,min_fixed));
                    fixed_point_vector.push_back(tensorflow_value(float_value,scale,max_fixed,min_fixed));
                }
            }
        } else if (this->getDimensions() == 3) {

            auto min_value = min_3D(this->data3D);
            auto max_value = max_3D(this->data3D);
            auto m = std::max(abs(max_value), abs(min_value));
            float scale;
            if (min_value == 0) {
                min_fixed = 0;
                max_fixed = 255;
                scale = (max_fixed - min_fixed) / m;
            } else {
                scale = (max_fixed - min_fixed) / (2 * m);
            }
            //auto range = (max_value - min_value) * range_adjust;
            //auto scale = num_discrete_values / range;

            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        auto float_value = this->data3D[i][j][k];
                        //fixed_point_vector.push_back(tensorflow_value(float_value,scale,min_value,max_fixed,min_fixed));
                        fixed_point_vector.push_back(tensorflow_value(float_value,scale,max_fixed,min_fixed));
                    }
                }
            }
        } else if (this->getDimensions() == 4) {

            auto min_value = min_4D(this->data4D);
            auto max_value = max_4D(this->data4D);
            auto m = std::max(abs(max_value), abs(min_value));
            float scale;
            if (min_value == 0) {
                min_fixed = 0;
                max_fixed = 255;
                scale = (max_fixed - min_fixed) / m;
            } else {
                scale = (max_fixed - min_fixed) / (2 * m);
            }
            //auto range = (max_value - min_value) * range_adjust;
            //auto scale = num_discrete_values / range;

            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++) {
                            auto float_value = this->data4D[i][j][k][l];
                            //fixed_point_vector.push_back(tensorflow_value(float_value,scale,min_value,max_fixed,
                            //        min_fixed));
                            fixed_point_vector.push_back(tensorflow_value(float_value,scale,max_fixed,min_fixed));
                        }
                    }
                }
            }
        } else throw std::runtime_error("Array dimensions error");

        Array<uint16_t> fixed_point_array;
        fixed_point_array.set_values(fixed_point_vector,this->shape);
        return fixed_point_array;
    }

    static inline
    uint16_t intel_inq_activation(float num) {
        int two_comp = num * 4096;
        return (uint16_t)two_comp;
    }

    static inline
    uint16_t intel_inq_weight(float num, float max_weight) {
        int two_comp = num * 128 / max_weight;
        return (uint16_t)two_comp;
    }

    template <typename T>
    Array<uint16_t> Array<T>::intel_inq_fixed_point(bool activations) const {

        std::vector<uint16_t> fixed_point_vector;
        if (this->getDimensions() == 1) {

            auto min_value = min_1D(this->data1D);
            auto max_value = max_1D(this->data1D);
            auto max_abs = std::max(abs(max_value),abs(min_value));

            for(int i = 0; i < this->shape[0]; i++) {
                auto float_value = this->data1D[i];
                if (activations) fixed_point_vector.push_back(intel_inq_activation(float_value));
                else fixed_point_vector.push_back(intel_inq_weight(float_value,max_abs));
            }
        } else if(this->getDimensions() == 2){

            auto min_value = min_2D(this->data2D);
            auto max_value = max_2D(this->data2D);
            auto max_abs = std::max(abs(max_value),abs(min_value));

            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    auto float_value = this->data2D[i][j];
                    if (activations) fixed_point_vector.push_back(intel_inq_activation(float_value));
                    else fixed_point_vector.push_back(intel_inq_weight(float_value,max_abs));
                }
            }
        } else if (this->getDimensions() == 3) {

            auto min_value = min_3D(this->data3D);
            auto max_value = max_3D(this->data3D);
            auto max_abs = std::max(abs(max_value),abs(min_value));

            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        auto float_value = this->data3D[i][j][k];
                        if (activations) fixed_point_vector.push_back(intel_inq_activation(float_value));
                        else fixed_point_vector.push_back(intel_inq_weight(float_value,max_abs));
                    }
                }
            }
        } else if (this->getDimensions() == 4) {

            auto min_value = min_4D(this->data4D);
            auto max_value = max_4D(this->data4D);
            auto max_abs = std::max(abs(max_value),abs(min_value));

            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++) {
                            auto float_value = this->data4D[i][j][k][l];
                            if (activations) fixed_point_vector.push_back(intel_inq_activation(float_value));
                            else fixed_point_vector.push_back(intel_inq_weight(float_value,max_abs));
                        }
                    }
                }
            }
        } else throw std::runtime_error("Array dimensions error");

        Array<uint16_t> fixed_point_array;
        fixed_point_array.set_values(fixed_point_vector,this->shape);
        return fixed_point_array;
    }

    template <typename T>
    void Array<T>::sign_magnitude_representation(int prec) {
        double intmax = (1u << (prec - 1)) - 1;
        auto mask = (uint16_t)(intmax + 1);
        if (this->getDimensions() == 1) {
            for(int i = 0; i < this->shape[0]; i++) {
                auto two_comp = (short)this->data1D[i];
                auto abs_value = (uint16_t)abs(two_comp);
                auto sign_mag = abs_value | (two_comp & mask);
                this->data1D[i] = sign_mag;
            }
        } else if(this->getDimensions() == 2){
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    auto two_comp = (short)this->data2D[i][j];
                    auto abs_value = (uint16_t)abs(two_comp);
                    auto sign_mag = abs_value | (two_comp & mask);
                    this->data2D[i][j] = sign_mag;
                }
            }
        } else if (this->getDimensions() == 3) {
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        auto two_comp = (short)this->data3D[i][j][k];
                        auto abs_value = (uint16_t)abs(two_comp);
                        auto sign_mag = abs_value | (two_comp & mask);
                        this->data3D[i][j][k] = sign_mag;
                    }
                }
            }
        } else if (this->getDimensions() == 4) {
            for(int i = 0; i < this->shape[0]; i++) {
                for(int j = 0; j < this->shape[1]; j++) {
                    for(int k = 0; k < this->shape[2]; k++) {
                        for(int l = 0; l < this->shape[3]; l++) {
                            auto two_comp = (short)this->data4D[i][j][k][l];
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
    void Array<T>::powers_of_two_representation(int prec) {
    double intmax = (1 << (prec - 1)) - 1;
    auto mask = (uint16_t)(intmax + 1);
    if (this->getDimensions() == 1) {
        for(int i = 0; i < this->shape[0]; i++) {
            auto two_comp = (short)this->data1D[i];
            auto abs_value = (uint16_t)abs(two_comp);
            auto powers_of_two = abs_value & ~mask;
            this->data1D[i] = powers_of_two;
        }
    } else if(this->getDimensions() == 2){
        for(int i = 0; i < this->shape[0]; i++) {
            for(int j = 0; j < this->shape[1]; j++) {
                auto two_comp = (short)this->data2D[i][j];
                auto abs_value = (uint16_t)abs(two_comp);
                auto powers_of_two = abs_value & ~mask;
                this->data2D[i][j] = powers_of_two;
            }
        }
    } else if (this->getDimensions() == 3) {
        for(int i = 0; i < this->shape[0]; i++) {
            for(int j = 0; j < this->shape[1]; j++) {
                for(int k = 0; k < this->shape[2]; k++) {
                    auto two_comp = (short)this->data3D[i][j][k];
                    auto abs_value = (uint16_t)abs(two_comp);
                    auto powers_of_two = abs_value & ~mask;
                    this->data3D[i][j][k] = powers_of_two;
                }
            }
        }
    } else if (this->getDimensions() == 4) {
        for(int i = 0; i < this->shape[0]; i++) {
            for(int j = 0; j < this->shape[1]; j++) {
                for(int k = 0; k < this->shape[2]; k++) {
                    for(int l = 0; l < this->shape[3]; l++) {
                        auto two_comp = (short)this->data4D[i][j][k][l];
                        auto abs_value = (uint16_t)abs(two_comp);
                        auto powers_of_two = abs_value & ~mask;
                        this->data4D[i][j][k][l] = powers_of_two;
                    }
                }
            }
        }
    } else throw std::runtime_error("Array dimensions error");
}

    /* PADDING */

    template <typename T>
    void Array<T>::zero_pad(int padding) {
        auto batch_size = this->shape[0];
        auto act_channels = this->shape[1];
        auto Nx = this->shape[2];
        auto Ny = this->shape[3];

        auto tmp_data4D = Array4D(batch_size, Array3D(act_channels, Array2D(Nx + 2*padding,Array1D(Ny + 2*padding,0))));

        for(int n = 0; n < batch_size; n++) {
            for (int k = 0; k < act_channels; k++) {
                for (int i = 0; i < Nx; i++) {
                    for(int j = 0; j < Ny; j++) {
                        tmp_data4D[n][k][padding + i][padding + j] = this->data4D[n][k][i][j];
                    }
                }
            }
        }

        this->data4D.clear();
        this->data4D = tmp_data4D;
        this->shape.clear();
        this->shape.push_back(batch_size);
        this->shape.push_back(act_channels);
        this->shape.push_back(Nx + 2*padding);
        this->shape.push_back(Ny + 2*padding);
    }

    template <typename T>
    void Array<T>::grid_zero_pad(uint64_t X, uint64_t Y) {
        auto batch_size = this->shape[0];
        auto act_channels = this->shape[1];
        auto Nx = this->shape[2];
        auto Ny = this->shape[3];

        auto tmp_data4D = Array4D(batch_size, Array3D(act_channels, Array2D(X, Array1D(Y,0))));

        for(int n = 0; n < batch_size; n++) {
            for (int k = 0; k < act_channels; k++) {
                for (int i = 0; i < Nx; i++) {
                    for(int j = 0; j < Ny; j++) {
                        tmp_data4D[n][k][i][j] = this->data4D[n][k][i][j];
                    }
                }
            }
        }

        this->data4D.clear();
        this->data4D = tmp_data4D;
        this->shape.clear();
        this->shape.push_back(batch_size);
        this->shape.push_back(act_channels);
        this->shape.push_back((unsigned)X);
        this->shape.push_back((unsigned)Y);
    }

    template <typename T>
    void Array<T>::channel_zero_pad(int K) {
        auto N = this->shape[0];
        auto old_k = this->shape[1];
        auto X = this->shape[2];
        auto Y = this->shape[3];


        auto tmp_data4D = Array4D(N, Array3D(K, Array2D(X, Array1D(Y, 0))));

        for(int n = 0; n < N; n++) {
            for (int k = 0; k < old_k; k++) {
                for (int i = 0; i < X; i++) {
                    for(int j = 0; j < Y; j++) {
                        tmp_data4D[n][k][i][j] = this->data4D[n][k][i][j];
                    }
                }
            }
        }

        this->data4D.clear();
        this->data4D = tmp_data4D;
        this->shape.clear();
        this->shape.push_back(N);
        this->shape.push_back((unsigned)K);
        this->shape.push_back(X);
        this->shape.push_back(Y);
    }

    /* RESHAPE */

    template <typename T>
    void Array<T>::reshape_to_4D() {
        this->data4D.clear();
        this->data4D = Array4D(this->shape[0],Array3D(this->shape[1],Array2D(1,Array1D(1))));
        for(int i = 0; i < this->shape[0]; i++) {
            for(int j = 0; j < this->shape[1]; j++) {
                this->data4D[i][j][0][0] = this->data2D[i][j];
            }
        }
        this->data2D.clear();
        this->shape.clear();
        this->shape.push_back(this->shape[0]);
        this->shape.push_back(this->shape[1]);
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
        this->data4D.clear();
        this->shape[1] = this->shape[1]*this->shape[2]*this->shape[3];
        this->shape.pop_back();
        this->shape.pop_back();
    }

    template <typename T>
    void Array<T>::split_4D(int K, int X, int Y) {
        auto N = this->shape[0];
        auto old_k = this->shape[1];
        auto old_X = this->shape[2];
        auto old_Y = this->shape[3];

        auto tmp_data4D = Array4D(N,Array3D((unsigned)K,Array2D((unsigned)X,Array1D((unsigned)Y,0))));

        for(int n = 0; n < N; n++) {
            for (int k = 0; k < old_k; k++) {
                for (int i = 0; i < old_X; i++) {
                    for(int j = 0; j < old_Y; j++) {
                        auto new_k = k / (X*Y);
                        auto rem = k % (X*Y);
                        auto new_i = rem / Y;
                        auto new_j = rem % Y;
                        tmp_data4D[n][new_k][new_i][new_j] = this->data4D[n][k][i][j];
                    }
                }
            }
        }

        this->data4D.clear();
        this->data4D = tmp_data4D;
        this->shape.clear();
        this->shape.push_back(N);
        this->shape.push_back((unsigned)K);
        this->shape.push_back((unsigned)X);
        this->shape.push_back((unsigned)Y);
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

        auto tmp_data4D = Array4D(batch_size, Array3D(new_act_channels, Array2D(new_Nx, Array1D(new_Ny, 0))));

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

        auto tmp_data4D = Array4D(num_filters, Array3D(new_wgt_channels, Array2D(new_Kx, Array1D(new_Ky, 0))));

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

    INITIALISE_DATA_TYPES(Array);

}
