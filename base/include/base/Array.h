#ifndef DNNSIM_ARRAY_H
#define DNNSIM_ARRAY_H

#include <sys/common.h>
#include <base/cnpy.h>

namespace base {

    /**
     * Numpy style array for the traces
     * @tparam T Data type of the array
     */
    template <typename T>
    class Array {

    private:

        typedef std::vector<std::vector<std::vector<std::vector<T>>>> Array4D;
        typedef std::vector<std::vector<std::vector<T>>> Array3D;
        typedef std::vector<std::vector<T>> Array2D;
        typedef std::vector<T> Array1D;

        /** Set to true to ensure the array is read as 4D */
        bool force4D = false;

        /** Vector with the size of the vector for each dimension
         * Example for 4D: filter index, channel index, X-dimension index, Y-dimension index
         */
        std::vector<size_t> shape;

        /* Vector containing the data */
        Array1D data1D;
        Array2D data2D;
        Array3D data3D;
        Array4D data4D;

    public:

        /** Constructor */
        Array() = default;

        /** Constructor
         * @param _data     Vector containing the data
         * @param _shape    Shape of the data
         */
        Array(const Array2D &_data, const std::vector<size_t> &_shape) {
            this->data2D = _data;
            this->shape = _shape;
        }

        /** Constructor
         * @param _data     Vector containing the data
         * @param _shape    Shape of the data
         */
        Array(const Array4D &_data, const std::vector<size_t> &_shape) {
            this->data4D = _data;
            this->shape = _shape;
        }

        /** Read the numpy array from the npy file, copy the direction, and set the size
         * @param path  Path to the numpy file with extension .npy
         */
        void set_values(const std::string &path);

        /** Load the vector into the data vector
         * @param _data     Dynamic vector containing the data
         * @param _shape    Shape of the data
         */
        void set_values(const Array1D &_data, const std::vector<size_t> &_shape);

        /** Return the value inside the vector given the fourth dimensions
         * @param i     Index for the first dimension
         * @param j     Index for the second dimension
         * @param k     Index for the third dimension
         * @param l     Index for the fourth dimension
         *
         * @return      return the value given by the index
         */
        T get(int i, int j, int k, int l) const;

        /** Return the value inside the vector given the fourth dimensions
         * @param i     Index for the first dimension
         * @param j     Index for the second dimension
         * @param k     Index for the third dimension
         *
         * @return      return the value given by the index
         */
        T get(int i, int j, int k) const;

        /** Return the value inside the vector given the two dimensions
         * @param i     Index for the first dimension
         * @param j     Index for the second dimension
         *
         * @return      return the value given by the index
         */
        T get(int i, int j) const;

        /** Return the value inside the vector given one dimension
         * @param index Index for the array
         *
         * @return      return the value given by the index
         */
        T get(unsigned long long index) const;

        /** Return the number of dimensions of the array
         * @return  Number of dimensions of the array
         */
        unsigned long getDimensions() const;

        /** Get shape of the array
         */
        const std::vector<size_t> &getShape() const;

        /** Get Maximum index
         */
        unsigned long long getMax_index() const;

        /** Return a fixed point array given profiled precisions
         * @param mag   Magnitude (without sign bit)
         * @param frac  Fraction
         * @return Fixed point quantized tensor
         */
        Array<uint16_t> profiled_fixed_point(int mag, int frac) const;

        /**
         * Return a fixed point array for tensorflow 8bits quantization
         * @return Fixed point quantized tensor
         */
        Array<uint16_t> tensorflow_fixed_point() const;

        /** Return a fixed point array for intel INQ quantization
         * @param activations True if activations traces
         * @return Fixed point quantized tensor
         */
        Array<uint16_t> intel_inq_fixed_point(bool activations) const;

        /** Change fixed point representation to sign-magnitude
         * @param prec   Precision: sign + magnitude + fraction
         */
        void sign_magnitude_representation(int prec);

        /** Change fixed point representation to powers of two
         * @param prec   Precision: sign + magnitude + fraction
         */
        void powers_of_two_representation(int prec);

        /** zero pad the activations
         * @param padding   Padding of the layer
         */
        void zero_pad(int padding);

        /** zero pad the activations to fit on the grid size
         * @param X   New X dimension for the activations
         * @param Y   New Y dimension for the activations
         */
        void grid_zero_pad(uint64_t X, uint64_t Y);

        /** zero pad the channel
         * @param K   New K dimension for the channels
         */
        void channel_zero_pad(int K);

        /** Transform a 2D array into 4D to allow accessing it as 4D */
        void reshape_to_4D();

        /** Transform a 4D array into 2D */
        void reshape_to_2D();

        /** Split a 4D array in the form N,old_K,1,1 into N,K,X,Y
         * @param K   New K dimension for the channels
         * @param X   New X dimension for the activations
         * @param Y   New Y dimension for the activations
         */
        void split_4D(int K, int X, int Y);

        /** Reshape the input activations to a better shape for the first layer
         * @param stride    Stride of the layer, must be bigger than 1
         */
        void reshape_first_layer_act(uint16_t stride);

        /** Reshape the weights to a better shape for the first layer
         * @param stride    Stride of the layer, must be bigger than 1
         */
        void reshape_first_layer_wgt(uint16_t stride);

    };

}


#endif //DNNSIM_ARRAY_H
