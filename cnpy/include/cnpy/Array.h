#ifndef DNNSIM_ARRAY_H
#define DNNSIM_ARRAY_H

#include <sys/common.h>
#include <cnpy/cnpy.h>

namespace cnpy {

    template <typename T>
    class Array {

    private:

        /* Set to true to ensure the array is read as 4D */
        bool force4D = false;

        /* Vector with the size of the vector for each dimension
         * Example for 4D: filter index, channel index, X-dimension index, Y-dimension index
         */
        std::vector<size_t> shape;

        /* Vector containing the data */
        std::vector<T> data1D;
        std::vector<std::vector<T>> data2D;
        std::vector<std::vector<std::vector<std::vector<T>>>> data4D;

    public:

        /* Read the numpy array from the npy file, copy the direction, and set the size
         * @param path  Path to the numpy file with extension .npy
         */
        void set_values(const std::string &path);

        /* Load the vector into the data vector, is_numpy set to false
         * @param _data     Dynamic vector containing the data
         * @param _shape    Shape of the data
         */
        void set_values(const std::vector<T> &_data, const std::vector<size_t> &_shape);

        /* Change fixed point representation to sign-magnitude
         * @param mag   Magnitude: position before the comma
         * @param prec  Precision: positions after the comma
         */
        void sign_magnitude_representation(int mag, int prec);

        /* Change fixed point representation to powers of two */
        void powers_of_two_representation();

        /* Transform a 2D array into 4D to allow accessing it as 4D */
        void reshape_to_4D();

        /* Transform a 4D array into 2D */
        void reshape_to_2D();

        /*  Return the value inside the vector given the fourth dimensions
         * @param i     Index for the first dimension
         * @param j     Index for the second dimension
         * @param k     Index for the third dimension
         * @param l     Index for the fourth dimension
         *
         * @return      return the value given by the index
         */
        T get(int i, int j, int k, int l) const;

        /*  Return the value inside the vector given the two dimensions
         * @param i     Index for the first dimension
         * @param j     Index for the second dimension
         *
         * @return      return the value given by the index
         */
        T get(int i, int j) const;

        /*  Return the value inside the vector given one dimension
         * @param index Index for the array
         *
         * @return      return the value given by the index
         */
        T get(unsigned long long index) const;

        /* Return the number of dimensions of the array
         * @return      return the number of dimensions of the array
         */
        unsigned long getDimensions() const;

        /* Getters */
        const std::vector<size_t> &getShape() const;
        unsigned long long getMax_index() const;

    };

}


#endif //DNNSIM_ARRAY_H
