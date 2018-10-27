#ifndef DNNSIM_ARRAY_H
#define DNNSIM_ARRAY_H

#include <sys/common.h>
#include <cnpy/cnpy.h>

namespace cnpy {

    template <typename T>
    class Array {

    private:

        /* Vector with the size of the vector for each dimension
         * Example for 4D: filter index, channel index, X-dimension index, Y-dimension index
         */
        std::vector<size_t> shape;

        /* Vector containing the data */
        std::vector<T> data;

    public:

        /* Read the numpy array from the npy file, copy the direction, and set the size
         * @param path  Path to the numpy file with extension .npy
         */
        void set_values(const std::string &path);

        /* Load the vector into the data vector, is_numpy set to false
         * @param _data     Dynamic float vector containing the data
         * @param _shape    Shape of the data
         */
        void set_values(const std::vector<T> &_data, const std::vector<size_t> &_shape);

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
        T get(unsigned long i, unsigned long j) const;

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
        unsigned long long int getMax_index() const;

    };

}


#endif //DNNSIM_ARRAY_H
