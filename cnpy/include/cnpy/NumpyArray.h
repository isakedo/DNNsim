#ifndef DNNSIM_NUMPYARRAY_H
#define DNNSIM_NUMPYARRAY_H

#include <cnpy/cnpy.h>

#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

namespace cnpy {

    class NumpyArray {

    private:

        /* Vector with the size of the vector for each dimension */
        std::vector<size_t> shape;

        /* Pointer to the data */
        cnpy::NpyArray data;

        /* Max index allowed */
        unsigned long long max_index;

    public:

        /* Read the numpy array from the npy file, copy the direction, and set the size
         * @param path  Path to the numpy file with extension .npy
         */
        void set_values(const std::string &path);

        /*  Return the value inside the vector given the fourth dimensions
         * @param i     Index for the first dimension
         * @param j     Index for the second dimension
         * @param k     Index for the third dimension
         * @param l     Index for the fourth dimension
         *
         * @return      return the value given by the index
         */
        float get(int i, int j, int k, int l) const;

        /*  Return the value inside the vector given the two dimensions
         * @param i     Index for the first dimension
         * @param j     Index for the second dimension
         *
         * @return      return the value given by the index
         */
        float get(int i, int j) const;

        /*  Return the value inside the vector given one dimension
         * @param index Index for the array
         *
         * @return      return the value given by the index
         */
        float get(unsigned long long index) const;

        /* Return the number of dimensions of the array
         * @return      return the number of dimensions of the array
         */
        unsigned long getDimensions() const;

        /* Getters */
        const std::vector<size_t> &getShape() const;
        unsigned long long int getMax_index() const;

    };

}

#endif //DNNSIM_NUMPYARRAY_H
