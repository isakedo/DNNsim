#ifndef DNNSIM_NUMPYARRAY_H
#define DNNSIM_NUMPYARRAY_H

#include <cstdlib>
#include <vector>
#include <string>

namespace cnpy {

    class NumpyArray {

    private:

        /* Vector with the size of the vector for each dimension */
        std::vector<size_t> shape;

        /* Pointer to the data */
        float* data;

    public:

        /* Destructor */
        ~NumpyArray() { delete data; }

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
        float get(int i, int j, int k, int l);

    };

}

#endif //DNNSIM_NUMPYARRAY_H
