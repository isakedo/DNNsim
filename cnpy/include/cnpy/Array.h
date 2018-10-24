#ifndef DNNSIM_ARRAY_H
#define DNNSIM_ARRAY_H


#include <cnpy/cnpy.h>

#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

namespace cnpy {

    class Array {

    private:

        /* Vector with the size of the vector for each dimension
         * Example for 4D: filter index, channel index, X-dimension index, Y-dimension index
         */
        std::vector<size_t> shape;

        /* Vector of float containing the data */
        std::vector<float> data;

    public:

        /* Read the numpy array from the npy file, copy the direction, and set the size
         * @param path  Path to the numpy file with extension .npy
         */
        void set_values(const std::string &path);

        /* Load the vector into the data vector, is_numpy set to false
         * @param _data     Dynamic float vector containing the data
         * @param _shape    Shape of the data
         */
        void set_values(const std::vector<float> &_data, const std::vector<size_t> &_shape);

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
        float get(unsigned long i, unsigned long j) const;

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
        void updateActivations(long i, long j, long k, long l, float n);
        void updateShape(int i, int j);

        /* Getters */
        const std::vector<size_t> &getShape() const;
        unsigned long long int getMax_index() const;

    };

}


#endif //DNNSIM_ARRAY_H
