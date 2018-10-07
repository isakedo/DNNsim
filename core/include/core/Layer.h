#ifndef DNNSIM_LAYER_H
#define DNNSIM_LAYER_H

#include <string>

namespace core {

    class Layer {

    protected:

        /* Name of the network */
        std::string name;

        /* Set of layers of the network*/
        std::string input;

        /* Number of filters */
        int Nn;

        /* Filters X size */
        int Kx;

        /* Filters Y size */
        int Ky;

        /* Stride */
        int stride;

        /* Padding */
        int padding;

        //Activations and weights not yet

    public:

        /* Constructor
         * @param _name     Name of the layer
         * @param _input    Name of the input layer
         * @param _Nn       Number of filters
         * @param _Kx       Filters X size
         * @param _Ky       Filters Y size
         * @param _stride   Stride
         * @param _padding  Padding
         */
        Layer(const std::string &_name, const std::string &_input, int _Nn, int _Kx, int _Ky,
              int _stride, int _padding) : Nn(_Nn), Kx(_Kx), Ky(_Ky), stride(_stride), padding(_padding)
              { name = _name; input = _input; }

         /* Getters */
         const std::string &getName() const {
             return name;
         }

        const std::string &getInput() const {
            return input;
        }

        int getNn() const {
            return Nn;
        }

        int getKx() const {
            return Kx;
        }

        int getKy() const {
            return Ky;
        }

        int getStride() const {
            return stride;
        }

        int getPadding() const {
            return padding;
        }


        /* Compute the time for this layer */
        virtual void compute() = 0;

    };

}

#endif //DNNSIM_LAYER_H
