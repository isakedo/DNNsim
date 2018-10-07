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
        uint16_t Nn;

        /* Filters X size */
        uint16_t Kx;

        /* Filters Y size */
        uint16_t Ky;

        /* Stride */
        uint16_t stride;

        /* Padding */
        uint16_t padding;

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
        Layer(const std::string &_name, const std::string &_input, uint16_t _Nn, uint16_t _Kx, uint16_t _Ky,
              uint16_t _stride, uint16_t _padding) : Nn(_Nn), Kx(_Kx), Ky(_Ky), stride(_stride), padding(_padding)
              { name = _name; input = _input; }

         /* Getters */
         const std::string &getName() const {
             return name;
         }

        const std::string &getInput() const {
            return input;
        }

        uint16_t getNn() const {
            return Nn;
        }

        uint16_t getKx() const {
            return Kx;
        }

        uint16_t getKy() const {
            return Ky;
        }

        uint16_t getStride() const {
            return stride;
        }

        uint16_t getPadding() const {
            return padding;
        }


        /* Compute the time for this layer */
        virtual void compute() = 0;

    };

}

#endif //DNNSIM_LAYER_H
