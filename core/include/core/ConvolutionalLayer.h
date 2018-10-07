#ifndef DNNSIM_CONVOLUTIONALLAYER_H
#define DNNSIM_CONVOLUTIONALLAYER_H

#include <core/Layer.h>

#include <vector>
#include <string>

namespace core {

    class ConvolutionalLayer : public Layer {

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
        ConvolutionalLayer(const std::string &_name, const std::string &_input, int _Nn, int _Kx, int _Ky, int _stride,
            int _padding) : Layer(_name,_input,_Nn,_Kx,_Ky,_stride,_padding) {}

        /* Compute the time for this layer */
        void compute() override;

    };

};

#endif //DNNSIM_CONVOLUTIONALLAYER_H
