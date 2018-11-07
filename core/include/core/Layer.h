#ifndef DNNSIM_LAYER_H
#define DNNSIM_LAYER_H

#include <sys/common.h>
#include <cnpy/Array.h>

namespace core {

    template <typename T>
    class Layer {

    private:

        /* Type of the layer */
        std::string type;

        /* Name of the network */
        std::string name;

        /* Set of layers of the network*/
        std::string input;

        /* Number of outputs */
        int Nn;

        /* Filters X size */
        int Kx;

        /* Filters Y size */
        int Ky;

        /* Stride */
        int stride;

        /* Padding */
        int padding;

        /* Activations magnitude and precision */
        std::tuple<int, int> act_precision;

        /* Weights magnitude and precision */
        std::tuple<int, int> wgt_precision;

        /* numpy array containing the weights for the layer */
        cnpy::Array<T> weights;

        /* numpy array containing the bias for the layer */
        cnpy::Array<T> bias;

        /* numpy array containing the activations for the layer */
        cnpy::Array<T> activations;

        /* numpy array containing the output activations for the layer */
        cnpy::Array<T> output_activations;

    public:

        /* Constructor
         * @param _type     Type of the network
         * @param _name     Name of the layer
         * @param _input    Name of the input layer
         * @param _Nn       Number of outputs
         * @param _Kx       Filters X size
         * @param _Ky       Filters Y size
         * @param _stride   Stride
         * @param _padding  Padding
         */
        Layer(const std::string &_type, const std::string &_name, const std::string &_input, int _Nn, int _Kx, int _Ky,
              int _stride, int _padding) : Nn(_Nn), Kx(_Kx), Ky(_Ky), stride(_stride), padding(_padding)
        { type = _type; name = _name; input = _input; }

        /* Constructor
         * @param _type     Type of the network
         * @param _name     Name of the layer
         * @param _input    Name of the input layer
         * @param _Nn       Number of outputs
         * @param _Kx       Filters X size
         * @param _Ky       Filters Y size
         * @param _stride   Stride
         * @param _padding  Padding
         * @param _act_pre  Activations precision
         * @param _wgt_pre  Weights precision
         */
        Layer(const std::string &_type, const std::string &_name, const std::string &_input, int _Nn, int _Kx, int _Ky,
              int _stride, int _padding, const std::tuple<int,int> _act_pre, const std::tuple<int,int> _wgt_pre ) :
                Nn(_Nn), Kx(_Kx), Ky(_Ky), stride(_stride), padding(_padding), act_precision(_act_pre),
                wgt_precision(_wgt_pre) { type = _type; name = _name; input = _input; }

        /* Getters */
        std::string getType() const { return type; }
        const std::string &getName() const { return name; }
        const std::string &getInput() const { return input; }
        int getNn() const { return Nn; }
        int getKx() const { return Kx; }
        int getKy() const { return Ky; }
        int getStride() const { return stride; }
        int getPadding() const { return padding; }
        const std::tuple<int, int> &getAct_precision() const { return act_precision; }
        const std::tuple<int, int> &getWgt_precision() const { return wgt_precision; }
        const cnpy::Array<T> &getWeights() const { return weights; }
        const cnpy::Array<T> &getBias() const { return bias; }
        const cnpy::Array<T> &getActivations() const { return activations; }
        const cnpy::Array<T> &getOutput_activations() const { return output_activations; }

        /* Setters */
        void setAct_precision(const std::tuple<int, int> &act_precision) { Layer::act_precision = act_precision; }
        void setWgt_precision(const std::tuple<int, int> &wgt_precision) { Layer::wgt_precision = wgt_precision; }
        void setWeights(const cnpy::Array<T> &weights) { Layer::weights = weights; }
        void setBias(const cnpy::Array<T> &bias) { Layer::bias = bias; }
        void setActivations(const cnpy::Array<T> &activations) { Layer::activations = activations; }
        void setOutput_activations(const cnpy::Array<T> &output_activations) {
            Layer::output_activations = output_activations; }

    };

}

#endif //DNNSIM_LAYER_H
