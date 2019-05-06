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

        /* Activations precision: magnitude + sign + frac */
        int act_precision;

        /* Activations magnitude */
        int act_magnitude;

        /* Activations fraction */
        int act_fraction;

        /* Weights precision: magnitude + sign + frac */
        int wgt_precision;

        /* Weights magnitude */
        int wgt_magnitude;

        /* Weights fraction */
        int wgt_fraction;

        /* numpy array containing the weights for the layer */
        cnpy::Array<T> weights;

        /* numpy array containing the bias for the layer */
        cnpy::Array<T> bias;

        /* numpy array containing the activations for the layer */
        cnpy::Array<T> activations;

        /* numpy array containing the output activations for the layer */
        cnpy::Array<T> output_activations;

        /* numpy array containing the weight gradients for the layer */
        cnpy::Array<T> weight_gradients;

        /* numpy array containing the bias gradients for the layer */
        cnpy::Array<T> bias_gradients;

        /* numpy array containing the activation gradients for the layer */
        cnpy::Array<T> activation_gradients;

        /* numpy array containing the output activation gradients for the layer */
        cnpy::Array<T> output_activation_gradients;

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
              int _stride, int _padding) : Nn(_Nn), Kx(_Kx), Ky(_Ky), stride(_stride), padding(_padding) {
            type = _type; name = _name; input = _input;
            act_precision = -1; act_magnitude = -1; act_fraction = -1;
            wgt_precision = -1; wgt_magnitude = -1; wgt_fraction = -1;
        }

        /* Getters */
        std::string getType() const { return type; }
        const std::string &getName() const { return name; }
        const std::string &getInput() const { return input; }
        int getNn() const { return Nn; }
        int getKx() const { return Kx; }
        int getKy() const { return Ky; }
        int getStride() const { return stride; }
        int getPadding() const { return padding; }
        int getAct_precision() const { return act_precision; }
        int getAct_magnitude() const { return act_magnitude; }
        int getAct_fraction() const { return act_fraction; }
        int getWgt_precision() const { return wgt_precision; }
        int getWgt_magnitude() const { return wgt_magnitude; }
        int getWgt_fraction() const { return wgt_fraction; }
        const cnpy::Array<T> &getWeights() const { return weights; }
        const cnpy::Array<T> &getBias() const { return bias; }
        const cnpy::Array<T> &getActivations() const { return activations; }
        const cnpy::Array<T> &getOutput_activations() const { return output_activations; }
        const cnpy::Array<T> &getWeight_gradients() const { return weight_gradients; }
        const cnpy::Array<T> &getBias_gradients() const { return bias_gradients; }
        const cnpy::Array<T> &getActivation_gradients() const { return activation_gradients; }
        const cnpy::Array<T> &getOutput_activation_gradients() const { return output_activation_gradients; }

        /* Setters */

        void setWeights(const cnpy::Array<T> &weights) { Layer::weights = weights; }
        void setBias(const cnpy::Array<T> &bias) { Layer::bias = bias; }
        void setActivations(const cnpy::Array<T> &activations) { Layer::activations = activations; }
        void setOutput_activations(const cnpy::Array<T> &output_activations) {
            Layer::output_activations = output_activations; }
        void setWeight_gradients(const cnpy::Array<T> &weight_gradients) { Layer::weight_gradients = weight_gradients; }
        void setBias_gradients(const cnpy::Array<T> &bias_gradients) { Layer::bias_gradients = bias_gradients; }
        void setActivation_gradients(const cnpy::Array<T> &activation_gradients) {
            Layer::activation_gradients = activation_gradients; }
        void setOutput_activation_gradients(const cnpy::Array<T> &output_activation_gradients) {
            Layer::output_activation_gradients = output_activation_gradients; }
        void setAct_precision(int act_precision, int act_magnitude, int act_fraction) {
            Layer::act_precision = act_precision;
            Layer::act_magnitude = act_magnitude;
            Layer::act_fraction = act_fraction;
        }
        void setWgt_precision(int wgt_precision, int wgt_magnitude, int wgt_fraction) {
            Layer::wgt_precision = wgt_precision;
            Layer::wgt_magnitude = wgt_magnitude;
            Layer::wgt_fraction = wgt_fraction;
        }

    };

}

#endif //DNNSIM_LAYER_H
