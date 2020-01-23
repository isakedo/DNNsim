#ifndef DNNSIM_LAYER_H
#define DNNSIM_LAYER_H

#include <sys/common.h>
#include <base/Array.h>

namespace base {

    /**
     * Container for the layers of the network
     * @tparam T Data type of the layer
     */
    template <typename T>
    class Layer {

    private:

        /** Type of the layer */
        std::string type;

        /** Name of the network */
        std::string name;

        /** Stride */
        int stride;

        /** Padding */
        int padding;

        /** Activations precision: magnitude + sign + frac */
        int act_precision;

        /** Activations magnitude */
        int act_magnitude;

        /** Activations fraction */
        int act_fraction;

        /** Weights precision: magnitude + sign + frac */
        int wgt_precision;

        /** Weights magnitude */
        int wgt_magnitude;

        /** Weights fraction */
        int wgt_fraction;

        /** numpy array containing the weights for the layer */
        Array<T> weights;

        /** numpy array containing the activations for the layer */
        Array<T> activations;

    public:

        /** Constructor
         * @param _name     Name of the layer
         * @param _type     Type of the network
         * @param _stride   Stride
         * @param _padding  Padding
         */
        Layer(const std::string &_name, const std::string &_type, int _stride, int _padding) : stride(_stride),
                padding(_padding) {
            name = _name; type = _type;
            act_precision = -1; act_magnitude = -1; act_fraction = -1;
            wgt_precision = -1; wgt_magnitude = -1; wgt_fraction = -1;
        }

        /** Constructor
         * @param _name             Name of the layer
         * @param _type             Type of the network
         * @param _stride           Stride
         * @param _padding          Padding
         * @param _act_precision    Activations precision
         * @param _act_magnitude    Activations magnitude
         * @param _act_fraction     Activations fraction
         * @param _wgt_precision    Weights precision
         * @param _wgt_magnitude    Weights magnitude
         * @param _wgt_fraction     Weights fraction
         */
        Layer(const std::string &_name, const std::string &_type, int _stride, int _padding, int _act_precision,
                int _act_magnitude, int _act_fraction, int _wgt_precision, int _wgt_magnitude, int _wgt_fraction) :
                stride(_stride), padding(_padding), act_precision(_act_precision), act_magnitude(_act_magnitude),
                act_fraction(_act_fraction), wgt_precision(_wgt_precision), wgt_magnitude(_wgt_magnitude),
                wgt_fraction(_wgt_fraction) {
            type = _type; name = _name;
        }

        /**
         * Get the type of the layer (Conv, Inner Productm etc.)
         * @return Type of the layer
         */
        std::string getType() const { return type; }

        /**
         * Get the name of the layer
         * @return Name of the layer
         */
        const std::string &getName() const { return name; }

        /**
         * Get the stride of the layer
         * @return Stride of the layer
         */
        int getStride() const { return stride; }

        /**
         * Get the padding of the layer
         * @return Padding of the layer
         */
        int getPadding() const { return padding; }

        /**
         * Get the activations precision
         * @return Activations precision
         */
        int getActPrecision() const { return act_precision; }

        /**
         * Get the activations magnitude
         * @return Activations magnitude
         */
        int getActMagnitude() const { return act_magnitude; }

        /**
         * Get the activations fraction
         * @return Activations fraction
         */
        int getActFraction() const { return act_fraction; }

        /**
         * Get the weights precision
         * @return Weights precision
         */
        int getWgtPrecision() const { return wgt_precision; }

        /**
         * Get the weights magnitude
         * @return Weights magnitude
         */
        int getWgtMagnitude() const { return wgt_magnitude; }

        /**
         * Get the weights fraction
         * @return Weights fraction
         */
        int getWgtFraction() const { return wgt_fraction; }

        /**
         * Get weights data
         * @return Weights data
         */
        const Array<T> &getWeights() const { return weights; }

        /**
         * Get activations data
         * @return Activations data
         */
        const Array<T> &getActivations() const { return activations; }

        /**
         * Set the name of the layer
         * @param _name Name of the layer
         */
        void setName(const std::string &_name) { Layer::name = _name; }

        /**
         * Set the weights data
         * @param _weights Weights data
         */
        void setWeights(const Array<T> &_weights) { Layer::weights = _weights; }

        /**
         * Set the activations data
         * @param _activations Activations data
         */
        void setActivations(const Array<T> &_activations) { Layer::activations = _activations; }

        /**
         * Set the activations precision
         * @param _act_precision Activations precision
         * @param _act_magnitude Activations magnitude
         * @param _act_fraction Activations fraction
         */
        void setAct_precision(int _act_precision, int _act_magnitude, int _act_fraction) {
            Layer::act_precision = _act_precision;
            Layer::act_magnitude = _act_magnitude;
            Layer::act_fraction = _act_fraction;
        }

        /**
         * Set the weights precision
         * @param _wgt_precision Weights precision
         * @param _wgt_magnitude Weights magnitude
         * @param _wgt_fraction Weights fraction
         */
        void setWgt_precision(int _wgt_precision, int _wgt_magnitude, int _wgt_fraction) {
            Layer::wgt_precision = _wgt_precision;
            Layer::wgt_magnitude = _wgt_magnitude;
            Layer::wgt_fraction = _wgt_fraction;
        }

    };

}

#endif //DNNSIM_LAYER_H
