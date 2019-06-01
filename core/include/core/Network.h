#ifndef DNNSIM_NETWORK_H
#define DNNSIM_NETWORK_H

#include <sys/common.h>
#include <core/Layer.h>

namespace core {

    template <typename T>
    class Network {

    private:

        /* Name of the network */
        std::string name;

        /* Set of layers of the network*/
        std::vector<Layer<T>> layers;

        /* Max number of bits for the network*/
        int network_bits;

        /* Active forward traces */
        bool forward;

        /* Active backward traces */
        bool backward;

    public:

        /* Default constructor */
        Network() = default;

        /* Constructor
         * @param _name      The name of the network
         * @param _layers    Vector of layers
         */
        Network(const std::string &_name, const std::vector<Layer<T>> &_layers) : network_bits(16) {
            name = _name; layers = _layers;
        }

        /* Getters */
        const std::string &getName() const { return name; }
        const std::vector<Layer<T>> &getLayers() const { return layers; }
        int getNetwork_bits() const { return network_bits; }
        bool getForward() const { return forward; }
        bool getBackward() const { return backward; }

        /* Setters */
        std::vector<Layer<T>> &updateLayers() { return layers; }
        void setNetwork_bits(int network_bits) { Network::network_bits = network_bits; }
        void setForkward(bool forward) { Network::forward = forward; }
        void setBackward(bool backward) { Network::backward = backward; }

        /* Duplicate the decoder layers to store all decode steps
         * @param decoder_states Number of decoder states in the traces
         */
        void duplicate_decoder_layers(int decoder_states) {

            std::vector<Layer<T>> tmp_layers;
            std::vector<Layer<T>> tmp_decoders;

            for(const auto layer : this->layers) {
                if(layer.getType() == "Decoder") tmp_decoders.push_back(layer);
                else tmp_layers.push_back(layer);
            }

            for(int decoder_state = 0; decoder_state < decoder_states; decoder_state++) {
                for(const auto layer : tmp_decoders) {
                    tmp_layers.push_back(layer);
                    tmp_layers.back().setName(tmp_layers.back().getName() + "_" + std::to_string(decoder_state));
                }
            }

            this->layers = tmp_layers;

        }

    };

}

#endif //DNNSIM_NETWORK_H
