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

    };

}

#endif //DNNSIM_NETWORK_H
