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

    public:

        /* Default constructor */
        Network() = default;

        /* Constructor
         * @param _name      The name of the network
         * @param _layers    Vector of layers
         */
        Network(const std::string &_name, const std::vector<Layer<T>> &_layers) {
            name = _name; layers = _layers;
        }

        /* Getters */
        const std::string &getName() const { return name; }
        const std::vector<Layer<T>> &getLayers() const { return layers; }

        /* Setters */
        std::vector<Layer<T>> &updateLayers() { return layers; }

    };

}

#endif //DNNSIM_NETWORK_H
