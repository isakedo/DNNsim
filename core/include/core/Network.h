#ifndef DNNSIM_NETWORK_H
#define DNNSIM_NETWORK_H

#include <core/Layer.h>

#include <vector>
#include <string>
#include <memory>

namespace core {

    class Network {

    private:

        /* Name of the network */
        std::string name;

        /* Set of layers of the network*/
        std::vector<Layer> layers;

    public:

        /* Default constructor */
        Network() = default;

        /* Constructor
         * @param _name      The name of the network
         * @param _layers    Vector of layers
         */
        Network(const std::string &_name, const std::vector<Layer> &_layers) {
            name = _name; layers = _layers;
        }

        /* Getters */
        const std::string &getName() const { return name; }
        const std::vector<Layer> &getLayers() const { return layers; }

        /* Setters */
        std::vector<Layer> &updateLayers() { return layers; }

    };

}

#endif //DNNSIM_NETWORK_H
