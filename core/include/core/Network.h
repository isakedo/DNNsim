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
        std::vector<std::shared_ptr<Layer>> layers;

    public:

        /* Constructor
         * @param _name      The name of the network
         * @param _layers    Vector of layers
         */
        Network(const std::string &_name, const std::vector<std::shared_ptr<Layer>> &_layers) {
            name = _name; layers = _layers;
        }

        /* Getters */
        const std::string &getName() const { return name; }
        const std::vector<std::shared_ptr<Layer>> &getLayers() const { return layers; }

    };

}

#endif //DNNSIM_NETWORK_H
