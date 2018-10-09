
#include <core/Network.h>

namespace core {

    void Network::start_simulation() {
        printf("Starting simulation\n");
        for(const std::shared_ptr<Layer> &layer: layers)
            layer->compute();
    }

    const std::string &Network::getName() const {
        return name;
    }

    const std::vector<std::shared_ptr<Layer>> &Network::getLayers() const {
        return layers;
    }

}

