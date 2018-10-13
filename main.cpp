#include <iostream>
#include <core/Network.h>
#include <interface/NetReader.h>

int main() {
    std::string folder =  "/home/isak/CLionProjects/DNNsim/models/bvlc_alexnet/";
    interface::NetReader reader = interface::NetReader("ALexNet",folder);
    std::shared_ptr<core::Network> net = reader.read_network_csv();
    reader.read_weights_npy(net);
    reader.read_activations_npy(net);
    reader.read_output_activations_npy(net);

    std::cout << net->getName() << std::endl;
    for(const std::shared_ptr<core::Layer> &layer : net->getLayers()) {
        std::cout << layer->getName() << " ";
        for(size_t i : layer->getWeights().getShape())  {
            std::cout << i << " ";
        }
        if(layer->getType() == core::CONV)
            std::cout << "CONVOLUTIONAL" << std::endl;
        else if(layer->getType() == core::FC)
            std::cout << "FULLY CONNECTED" << std::endl;
    }

    return 0;
}