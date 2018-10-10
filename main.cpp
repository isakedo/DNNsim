#include <iostream>
#include <core/Network.h>
#include <interface/NetReader.h>

int main() {
    std::string folder =  "/home/isak/CLionProjects/DNNsim/models/bvlc_alexnet/";
    interface::NetReader reader = interface::NetReader("ALexNet",folder);
    core::Network* net = reader.read_network_csv();
    reader.read_weights_npy(net);
    reader.read_activations_npy(net);
    reader.read_output_activations_npy(net);
    net->start_simulation();
    return 0;
}