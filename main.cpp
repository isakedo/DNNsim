#include <iostream>
#include <core/Network.h>
#include <loader/NetLoader.h>
#include <loader/NumpyLoader.h>

int main() {
    std::string folder =  "/home/isak/CLionProjects/DNNsim/models/bvlc_alexnet/";
    loader::NetLoader netloader = loader::NetLoader("ALexNet",folder);
    loader::NumpyLoader npyLoader = loader::NumpyLoader(folder);
    core::Network* net = netloader.load_network();
    npyLoader.load_weights(net);
    npyLoader.load_activations(net);
    npyLoader.load_activations(net);
    net->start_simulation();
    return 0;
}