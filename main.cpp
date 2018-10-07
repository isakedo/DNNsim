#include <iostream>
#include <core/Network.h>
#include <loader/NetLoader.h>

int main() {
    loader::NetLoader netloader = loader::NetLoader("ALexNet","/home/isak/CLionProjects/DNNsim/models/bvlc_alexnet/trace_params.csv");
    core::Network* net = netloader.load_network();
    net->start_simulation();
    return 0;
}