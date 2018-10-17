#include <iostream>
#include <core/Network.h>
#include <interface/NetReader.h>
#include <interface/NetWriter.h>

/*
 * Exit states:
 *  0: Correct output
 *  1: Out of index when accessing numpy array
 *  2: Fail writing protobuf
 *  3: Fail reading protobuf
 */

int main() {
    /*  TODO
     *  Script style:
     *  Operation mode (Simulate or transform)
     *      If transform:
     *          -o path to the output file, -otype output type (Protobuf, Gzip)
     *          -i path to the input file/folder, -itype input type (Protobuf, Gzip, Trace)
     *      If simulator:
     *
     */
    std::string folder =  "/home/ali/CLionProjects/DNNsim/models/bvlc_alexnet/";
    interface::NetReader reader = interface::NetReader("bvlc_alexnet",folder);
    core::Network net = reader.read_network_csv();
    reader.read_weights_npy(net);
    reader.read_activations_npy(net);
    reader.read_output_activations_npy(net);

    std::cout << net.getName() << std::endl;
    for(const core::Layer &layer : net.getLayers()) {
        std::cout << layer.getName() << " ";
        for(size_t i : layer.getWeights().getShape())  {
            std::cout << i << " ";
        }
        if(layer.getType() == core::CONV)
            std::cout << "CONVOLUTIONAL" << std::endl;
        else if(layer.getType() == core::FC)
            std::cout << "FULLY CONNECTED" << std::endl;
    }
    std::cout << net.getLayers()[0].getWeights().get(0,0,0,0) << std::endl;
    std::cout << net.getLayers()[0].getWeights().get(0,0,1,0) << std::endl;
    std::cout << net.getLayers()[0].getWeights().get(0,2,1,0) << std::endl;
    std::cout << net.getLayers()[0].getWeights().get(0,2,1,8) << std::endl;

    interface::NetWriter writer = interface::NetWriter(folder);
    writer.write_network_protobuf(net,"alexnet");

    core::Network net2 = reader.read_network_protobuf("alexnet");
    std::cout << net2.getName() << std::endl;
    for(const core::Layer &layer : net2.getLayers()) {
        std::cout << layer.getName() << " ";
        for(size_t i : layer.getWeights().getShape())  {
            std::cout << i << " ";
        }
        if(layer.getType() == core::CONV)
            std::cout << "CONVOLUTIONAL" << std::endl;
        else if(layer.getType() == core::FC)
            std::cout << "FULLY CONNECTED" << std::endl;
    }
    std::cout << net2.getLayers()[0].getWeights().get(0,0,0,0) << std::endl;
    std::cout << net2.getLayers()[0].getWeights().get(0,0,1,0) << std::endl;
    std::cout << net2.getLayers()[0].getWeights().get(0,2,1,0) << std::endl;
    std::cout << net2.getLayers()[0].getWeights().get(0,2,1,8) << std::endl;

    writer.write_network_gzip(net2,"alexnet");

    core::Network net3 = reader.read_network_gzip("alexnet");
    std::cout << net3.getName() << std::endl;
    for(const core::Layer &layer : net3.getLayers()) {
        std::cout << layer.getName() << " ";
        for(size_t i : layer.getWeights().getShape())  {
            std::cout << i << " ";
        }
        if(layer.getType() == core::CONV)
            std::cout << "CONVOLUTIONAL" << std::endl;
        else if(layer.getType() == core::FC)
            std::cout << "FULLY CONNECTED" << std::endl;
    }
    std::cout << net3.getLayers()[0].getWeights().get(0,0,0,0) << std::endl;
    std::cout << net3.getLayers()[0].getWeights().get(0,0,1,0) << std::endl;
    std::cout << net3.getLayers()[0].getWeights().get(0,2,1,0) << std::endl;
    std::cout << net3.getLayers()[0].getWeights().get(0,2,1,8) << std::endl;


    return 0;
}