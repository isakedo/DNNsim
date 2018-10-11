#ifndef DNNSIM_NETREADER_H
#define DNNSIM_NETREADER_H

#include <core/Network.h>
#include <core/Layer.h>
#include <core/ConvolutionalLayer.h>
#include <core/FullyConnectedLayer.h>
#include <network.pb.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace interface {

    class NetReader {

    private:

        /* Name of the network */
        std::string name;

        /* Path to the folder containing the definition files */
        std::string path;

    public:

        /* Constructor
         * @param _name     The name of the network
         * @param _path     Path to the folder containing csv file with the network architecture
         */
        NetReader(const std::string &_name, const std::string &_path){ name = _name; path = _path;}

        /* Load the trace file inside the folder path and returns the network
         * @return          Network architecture
         * */
        core::Network* read_network_csv();

        /* Read the protobuf with the network in the path and returns the network
         * @return          Network architecture
         * */
        core::Network* read_network_protobuf();

        /* Read the weights into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_weights_npy(core::Network* network);

        /* Read the activations into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_activations_npy(core::Network* network);

        /* Read the output activations into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_output_activations_npy(core::Network* network);

    };

}

#endif //DNNSIM_NETREADER_H
