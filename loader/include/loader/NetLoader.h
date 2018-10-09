#ifndef DNNSIM_NETLOADER_H
#define DNNSIM_NETLOADER_H

#include <core/Network.h>
#include <core/Layer.h>
#include <core/ConvolutionalLayer.h>
#include <core/FullyConnectedLayer.h>

#include <string>
#include <fstream>
#include <sstream>

namespace loader {

    class NetLoader {

    private:

        /* Name of the network */
        std::string name;

        /* Path to the csv file with the network architecture */
        std::string path;

    public:

        /* Constructor
         * @param _name     The name of the network
         * @param _path     Path to the folder containing csv file with the network architecture
         */
        NetLoader(const std::string &_name, const std::string &_path){ name = _name; path = _path + "trace_params.csv";}

        /* Load the trace file in the path and returns the network
         * @return          Network architecture
         * */
        core::Network* load_network();

    };

}


#endif //DNNSIM_NETLOADER_H
