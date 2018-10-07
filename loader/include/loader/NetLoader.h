#ifndef DNNSIM_NETLOADER_H
#define DNNSIM_NETLOADER_H

#include <string>
#include <core/Network.h>
#include <core/Layer.h>
#include <core/ConvolutionalLayer.h>
#include <core/FullyConnectedLayer.h>


namespace loader {

    class NetLoader {

    private:

        /* Name of the network */
        std::string name;

        /* Path to the csv file containing the network architecture */
        std::string path;

    public:

        /* Constructor
         * @param _name     The name of the network
         * @param _path     Path to the csv file containing the network architecture
         */
        NetLoader(const std::string &_name, const std::string &_path) { name = _name; path = _path; }

        /* Load the trace file in the path and returns the network
         * @return          Network architecture
         * */
        core::Network* load_network();

    };

};


#endif //DNNSIM_NETLOADER_H
