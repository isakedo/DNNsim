#ifndef DNNSIM_NETWRITER_H
#define DNNSIM_NETWRITER_H

#include <core/Network.h>
#include <network.pb.h>

namespace interface {

    class NetWriter {

    public:

        /* Store the network in protobuf format
         * @param network       Network that want to be stored
         * @param path          Output file to store the network
         * */
        void write_network_protobuf(const core::Network &network, const std::string &path);

    };

}

#endif //DNNSIM_NETWRITER_H
