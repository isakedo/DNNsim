#ifndef DNNSIM_NETWRITER_H
#define DNNSIM_NETWRITER_H

#include <sys/common.h>
#include <core/Network.h>
#include <network.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>

namespace interface {

    template <typename T>
    class NetWriter {

    private:

        /* Layers that we want weights, activations, and output activations */
        const std::set<std::string> layers_data = {"Convolution","InnerProduct"};

        /* Path containing the network files */
        std::string path;

        /* Store a layer of the network into a protobuf layer
         * @param layer_proto   Pointer to a protobuf layer
         * @param layer         Layer of the network that want to be stored
         */
        void fillLayer(protobuf::Network_Layer* layer_proto, const core::Layer<T> &layer);

    public:

        /* Constructor
         * @param _name     The name of the network
         * @param _path     Path containing the files with the network architecture
         */

        explicit NetWriter(const std::string &_path){ path = _path;}

        /* Store the network in protobuf format
         * @param network       Network that want to be stored
         * @param path          Output file to store the network
         */
        void write_network_protobuf(const core::Network<T> &network);

        /* Store the network in Gzip protobuf format
         * @param network       Network that want to be stored
         * @param path          Output file to store the network
         */
        void write_network_gzip(const core::Network<T> &network);

    };

}

#endif //DNNSIM_NETWRITER_H
