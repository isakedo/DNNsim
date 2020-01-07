#ifndef DNNSIM_NETWRITER_H
#define DNNSIM_NETWRITER_H

#include "Interface.h"

namespace interface {

    /**
     * Network writer
     * @tparam T Data type of the network to write
     */
    template <typename T>
    class NetWriter : public Interface {

    private:

        /** Name of the network */
        std::string name;

        /** Store a layer of the network into a protobuf layer
         * @param layer_proto   Pointer to a protobuf layer
         * @param layer         Layer of the network that want to be stored
         */
        void fill_layer(protobuf::Network_Layer* layer_proto, const base::Layer<T> &layer);

    public:

        /** Constructor
         * @param _name     The name of the network
         * @param _QUIET    Remove stdout messages
         */
        NetWriter(const std::string &_name, bool _QUIET) : Interface(_QUIET) {
            this->name = _name;
        }

        /** Store the network in protobuf format
         * @param network       Network that want to be stored
         */
        void write_network_protobuf(const base::Network<T> &network);

    };

}

#endif //DNNSIM_NETWRITER_H
