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

        /* Name of the network */
        std::string name;

        /* Specify if transform the data when writing the network */
        /* Allowed values: Not, Fixed16 */
        std::string data_conversion;

        /* Also write bias and output activations */
        bool activate_bias_and_out_act;

        /* Check if the path exists
         * @param path  Path we want to check
         */
        void check_path(const std::string &path);

        /* Return the name of the file depending on current type and conversion
         * @return Name of output file
         */
        std::string outputName();

        /* Store a layer of the network into a protobuf layer
         * @param layer_proto   Pointer to a protobuf layer
         * @param layer         Layer of the network that want to be stored
         */
        void fillLayer(protobuf::Network_Layer* layer_proto, const core::Layer<T> &layer);

    public:

        /* Constructor
         * @param _path                         Path containing the files with the network architecture
         * @param _data_conversion              Specification of the data transformation when writing the network
         * @param _activate_bias_and_out_act    Also write bias and output activations
         */
        NetWriter(const std::string &_name, const std::string &_data_conversion, bool _activate_bias_and_out_act) :
            activate_bias_and_out_act(_activate_bias_and_out_act) { this->name = _name;
            this->data_conversion = _data_conversion; }

        /* Store the network in protobuf format
         * @param network       Network that want to be stored
         */
        void write_network_protobuf(const core::Network<T> &network);

        /* Store the network in Gzip protobuf format
         * @param network       Network that want to be stored
         */
        void write_network_gzip(const core::Network<T> &network);

    };

}

#endif //DNNSIM_NETWRITER_H
