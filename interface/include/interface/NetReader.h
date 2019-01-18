#ifndef DNNSIM_NETREADER_H
#define DNNSIM_NETREADER_H

#include <sys/common.h>
#include <core/Network.h>
#include <core/Layer.h>
#include <core/BitTactical.h>
#include <network.pb.h>
#include <caffe.pb.h>
#include <schedule.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>

namespace interface {

    template <typename T>
    class NetReader {

    private:

        /* Layers we want to load in the model */
        const std::set<std::string> layers_allowed = {"Convolution","InnerProduct","ReLU"};

        /* Layers that we want weights, activations, and output activations */
        const std::set<std::string> layers_data = {"Convolution","InnerProduct"};

        /* Name of the network */
        std::string name;

        /* Also read bias and output activations */
        bool activate_bias_and_out_act;

        /* Check if the path exists
         * @param path  Path we want to check
         */
        void check_path(const std::string &path);

        /* Return the name of the file depending on current type
         * @return Name of input file
         */
        std::string inputName();

        /* Return the layer parsed from the caffe prototxt file
         * @param layer_caffe   prototxt layer
         */
        core::Layer<T> read_layer_caffe(const caffe::LayerParameter &layer_caffe);

        /* Return the layer parsed from the protobuf file
         * @param layer_proto   protobuf layer
         */
        core::Layer<T> read_layer_proto(const protobuf::Network_Layer &layer_proto);

    public:

        /* Constructor
         * @param _name     The name of the network
         * @param _activate_bias_and_out_act    Also write bias and output activations
         */
        NetReader(const std::string &_name, bool _activate_bias_and_out_act) :
                activate_bias_and_out_act(_activate_bias_and_out_act) { this->name = _name; }

        /* Load the trace file inside the folder path and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_caffe();

        /* Read the protobuf with the network in the path and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_protobuf();

        /* Read the gzip with the network in the path and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_gzip();

        /* Read the weights schedule from the schedule in the path and the schedule
         * @return          Schedule for the network
         */
        std::vector<schedule> read_schedule_protobuf(const std::string &schedule_type);

        /* Read the weights into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_weights_npy(core::Network<T> &network);

        /* Read the bias into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_bias_npy(core::Network<T> &network);

        /* Read the activations into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_activations_npy(core::Network<T> &network);

        /* Read the output activations into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_output_activations_npy(core::Network<T> &network);

        /* Read the precision for each layer
         * @param network       Network with the layers already initialized
         */
        void read_precision(core::Network<T> &network);

    };

}

#endif //DNNSIM_NETREADER_H
