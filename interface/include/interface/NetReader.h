#ifndef DNNSIM_NETREADER_H
#define DNNSIM_NETREADER_H

#include "Interface.h"
#include <core/Layer.h>
#include <caffe.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace interface {

    template <typename T>
    class NetReader : public Interface {

    private:

        /* Layers we want to load in the model */
        const std::set<std::string> allowed_layers = {"Convolution","InnerProduct","LSTM","Encoder","Decoder"};

        /* Name of the network */
        std::string name;

        /* Numpy activations batch to read from */
        uint32_t batch;

        /* Numpy activations epoch to read from */
        uint32_t epoch;

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
         * @param _batch    Numpy batch of the activations
         * @param _epoch    Numpy epoch of the training traces
         * @param _QUIET    Remove stdout messages
         */
        NetReader(const std::string &_name, uint32_t _batch, uint32_t _epoch, bool _QUIET) : Interface(_QUIET),
                batch(_batch), epoch(_epoch) {
            this->name = _name;
        }

        /* Load the caffe prototxt file inside the folder models and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_caffe();

        /* Load the trace file inside the folder models and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_trace_params();

        /* Load the conv params file inside the folder models and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_conv_params();

        /* Read the protobuf with the network in the models and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_protobuf();

        /* Read the weights schedule from the schedule in the models folder and the schedule
         * @return          Schedule for the network
         */
        std::vector<schedule> read_schedule_protobuf(const std::string &schedule_type);

        /* Read the weights into given network
         * @param network       Network with the layers already initialized
         */
        void read_weights_npy(core::Network<T> &network);

        /* Read the activations into given network
         * @param network       Network with the layers already initialized
         */
        void read_activations_npy(core::Network<T> &network);

        /* Read the weights from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_weights_npy(core::Network<T> &network);

        /* Read the activations from training traces into given network
         * @param network           Network with the layers already initialized
         */
        void read_training_activations_npy(core::Network<T> &network);

        /* Read the weight gradients from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_weight_gradients_npy(core::Network<T> &network);

        /* Read the activation gradients from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_input_gradients_npy(core::Network<T> &network);

        /* Read the output activation gradients from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_output_activation_gradients_npy(core::Network<T> &network);

        /* Read the precision for each layer
         * @param network       Network with the layers already initialized
         */
        void read_precision(core::Network<T> &network);

    };

}

#endif //DNNSIM_NETREADER_H
