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
        const std::set<std::string> layers_allowed = {"Convolution","InnerProduct","LSTM","ReLU"};

        /* Layers that we want weights, activations, and output activations */
        const std::set<std::string> layers_data = {"Convolution","InnerProduct","LSTM","Encoder","Decoder"};

        /* Name of the network */
        std::string name;

        /* Also read bias and output activations */
        bool bias_and_out_act;

        /* Numpy activations batch to read from */
        int batch;

        /* Numpy activations epoch to read from */
        int epoch;

        /* Tensorflow 8 bit quantization */
        bool TENSORFLOW_8b;

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
         * @param _name                 The name of the network
         * @param _bias_and_out_act     Also write bias and output activations
         * @param _batch                Numpy batch of the activations
         * @param _epoch                Numpy epoch of the training traces
         * @param _TENSORFLOW_8b        Activate Tensorflow 8b quantization
         */
        NetReader(const std::string &_name, bool _bias_and_out_act, int _batch, int _epoch, bool _TENSORFLOW_8b) :
                bias_and_out_act(_bias_and_out_act), batch(_batch), epoch(_epoch), TENSORFLOW_8b(_TENSORFLOW_8b) {
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

        /* Read the gzip with the network in the models and returns the network
         * @return          Network architecture
         */
        core::Network<T> read_network_gzip();

        /* Read the weights schedule from the schedule in the models folder and the schedule
         * @return          Schedule for the network
         */
        std::vector<schedule> read_schedule_protobuf(const std::string &schedule_type);

        /* Read the weights into given network
         * @param network       Network with the layers already initialized
         */
        void read_weights_npy(core::Network<T> &network);

        /* Read the bias into given network
         * @param network       Network with the layers already initialized
         */
        void read_bias_npy(core::Network<T> &network);

        /* Read the activations into given network
         * @param network       Network with the layers already initialized
         */
        void read_activations_npy(core::Network<T> &network);

        /* Read the output activations into initialized given network
         * @param network       Network with the layers already initialized
         */
        void read_output_activations_npy(core::Network<T> &network);

        /* Read the weights from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_weights_npy(core::Network<T> &network);

        /* Read the bias from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_bias_npy(core::Network<T> &network);

        /* Read the activations from training traces into given network
         * @param network           Network with the layers already initialized
         * @param decoder_states    Number of states(steps) in the decoder traces
         */
        void read_training_activations_npy(core::Network<T> &network, uint16_t decoder_states = 0);

        /* Read the weight gradients from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_weight_gradients_npy(core::Network<T> &network);

        /* Read the bias gradients from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_bias_gradients_npy(core::Network<T> &network);

        /* Read the activation gradients from training traces into given network
         * @param network       Network with the layers already initialized
         */
        void read_training_activation_gradients_npy(core::Network<T> &network);

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
