
#include <interface/NetReader.h>

namespace interface {

    // From https://github.com/BVLC/caffe/blob/2a1c552b66f026c7508d390b526f2495ed3be594/src/caffe/util/io.cpp
    bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto) {
        int fd = open(filename, O_RDONLY);
        auto input = new google::protobuf::io::FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }

    template <typename T>
    core::Layer<T> read_layer_caffe(const caffe::LayerParameter &layer_caffe) {
        int Nn = -1, Kx = -1, Ky = -1, stride = -1, padding = -1;

        if(layer_caffe.type() == "Convolution") {
            Nn = layer_caffe.convolution_param().num_output();
            Kx = layer_caffe.convolution_param().kernel_size(0);
            Ky = layer_caffe.convolution_param().kernel_size(0);
            stride = layer_caffe.convolution_param().stride_size() == 0 ? 1 : layer_caffe.convolution_param().stride(0);
            padding = layer_caffe.convolution_param().pad_size() == 0 ? 0 : layer_caffe.convolution_param().pad(0);
        } else if (layer_caffe.type() == "InnerProduct") {
            Nn = layer_caffe.inner_product_param().num_output();
            Kx = 1; Ky = 1; stride = 1; padding = 0;

        } else if (layer_caffe.type() == "Pooling") {
            Kx = layer_caffe.pooling_param().kernel_size();
            Ky = layer_caffe.pooling_param().kernel_size();
            stride = layer_caffe.pooling_param().stride();
        }

        return core::Layer<T>(layer_caffe.type(),layer_caffe.name(),layer_caffe.bottom(0), Nn, Kx, Ky, stride, padding);
    }

    template <typename T>
    core::Network<T> NetReader<T>::read_network_caffe() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        caffe::NetParameter network;

        std::string name = (this->path.back() == '/' ? this->path : this->path + '/') + "train_val.prototxt";
        if (!ReadProtoFromTextFile(name.c_str(),&network)) {
            throw std::runtime_error("Failed to read prototxt");
        }

        for(const auto &layer : network.layer()) {
            if(this->layers_allowed.find(layer.type()) != this->layers_allowed.end()) {
                layers.emplace_back(read_layer_caffe<T>(layer));
            }
        }

        google::protobuf::ShutdownProtobufLibrary();

        return core::Network<T>(this->name,layers);
    }

    template <typename T>
    core::Layer<T> NetReader<T>::read_layer_proto(const protobuf::Network_Layer &layer_proto) {
        core::Layer<T> layer = core::Layer<T>(layer_proto.type(),layer_proto.name(),layer_proto.input(),
            layer_proto.nn(),layer_proto.kx(),layer_proto.ky(),layer_proto.stride(),layer_proto.padding());


        // Read weights, activations, and output activations only to the desired layers
        if(this->layers_data.find(layer_proto.type()) != this->layers_data.end()) {

            std::vector<size_t > weights_shape;
            for(const int value : layer_proto.wgt_shape())
                weights_shape.push_back((size_t)value);

            std::vector<T> weights_data;
            for(const T value : layer_proto.wgt_data())
                weights_data.push_back(value);

            cnpy::Array<T> weights; weights.set_values(weights_data,weights_shape);
            layer.setWeights(weights);

            std::vector<size_t > activations_shape;
            for(const int value : layer_proto.act_shape())
                activations_shape.push_back((size_t)value);

            std::vector<T> activations_data;
            for(const T value : layer_proto.act_data())
                activations_data.push_back(value);

            cnpy::Array<T> activations; activations.set_values(activations_data,activations_shape);
            layer.setActivations(activations);

            std::vector<size_t > out_activations_shape;
            for(const int value : layer_proto.out_act_shape())
                out_activations_shape.push_back((size_t)value);

            std::vector<T> out_activations_data;
            for(const T value : layer_proto.out_act_data())
                out_activations_data.push_back(value);

            cnpy::Array<T> out_activations; out_activations.set_values(out_activations_data,out_activations_shape);
            layer.setOutput_activations(out_activations);

        }

        return layer;
    }

    template <typename T>
    core::Network<T> NetReader<T>::read_network_protobuf() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        protobuf::Network network_proto;

        {
            // Read the existing network.
            std::fstream input(this->path, std::ios::in | std::ios::binary);
            if (!network_proto.ParseFromIstream(&input)) {
                throw std::runtime_error("Failed to read protobuf");
            }
        }

        std::string name = network_proto.name();

        for(const protobuf::Network_Layer &layer_proto : network_proto.layers())
            layers.emplace_back(read_layer_proto(layer_proto));

        google::protobuf::ShutdownProtobufLibrary();

        return core::Network<T>(this->name,layers);
    }


    template <typename T>
    core::Network<T> NetReader<T>::read_network_gzip() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        protobuf::Network network_proto;

        // Read the existing network.
        std::fstream input(this->path, std::ios::in | std::ios::binary);

        google::protobuf::io::IstreamInputStream inputFileStream(&input);
        google::protobuf::io::GzipInputStream gzipInputStream(&inputFileStream);

        if (!network_proto.ParseFromZeroCopyStream(&gzipInputStream)) {
            throw std::runtime_error("Failed to read Gzip protobuf");
        }

        std::string name = network_proto.name();

        for(const protobuf::Network_Layer &layer_proto : network_proto.layers())
            layers.emplace_back(read_layer_proto(layer_proto));

        google::protobuf::ShutdownProtobufLibrary();

        return core::Network<T>(this->name,layers);
    }

    template <typename T>
    void NetReader<T>::read_weights_npy(core::Network<T> &network) {
        std::string file_path = this->path.back() == '/' ? this->path : this->path + '/';
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "wgt-" + layer.getName() + ".npy" ;
                cnpy::Array<T> weights; weights.set_values(file_path + file);
                layer.setWeights(weights);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_activations_npy(core::Network<T> &network) {
        std::string file_path = this->path.back() == '/' ? this->path : this->path + '/';
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "act-" + layer.getName() + "-0.npy";
                cnpy::Array<T> activations; activations.set_values(file_path + file);
                layer.setActivations(activations);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_output_activations_npy(core::Network<T> &network) {
        std::string file_path = this->path.back() == '/' ? this->path : this->path + '/';
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "act-" + layer.getName() + "-0-out.npy" ;
                cnpy::Array<T> activations; activations.set_values(file_path + file);
                layer.setOutput_activations(activations);
            }
        }
    }

    INITIALISE_DATA_TYPES(NetReader);

}
