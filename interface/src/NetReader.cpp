
#include <interface/NetReader.h>

namespace interface {

    bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto) {
        int fd = open(filename, O_RDONLY);
        auto input = new google::protobuf::io::FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }

    template <typename T>
    void NetReader<T>::check_path(const std::string &path) {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    template <typename T>
    std::string NetReader<T>::inputName() {
        std::string output_name = this->name;
        std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time
        output_name += "-" + type;
        return output_name;
    }

    template <typename T>
    core::Layer<T> NetReader<T>::read_layer_caffe(const caffe::LayerParameter &layer_caffe) {
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

        } else if (layer_caffe.type() == "LSTM") {
            Nn = layer_caffe.recurrent_param().num_output();
            Kx = 1; Ky = 1; stride = 1; padding = 0;
        }

        std::string name = layer_caffe.name();
        std::replace( name.begin(), name.end(), '/', '-'); // Sanitize name
        return core::Layer<T>(layer_caffe.type(),name,layer_caffe.bottom(0), Nn, Kx, Ky, stride, padding);
    }

    template <typename T>
    core::Network<T> NetReader<T>::read_network_caffe() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        caffe::NetParameter network;

        check_path("models/" + this->name);
        std::string path = "models/" + this->name + "/train_val.prototxt";
        check_path(path);
        if (!ReadProtoFromTextFile(path.c_str(),&network)) {
            throw std::runtime_error("Failed to read prototxt");
        }

        for(const auto &layer : network.layer()) {
            if(this->layers_allowed.find(layer.type()) != this->layers_allowed.end()) {
                layers.emplace_back(read_layer_caffe(layer));
            }
        }

        return core::Network<T>(this->name,layers);
    }

    template <typename T>
    core::Network<T> NetReader<T>::read_network_trace_params() {

        std::vector<core::Layer<T>> layers;

        check_path("models/" + this->name);
        std::string path = "models/" + this->name + "/trace_params.csv";
        check_path(path);

        std::ifstream myfile (path);
        if (myfile.is_open()) {

            std::string line;
            while (getline(myfile,line)) {

                std::vector<std::string> words;
                std::string word;
                std::stringstream ss_line(line);
                while (getline(ss_line,word,','))
                    words.push_back(word);

                std::string type;
                if(words[3] == words[4] && words[4] == words[5] && words[5] == "1") {
                    if (words[0].at(0) == 'l' && words[0].at(1) == 's')
                        type = "LSTM";
                    else
                        type = "InnerProduct";
                } else
                    type = "Convolution";

                layers.emplace_back(core::Layer<T>(type,words[0],words[1],std::stoi(words[2]), std::stoi(words[3]),
                        std::stoi(words[4]), std::stoi(words[5]),std::stoi(words[6])));
            }
            myfile.close();
        }

        return core::Network<T>(this->name,layers);
    }

    template <typename T>
    core::Layer<T> NetReader<T>::read_layer_proto(const protobuf::Network_Layer &layer_proto) {
        core::Layer<T> layer = core::Layer<T>(layer_proto.type(),layer_proto.name(),layer_proto.input(),
        layer_proto.nn(),layer_proto.kx(),layer_proto.ky(),layer_proto.stride(),layer_proto.padding(),
        std::make_tuple<int,int>(layer_proto.act_mag(),layer_proto.act_prec()),
        std::make_tuple<int,int>(layer_proto.wgt_mag(),layer_proto.wgt_prec()));

        // Read weights, activations, and output activations only to the desired layers
        if(this->layers_data.find(layer_proto.type()) != this->layers_data.end()) {

            std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time

            std::vector<size_t> weights_shape;
            for (const int value : layer_proto.wgt_shape())
                weights_shape.push_back((size_t) value);

            std::vector<size_t> activations_shape;
            for (const int value : layer_proto.act_shape())
                activations_shape.push_back((size_t) value);

            std::vector<size_t> biases_shape;
            std::vector<size_t> out_activations_shape;
            if (this->activate_bias_and_out_act) {
                for (const int value : layer_proto.bias_shape())
                    biases_shape.push_back((size_t) value);
                for (const int value : layer_proto.out_act_shape())
                    out_activations_shape.push_back((size_t) value);
            }

            std::vector<T> weights_data;
            std::vector<T> biases_data;
            std::vector<T> activations_data;
            std::vector<T> out_activations_data;


            if (type == "f4") {
                for (const auto &value : layer_proto.wgt_data_flt())
                    weights_data.push_back(value);

                for (const auto value : layer_proto.act_data_flt())
                    activations_data.push_back(value);

                if (this->activate_bias_and_out_act) {
                    for (const auto value : layer_proto.bias_data_flt())
                        biases_data.push_back(value);
                    for (const auto value : layer_proto.out_act_data_flt())
                        out_activations_data.push_back(value);
                }
            } else if (type == "t2") {
                for (const auto &value : layer_proto.wgt_data_fxd())
                    weights_data.push_back(value);

                for (const auto value : layer_proto.act_data_fxd())
                    activations_data.push_back(value);

                if (this->activate_bias_and_out_act) {
                    for (const auto value : layer_proto.bias_data_fxd())
                        biases_data.push_back(value);
                    for (const auto value : layer_proto.out_act_data_fxd())
                        out_activations_data.push_back(value);
                }
            }

            cnpy::Array<T> weights; weights.set_values(weights_data,weights_shape);
            layer.setWeights(weights);

            cnpy::Array<T> activations; activations.set_values(activations_data,activations_shape);
            layer.setActivations(activations);

            if (this->activate_bias_and_out_act) {
                cnpy::Array<T> biases; biases.set_values(biases_data,biases_shape);
                layer.setBias(biases);
                cnpy::Array<T> out_activations; out_activations.set_values(out_activations_data,out_activations_shape);
                layer.setOutput_activations(out_activations);
            }

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
            check_path("net_traces/" + this->name);
            std::string path = "net_traces/" + this->name + '/' + inputName() + ".proto";
            check_path(path);
            std::fstream input(path,std::ios::in | std::ios::binary);
            if (!network_proto.ParseFromIstream(&input)) {
                throw std::runtime_error("Failed to read protobuf");
            }
        }

        std::string name = network_proto.name();

        for(const protobuf::Network_Layer &layer_proto : network_proto.layers())
            layers.emplace_back(read_layer_proto(layer_proto));

        return core::Network<T>(this->name,layers);
    }


    template <typename T>
    core::Network<T> NetReader<T>::read_network_gzip() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        protobuf::Network network_proto;

        // Read the existing network.
        check_path("net_traces/" + this->name);
        std::string path = "net_traces/" + this->name + '/' + inputName() + ".gz";
        check_path(path);
        std::fstream input(path, std::ios::in | std::ios::binary);

        google::protobuf::io::IstreamInputStream inputFileStream(&input);
        google::protobuf::io::GzipInputStream gzipInputStream(&inputFileStream);

        if (!network_proto.ParseFromZeroCopyStream(&gzipInputStream)) {
            throw std::runtime_error("Failed to read Gzip protobuf");
        }

        std::string name = network_proto.name();

        for(const protobuf::Network_Layer &layer_proto : network_proto.layers())
            layers.emplace_back(read_layer_proto(layer_proto));

        return core::Network<T>(this->name,layers);
    }

    template <typename T>
    std::vector<schedule> NetReader<T>::read_schedule_protobuf(const std::string &schedule_type) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        protobuf::Schedule network_schedule_proto;

        {
            // Read the existing network.
            check_path("net_traces/" + this->name);
            std::string path = "net_traces/" + this->name + '/' + inputName() + "_" + schedule_type + "_schedule.proto";
            check_path(path);
            std::fstream input(path, std::ios::in | std::ios::binary);
            if (!network_schedule_proto.ParseFromIstream(&input)) {
                throw std::runtime_error("Failed to read protobuf");
            }
        }

        std::vector<schedule> network_schedule;

        for(const auto &schedule_layer_proto : network_schedule_proto.layers()) {
            schedule dense_schedule;
            for(const auto &schedule_time_proto : schedule_layer_proto.times()) {
                time_schedule window_schedule;
                for(const auto &schedule_tuple_proto : schedule_time_proto.tuples()) {
                    schedule_tuple dense_schedule_tuple = std::make_tuple(schedule_tuple_proto.channel(),
                            schedule_tuple_proto.kernel_x(), schedule_tuple_proto.kernel_y(),
                            schedule_tuple_proto.wgt_bits());
                    window_schedule.push_back(dense_schedule_tuple);
                }
                dense_schedule.push_back(window_schedule);
            }
            network_schedule.push_back(dense_schedule);
        }

        return network_schedule;
    }

    template <typename T>
    void NetReader<T>::read_weights_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/wgt-" + layer.getName() + ".npy" ;
                cnpy::Array<T> weights; weights.set_values("net_traces/" + this->name + file);
                layer.setWeights(weights);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_bias_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/bias-" + layer.getName() + ".npy" ;
                cnpy::Array<T> bias; bias.set_values("net_traces/" + this->name + file);
                layer.setBias(bias);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_activations_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/act-" + layer.getName() + "-0.npy";
                cnpy::Array<T> activations; activations.set_values("net_traces/" + this->name + file);
                layer.setActivations(activations);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_output_activations_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/act-" + layer.getName() + "-0-out.npy" ;
                cnpy::Array<T> activations; activations.set_values("net_traces/" + this->name + file);
                layer.setOutput_activations(activations);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_precision(core::Network<T> &network) {

        std::string line;
        std::stringstream ss_line;
        std::vector<int> act_mag;
        std::vector<int> act_prec;
        std::vector<int> wgt_mag;
        std::vector<int> wgt_prec;

        std::ifstream myfile ("models/" + this->name + "/precision.txt");
        if (myfile.is_open()) {
            std::string word;

            getline(myfile,line); // Remove first line

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                act_mag.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                act_prec.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                wgt_mag.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                wgt_prec.push_back(stoi(word));

            myfile.close();

            int i = 0;
            for(core::Layer<T> &layer : network.updateLayers()) {
                if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                    layer.setAct_precision(std::make_tuple(act_mag[i], act_prec[i]));
                    layer.setWgt_precision(std::make_tuple(wgt_mag[i], wgt_prec[i]));
                    i++;
                } else {
                    layer.setAct_precision(std::make_tuple(0,0));
                    layer.setWgt_precision(std::make_tuple(0,0));
                }
            }

        } else {
            // Generic precision
            int i = 0;
            for(core::Layer<T> &layer : network.updateLayers()) {
                if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                    layer.setAct_precision(std::make_tuple(1 + 13, 2));
                    layer.setWgt_precision(std::make_tuple(1 + 0, 15));
                    i++;
                } else {
                    layer.setAct_precision(std::make_tuple(0,0));
                    layer.setWgt_precision(std::make_tuple(0,0));
                }
            }
        }
    }

    INITIALISE_DATA_TYPES(NetReader);

}