
#include <interface/NetReader.h>

namespace interface {

    core::Network NetReader::read_network_csv() {
        std::vector<core::Layer> vec;
        std::string line;

        std::ifstream myfile (this->path + "trace_params.csv");
        if (myfile.is_open()) {

            while (getline(myfile,line)) {

                std::vector<std::string> words;
                std::string word;
                std::stringstream ss_line(line);
                while (getline(ss_line,word,','))
                    words.push_back(word);

                core::Type type = core::INIT;
                if(words[0].at(0) == 'c')
                    type = core::CONV;
                else if (words[0].at(0) == 'f')
                    type = core::FC;
                else if (words[0].at(0) == 'i')
                    type = core::INCEP;
                else if (words[0].at(0) == 'l')
                    type = core::LOSS;

                vec.emplace_back(core::Layer(type,words[0],words[1],std::stoi(words[2]), std::stoi(words[3]),
                        std::stoi(words[4]), std::stoi(words[5]),std::stoi(words[6])));
            }
            myfile.close();
        }

        return core::Network(this->name,vec);
    }

    core::Layer NetReader::read_layer_proto(const protobuf::Network_Layer &layer_proto) {
        core::Layer layer = core::Layer((core::Type)layer_proto.type(),layer_proto.name(),layer_proto.input(),
            layer_proto.nn(),layer_proto.kx(),layer_proto.ky(),layer_proto.stride(),layer_proto.padding());

        std::vector<size_t > weights_shape;
        for(const int value : layer_proto.wgt_shape())
            weights_shape.push_back((size_t)value);

        std::vector<float> weights_data;
        for(const float value : layer_proto.wgt_data())
            weights_data.push_back(value);

        cnpy::Array weights; weights.set_values(weights_data,weights_shape);
        layer.setWeights(weights);

        std::vector<size_t > activations_shape;
        for(const int value : layer_proto.act_shape())
            activations_shape.push_back((size_t)value);

        std::vector<float> activations_data;
        for(const float value : layer_proto.act_data())
            activations_data.push_back(value);

        cnpy::Array activations; activations.set_values(activations_data,activations_shape);
        layer.setActivations(activations);

        std::vector<size_t > out_activations_shape;
        for(const int value : layer_proto.out_act_shape())
            out_activations_shape.push_back((size_t)value);

        std::vector<float> out_activations_data;
        for(const float value : layer_proto.out_act_data())
            out_activations_data.push_back(value);

        cnpy::Array out_activations; out_activations.set_values(out_activations_data,out_activations_shape);
        layer.setOutput_activations(out_activations);

        return layer;
    }

    core::Network NetReader::read_network_protobuf() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer> layers;
        protobuf::Network network_proto;

        {
            // Read the existing network.
            std::fstream input(this->path + "alexnet", std::ios::in | std::ios::binary);
            if (!network_proto.ParseFromIstream(&input)) {
                std::cerr << "Failed to parse address book." << std::endl;
                exit(3);
            }
        }

        std::string name = network_proto.name();

        for(const protobuf::Network_Layer &layer_proto : network_proto.layers())
            layers.emplace_back(read_layer_proto(layer_proto));

        google::protobuf::ShutdownProtobufLibrary();

        return core::Network(this->name,layers);
    }

    void NetReader::read_weights_npy(core::Network &network) {
        for(core::Layer &layer : network.updateLayers()) {
            std::string file = "wgt-" + layer.getName() + ".npy" ;
            cnpy::Array weights; weights.set_values(this->path + file);
            layer.setWeights(weights);
        }
    }

    void NetReader::read_activations_npy(core::Network &network) {
        for(core::Layer &layer : network.updateLayers()) {
            std::string file = "act-" + layer.getName() + "-0.npy" ;
            cnpy::Array activations; activations.set_values(this->path + file);
            layer.setActivations(activations);
        }
    }

    void NetReader::read_output_activations_npy(core::Network &network) {
        for(core::Layer &layer : network.updateLayers()) {
            std::string file = "act-" + layer.getName() + "-0-out.npy" ;
            cnpy::Array activations; activations.set_values(this->path + file);
            layer.setOutput_activations(activations);
        }
    }

}
