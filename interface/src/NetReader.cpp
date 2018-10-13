
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

    core::Network NetReader::read_network_protobuf() {
        std::vector<core::Layer> layers;
        return core::Network(this->name,layers);     //TODO
    }

    void NetReader::read_weights_npy(core::Network &network) {
        for(core::Layer &layer : network.updateLayers()) {
            std::string file = "wgt-" + layer.getName() + ".npy" ;
            cnpy::NumpyArray weights; weights.set_values(this->path + file);
            layer.setWeights(weights);
        }
    }

    void NetReader::read_activations_npy(core::Network &network) {
        for(core::Layer &layer : network.updateLayers()) {
            std::string file = "act-" + layer.getName() + "-0.npy" ;
            cnpy::NumpyArray activations; activations.set_values(this->path + file);
            layer.setActivations(activations);
        }
    }

    void NetReader::read_output_activations_npy(core::Network &network) {
        for(core::Layer &layer : network.updateLayers()) {
            std::string file = "act-" + layer.getName() + "-0-out.npy" ;
            cnpy::NumpyArray activations; activations.set_values(this->path + file);
            layer.setOutput_activations(activations);
        }
    }

}
