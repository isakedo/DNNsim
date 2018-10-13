
#include <interface/NetReader.h>

namespace interface {

    std::shared_ptr<core::Network> NetReader::read_network_csv() {
        std::vector<std::shared_ptr<core::Layer>> vec;
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

                vec.push_back(std::make_shared<core::Layer>(core::Layer(type,words[0],words[1],std::stoi(words[2]),
                        std::stoi(words[3]),std::stoi(words[4]), std::stoi(words[5]),std::stoi(words[6]))));
            }
            myfile.close();
        }
        std::shared_ptr<core::Network> network = std::make_shared<core::Network>(core::Network(this->name,vec));
        return network;
    }

    std::shared_ptr<core::Network> NetReader::read_network_protobuf() {
        return nullptr;     //TODO
    }

    void NetReader::read_weights_npy(std::shared_ptr<core::Network> network) {
        for(const std::shared_ptr<core::Layer> &layer : network->getLayers()) {
            std::string file = "wgt-" + layer->getName() + ".npy" ;
            cnpy::NumpyArray weights; weights.set_values(this->path + file);
            layer->setWeights(weights);
        }
    }

    void NetReader::read_activations_npy(std::shared_ptr<core::Network> network) {
        for(const std::shared_ptr<core::Layer> &layer : network->getLayers()) {
            std::string file = "act-" + layer->getName() + "-0.npy" ;
            cnpy::NumpyArray activations; activations.set_values(this->path + file);
            layer->setActivations(activations);
        }
    }

    void NetReader::read_output_activations_npy(std::shared_ptr<core::Network> network) {
        for(const std::shared_ptr<core::Layer> &layer : network->getLayers()) {
            std::string file = "act-" + layer->getName() + "-0-out.npy" ;
            cnpy::NumpyArray activations; activations.set_values(this->path + file);
            layer->setOutput_activations(activations);
        }
    }

}
