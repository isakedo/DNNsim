
#include <interface/NetReader.h>

namespace interface {

    core::Network* NetReader::read_network_csv() {
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

                //This can be improved
                if(words[0].at(0) == 'c')
                    vec.push_back(std::make_shared<core::ConvolutionalLayer>(core::ConvolutionalLayer(
                            words[0],words[1],std::stoi(words[2]),std::stoi(words[3]),std::stoi(words[4]),
                            std::stoi(words[5]),std::stoi(words[6]))));
                else if (words[0].at(0) == 'f')
                    vec.push_back(std::make_shared<core::FullyConnectedLayer>(core::FullyConnectedLayer(
                            words[0],words[1],std::stoi(words[2]),std::stoi(words[3]),std::stoi(words[4]),
                            std::stoi(words[5]),std::stoi(words[6]))));
            }
            myfile.close();
        }
        return new core::Network(this->name,vec);
    }

    core::Network* NetReader::read_network_protobuf() {
        return nullptr;     //TODO
    }

    void NetReader::read_weights_npy(core::Network* network) {
        //TODO
    }

    void NetReader::read_activations_npy(core::Network* network) {
        //TODO
    }

    void NetReader::read_output_activations_npy(core::Network *network) {
        //TODO
    }

}
