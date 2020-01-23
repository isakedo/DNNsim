
#include <base/NetReader.h>

namespace base {

    void check_path(const std::string &path) {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto) {
        int fd = open(filename, O_RDONLY);
        auto input = new google::protobuf::io::FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }

    template <typename T>
    base::Layer<T> NetReader<T>::read_layer_caffe(const caffe::LayerParameter &layer_caffe) {
        int stride = -1, padding = -1;

        if(layer_caffe.type() == "Convolution") {
            stride = layer_caffe.convolution_param().stride_size() == 0 ? 1 : layer_caffe.convolution_param().stride(0);
            padding = layer_caffe.convolution_param().pad_size() == 0 ? 0 : layer_caffe.convolution_param().pad(0);
        } else if (layer_caffe.type() == "InnerProduct" || layer_caffe.type() == "LSTM") {
            stride = 1;
            padding = 0;
        }

        std::string layer_name = layer_caffe.name();
        std::replace( layer_name.begin(), layer_name.end(), '/', '-'); // Sanitize name
        std::string type = (layer_name.find("fc") != std::string::npos) ? "InnerProduct" : layer_caffe.type();
        type = (layer_name.find("lstm") != std::string::npos) ? "LSTM" : type;
        type = (layer_name.find("forward") != std::string::npos) ? "LSTM" : type;
        type = (layer_name.find("backward") != std::string::npos) ? "LSTM" : type;
        return base::Layer<T>(layer_name, type, stride, padding);
    }

    template <typename T>
    base::Network<T> NetReader<T>::read_network_caffe() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<base::Layer<T>> layers;
        caffe::NetParameter network;

        check_path("models/" + this->name);
        std::string path = "models/" + this->name + "/train_val.prototxt";
        check_path(path);
        if (!ReadProtoFromTextFile(path.c_str(),&network)) {
            throw std::runtime_error("Failed to read prototxt");
        }

        for(const auto &layer : network.layer()) {
           if(this->allowed_layers.find(layer.type()) != this->allowed_layers.end()) {
               layers.emplace_back(read_layer_caffe(layer));
           }
        }

        if(!QUIET) std::cout << "Network loaded from Caffe prototxt model definition" << std::endl;

        return base::Network<T>(this->name,layers);
    }

    template <typename T>
    base::Network<T> NetReader<T>::read_network_csv() {

        std::vector<base::Layer<T>> layers;

        check_path("models/" + this->name);
        std::string path = "models/" + this->name + "/model.csv";
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
                if(words[1] == "conv")
                    type = "Convolution";
                else if (words[1] == "fc")
                    type = "InnerProduct";
                else if (words[1] == "lstm")
                    type = "LSTM";
                else
                    throw std::runtime_error("Failed to read model.csv: Unknown layer type");

                auto layer_name = words[0];
                std::replace(layer_name.begin(), layer_name.end(), '/', '-'); // Sanitize name

                auto layer = base::Layer<T>(layer_name, type, stoi(words[2]), stoi(words[3]));
                layers.emplace_back(layer);
            }
            myfile.close();
        }

        if(!QUIET) std::cout << "Network loaded from convolutional params model definition" << std::endl;

        return base::Network<T>(this->name, layers);
    }

    template <typename T>
    void NetReader<T>::read_weights_npy(base::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(base::Layer<T> &layer : network.updateLayers()) {
            std::string file = "/wgt-" + layer.getName() + ".npy" ;
            base::Array<T> weights; weights.set_values("net_traces/" + this->name + file);
            layer.setWeights(weights);
        }

        if(!QUIET) std::cout << "Weight traces loaded from numpy arrays" << std::endl;

    }

    template <typename T>
    void NetReader<T>::read_activations_npy(base::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(base::Layer<T> &layer : network.updateLayers()) {
            std::string file = "/act-" + layer.getName() + "-" + std::to_string(batch) + ".npy";
            base::Array<T> activations; activations.set_values("net_traces/" + this->name + file);
            layer.setActivations(activations);
        }

        if(!QUIET) std::cout << "Activation traces loaded from numpy arrays" << std::endl;

    }

    template <typename T>
    void NetReader<T>::read_precision(base::Network<T> &network) {

        if(network.isTensorflow_8b()) {
            int i = 0;
            for(base::Layer<T> &layer : network.updateLayers()) {
                layer.setAct_precision(8,7,0);
                layer.setWgt_precision(8,7,0);
                i++;
            }

            if(!QUIET) std::cout << "Using generic precisions for Tensorflow 8b quantization" << std::endl;

            return;
        }

        if(network.isIntelINQ()) {
            int i = 0;
            for(base::Layer<T> &layer : network.updateLayers()) {
                layer.setAct_precision(16,15,0);
                layer.setWgt_precision(8,7,0);
                i++;
            }

            if(!QUIET) std::cout << "Using generic precisions for Intel INQ quantization" << std::endl;

            return;
        }

        std::string line;
        std::stringstream ss_line;
        std::vector<int> act_mag;
        std::vector<int> act_frac;
        std::vector<int> wgt_mag;
        std::vector<int> wgt_frac;

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
                act_frac.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                wgt_mag.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                wgt_frac.push_back(stoi(word));

            myfile.close();

            int i = 0;
            for(base::Layer<T> &layer : network.updateLayers()) {
                layer.setAct_precision(act_mag[i] + act_frac[i], act_mag[i] - 1, act_frac[i]);
                layer.setWgt_precision(wgt_mag[i] + wgt_frac[i], wgt_mag[i] - 1, wgt_frac[i]);
                i++;
            }

            if(!QUIET) std::cout << "Profiled precisions read from file" << std::endl;

        } else {
            // Generic precision
            int i = 0;
            for(base::Layer<T> &layer : network.updateLayers()) {
                if (network.getNetwork_bits() == 8) {
                    layer.setAct_precision(8,6,1);
                    layer.setWgt_precision(8,0,7);
                } else {
                    layer.setAct_precision(16,13,2);
                    layer.setWgt_precision(16,0,15);
                }
                i++;
            }

            if(!QUIET) std::cout << "No profiled precisions: Using generic precisions" << std::endl;

        }
    }

    INITIALISE_DATA_TYPES(NetReader);

}
