
#include <loader/NetLoader.h>

namespace loader {

    core::Network* NetLoader::load_network() {
        std::vector<std::shared_ptr<core::Layer>> vec;
        vec.push_back(std::make_shared<core::ConvolutionalLayer>(core::ConvolutionalLayer("a","b",1,1,1,1,1)));
        return new core::Network(this->name,vec);
    }

};

