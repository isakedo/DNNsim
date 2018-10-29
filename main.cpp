/*
Copyright (c) 2018 Andreas Moshovos
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <sys/common.h>
#include <sys/cxxopts.h>
#include <core/Network.h>
#include <core/InferenceSimulator.h>
#include <interface/NetReader.h>
#include <interface/NetWriter.h>

template <typename T>
core::Network<T> read(const cxxopts::Options &options) {
    core::Network<T> network;
    std::string input_type = options["itype"].as<std::string>();
    std::string path = options["i"].as<std::string>();
    if (input_type == "Caffe") path = path.back() == '/' ? path : path + '/';

    // Get name from the path
    size_t second_to_last = path.find_last_of('/', path.find_last_of('/')-1);
    std::string name = path.substr(second_to_last+1);
    std::size_t last = name.find_last_of('/');
    name = name.substr(0,last);

    // Read the network
    interface::NetReader<T> reader = interface::NetReader<T>(name,path);
    if (input_type == "Caffe") {
        network = reader.read_network_caffe();
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);
        reader.read_output_activations_npy(network);
    } else if (input_type == "Protobuf") {
        network = reader.read_network_protobuf();
    } else {
        network = reader.read_network_gzip();
    }

    return network;

}

template <typename T>
void write(const core::Network<T> &network, const cxxopts::Options &options) {
    // Write network
    interface::NetWriter<T> writer = interface::NetWriter<T>(options["o"].as<std::string>());
    std::string output_type = options["otype"].as<std::string>();
    if (output_type == "Protobuf") {
        writer.write_network_protobuf(network);
    } else {
        writer.write_network_gzip(network);
    }
}

void check_path(std::string const &path)
{
    std::ifstream file(path);
    if(!file.good()) {
        throw std::runtime_error("The path " + path + " does not exist.");
    }
}

void check_options(const cxxopts::Options &options)
{
    if(options.count("tool") == 0) {
        throw std::runtime_error("Please provide first the desired tool <Simulator|Transform>.");
    } else {
        std::string value = options["tool"].as<std::string>();
        if(value  != "Simulator" && value != "Transform")
            throw std::runtime_error("Please provide first the desired tool <Simulator|Transform>.");
    }

    if(options.count("d") == 0) {
        throw std::runtime_error("Please provide the data type with -d <Float32|Fixed16>.");
    } else {
        std::string value = options["d"].as<std::string>();
        if(value  != "Float32")
            throw std::runtime_error("Please provide the data type with -d <Float32|Fixed16>.");
    }

    if(options.count("i") == 0) {
        throw std::runtime_error("Please provide the input file/folder configuration with -i <File>.");
    } else {
        check_path(options["i"].as<std::string>());
    }

    if(options.count("itype") == 0) {
        throw std::runtime_error("Please provide the input type configuration with --itype <Caffe|Protobuf|Gzip>.");
    } else {
        std::string value = options["itype"].as<std::string>();
        if(value  != "Caffe" && value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Please provide the input type configuration with --itype <Caffe|Protobuf|Gzip>.");
    }

    if(options["tool"].as<std::string>() == "Transform") {
        if (options.count("o") == 0) {
            throw std::runtime_error("Please provide the output file/folder configuration with -o <File>.");
        }

        if (options.count("otype") == 0) {
            throw std::runtime_error("Please provide the output type configuration with --otype <Protobuf|Gzip>.");
        } else {
            std::string value = options["otype"].as<std::string>();
            if (value != "Protobuf" && value != "Gzip")
                throw std::runtime_error("Please provide the output type configuration with --otype <Protobuf|Gzip>.");
        }
    }
}

cxxopts::Options parse_options(int argc, char *argv[]) {
    cxxopts::Options options("DNNsim", "Deep Neural Network simulator");

    // help-related options
    options.add_options("help")("h,help", "Print this help message", cxxopts::value<bool>(), "");

    options.add_options("tools")
    ("tool", "Select the desired DNNsim",cxxopts::value<std::string>(),"<Simulator|Transform>");

    options.add_options("input")
            ("d,dtype", "Data type", cxxopts::value<std::string>(), "<Float32|Fixed16>")
            ("i,input", "Path to the input file/folder", cxxopts::value<std::string>(), "<File>")
            ("itype", "Input type", cxxopts::value<std::string>(), "<Caffe|Protobuf|Gzip>");

    options.add_options("Transform: output")
            ("o,output", "Path to the input file/folder", cxxopts::value<std::string>(), "<File>")
            ("otype", "Output type", cxxopts::value<std::string>(), "<Protobuf|Gzip>");

    options.parse_positional("tool");

    options.parse(argc, argv);

    return options;
}

int main(int argc, char *argv[]) {
    try {
        auto const options = parse_options(argc, argv);

        // Help
        if (options.count("h") == 1) {
            std::cout << options.help({"help", "tools", "input", "Transform: output"}) << std::endl;
            return 0;
        }

        check_options(options);

        std::string tool = options["tool"].as<std::string>();
        std::string data_type = options["d"].as<std::string>();

        // Separate in order to simplify simulators instantiation
        // (not all the simulator works on all the data types allowed)
        if (data_type == "Float32") {
            core::Network<float> network;
            network = read<float>(options);
            if (tool == "Transform") write<float>(network,options);
            else if (tool == "Simulator") {
                core::InferenceSimulator<float> DNNsim;
                DNNsim.run(network);
            }
        } else if (data_type == "Fixed16") {
            core::Network<uint16_t > network;
            network = read<uint16_t>(options);
            if (tool == "Transform") write<uint16_t >(network,options);
            else if (tool == "Simulator") {
                std::cout << "Under development :(" << std::endl;
            }
        }


    } catch (std::exception &exception) {
        std::cout << "Error: " << exception.what() << std::endl;
        exit(1);
    }
    return 0;
}