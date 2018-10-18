#include "cxxopts.h"
#include <iostream>
#include <core/Network.h>
#include <interface/NetReader.h>
#include <interface/NetWriter.h>

void check_path(std::string const &path)
{
    std::ifstream file(path);
    if(!file.good()) {
        throw std::runtime_error("The path " + path + " does not exist.");
    }
}

void check_options(cxxopts::Options const &options)
{
    if(options.count("n") == 0) {
        throw std::runtime_error("Please provide the network name with -o <Name>.");
    }

    if(options.count("i") == 0) {
        throw std::runtime_error("Please provide the input file/folder configuration with -i <File>.");
    } else {
        check_path(options["i"].as<std::string>());
    }

    if(options.count("itype") == 0) {
        throw std::runtime_error("Please provide the input type configuration with --itype <Trace|Protobuf|Gzip>.");
    } else {
        std::string value = options["itype"].as<std::string>();
        if(value  != "Trace" && value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Please provide the input type configuration with --itype <Trace|Protobuf|Gzip>.");
    }

    if(options.count("o") == 0) {
        throw std::runtime_error("Please provide the output file/folder configuration with -o <File>.");
    }

    if(options.count("otype") == 0) {
        throw std::runtime_error("Please provide the output type configuration with --otype <Protobuf|Gzip>.");
    } else {
        std::string value = options["otype"].as<std::string>();
        if(value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Please provide the output type configuration with --otype <Protobuf|Gzip>.");
    }

}

cxxopts::Options parse_options(int argc, char *argv[]) {
    cxxopts::Options options("DNNsim", "Deep Neural Network simulator");

    // help-related options
    options.add_options("help")("h,help", "Print this help message", cxxopts::value<bool>(), "");

    options.add_options("input")
            ("n,name", "Network name", cxxopts::value<std::string>(), "<Name>")
            ("i,input", "Path to the input file/folder", cxxopts::value<std::string>(), "<File>")
            ("itype", "Input type", cxxopts::value<std::string>(), "<Trace|Protobuf|Gzip>");

    options.add_options("Transform: output")
            ("o,output", "Path to the input file/folder", cxxopts::value<std::string>(), "<File>")
            ("otype", "Output type", cxxopts::value<std::string>(), "<Protobuf|Gzip>");

    options.parse(argc, argv);

    return options;
}

int main(int argc, char *argv[]) {
    /*  Script style:
     *  Operation mode (Simulate or transform)
     *  -n name of the network
     *  If transform:
     *      -o path to the output file, --otype output type (Protobuf, Gzip)
     *      -i path to the input file/folder, --itype input type (Protobuf, Gzip, Trace)
     *  If simulator:
     *
     */
    auto const options = parse_options(argc, argv);

    // Help
    if(options.count("h") == 1) {
        std::cout << options.help({"help", "input", "Transform: output"}) << std::endl;
        return 0;
    }

    check_options(options);

    // Read the network
    core::Network network;
    interface::NetReader reader = interface::NetReader(options["n"].as<std::string>(),options["i"].as<std::string>());
    std::string input_type = options["itype"].as<std::string>();
    if(input_type == "Trace") {
        network = reader.read_network_csv();
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);
        reader.read_output_activations_npy(network);
    } else if (input_type == "Protobuf") {
        network = reader.read_network_protobuf();
    } else {
        network = reader.read_network_gzip();
    }

    // Write network
    interface::NetWriter writer = interface::NetWriter(options["o"].as<std::string>());
    if (input_type == "Protobuf") {
        writer.write_network_protobuf(network);
    } else {
        writer.write_network_gzip(network);
    }

    return 0;
}