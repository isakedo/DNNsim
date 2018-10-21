#include "cxxopts.h"
#include <iostream>
#include <core/Network.h>
#include <interface/NetReader.h>
#include <interface/NetWriter.h>
#include <core/Simulator.h>

void check_path(std::string const &path)
{
    std::ifstream file(path);
    if(!file.good()) {
        throw std::runtime_error("The path " + path + " does not exist.");
    }
}

void check_options(cxxopts::Options const &options)
{
    if(options.count("tool") == 0) {
        throw std::runtime_error("Please provide first the desired tool <Simulator|Transform>.");
    } else {
        std::string value = options["tool"].as<std::string>();
        if(value  != "Simulator" && value != "Transform")
            throw std::runtime_error("Please provide first the desired tool <Simulator|Transform>.");
    }

    if(options.count("n") == 0) {
        throw std::runtime_error("Please provide the network name with -n <Name>.");
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
            ("n,name", "Network name", cxxopts::value<std::string>(), "<Name>")
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
    /*  Script style:
     *  Operation mode (Simulate or transform)
     *  -n name of the network
     *  If transform:
     *      -o path to the output file, --otype output type (Protobuf, Gzip)
     *      -i path to the input file/folder, --itype input type (Protobuf, Gzip, Trace)
     *  If simulator:
     *
     */
    try {
        auto const options = parse_options(argc, argv);

        // Help
        if (options.count("h") == 1) {
            std::cout << options.help({"help", "tools", "input", "Transform: output"}) << std::endl;
            return 0;
        }

        check_options(options);
        std::string tool = options["tool"].as<std::string>();

        // Read the network
        core::Network network;
        interface::NetReader reader = interface::NetReader(options["n"].as<std::string>(),
                                                           options["i"].as<std::string>());
        std::string input_type = options["itype"].as<std::string>();
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

        if (tool == "Transform") {
            // Write network
            interface::NetWriter writer = interface::NetWriter(options["o"].as<std::string>());
            std::string output_type = options["otype"].as<std::string>();
            if (output_type == "Protobuf") {
                writer.write_network_protobuf(network);
            } else {
                writer.write_network_gzip(network);
            }
        } else if (tool == "Simulator") {
            //Simulation
            core::Simulator DNNsim;
            DNNsim.run(network);
        }
    } catch(std::exception &exception) {
        std::cout << "Error: " << exception.what() << std::endl;
        exit(1);
    }
    return 0;
}