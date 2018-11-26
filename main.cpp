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
#include <core/BitPragmatic.h>
#include <core/Laconic.h>
#include <interface/StatsWriter.h>

template <typename T>
core::Network<T> read(const cxxopts::Options &options) {
    core::Network<T> network;
    const auto &input_type = options["itype"].as<std::string>();
    const auto &network_name = options["n"].as<std::string>();

    // Read the network
    interface::NetReader<T> reader = interface::NetReader<T>(network_name);
    if (input_type == "Caffe") {
        network = reader.read_network_caffe();
        reader.read_precision(network);
        reader.read_weights_npy(network);

        #ifdef BIAS
        reader.read_bias_npy(network);
        #endif

        reader.read_activations_npy(network);

        #ifdef OUTPUT_ACTIVATIONS
        reader.read_output_activations_npy(network);
        #endif

    } else if (input_type == "Protobuf") {
        network = reader.read_network_protobuf();
    } else {
        network = reader.read_network_gzip();
    }

    return network;

}

template <typename T>
void write(const core::Network<T> &network, const cxxopts::Options &options, const std::string &data_conversion) {
    // Write network
    const auto &network_name = options["n"].as<std::string>();
    interface::NetWriter<T> writer = interface::NetWriter<T>(network_name,data_conversion);
    std::string output_type = options["otype"].as<std::string>();
    if (output_type == "Protobuf") {
        writer.write_network_protobuf(network);
    } else {
        writer.write_network_gzip(network);
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

    if(options.count("n") == 0) {
        throw std::runtime_error("Please provide the name configuration with -n <Name>.");
    }

    if(options.count("ditype") == 0) {
        throw std::runtime_error("Please provide the data input type with --ditype <Float32|Fixed16>.");
    } else {
        std::string value = options["ditype"].as<std::string>();
        if(value  != "Float32" && value != "Fixed16")
            throw std::runtime_error("Please provide the data input type with --ditype <Float32|Fixed16>.");
    }

    if(options.count("itype") == 0) {
        throw std::runtime_error("Please provide the input type configuration with --itype <Caffe|Protobuf|Gzip>.");
    } else {
        std::string value = options["itype"].as<std::string>();
        if(value  != "Caffe" && value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Please provide the input type configuration with --itype <Caffe|Protobuf|Gzip>.");
    }

    if(options["tool"].as<std::string>() == "Transform") {

        //Optional
        std::string value = options["dotype"].as<std::string>();
        if(options.count("dotype") == 1 && value  != "Fixed16") {
            throw std::runtime_error("Please provide a correct data output type with --dotype <Fixed16>.");
        }

        if (options.count("otype") == 0) {
            throw std::runtime_error("Please provide the output type configuration with --otype <Protobuf|Gzip>.");
        } else {
            std::string value2 = options["otype"].as<std::string>();
            if (value2 != "Protobuf" && value2 != "Gzip")
                throw std::runtime_error("Please provide the output type configuration with --otype <Protobuf|Gzip>.");
        }

    } else if(options["tool"].as<std::string>() == "Simulator") {

        if(options["ditype"].as<std::string>() == "Fixed16") {
            if (options.count("a") == 0) {
                throw std::runtime_error("Please provide the architecture to simulate with -a <Laconic|BitPragmatic>.");
            } else {
                std::string value = options["a"].as<std::string>();
                if (value != "Laconic" && value != "BitPragmatic")
                    throw std::runtime_error(
                            "Please provide the architecture to simulate with -a <Laconic|BitPragmatic>.");
            }
        }

        /*if (options.count("o") == 0) {
            throw std::runtime_error("Please provide the output file/folder configuration with -o <File>.");
        }

        if (options.count("otype") == 0) {
            throw std::runtime_error("Please provide the output statistics file type with --otype <Text|csv>.");
        } else {
            std::string value = options["otype"].as<std::string>();
            if (value2 != "Text" && value != "csv")
                throw std::runtime_error("Please provide the output statistics file type with --otype <Text|csv>.");
        }*/

    }

}

cxxopts::Options parse_options(int argc, char *argv[]) {
    cxxopts::Options options("DNNsim", "Deep Neural Network simulator");

    // help-related options
    options.add_options("help")("h,help", "Print this help message", cxxopts::value<bool>(), "");

    options.add_options("tools")
    ("tool", "Select the desired DNNsim function",cxxopts::value<std::string>(),"<Simulator|Transform>");

    options.add_options("input")
            ("ditype", "Data input type", cxxopts::value<std::string>(), "<Float32|Fixed16>")
            ("n,name", "Name of the network", cxxopts::value<std::string>(), "<Name>")
            ("itype", "Input type", cxxopts::value<std::string>(), "<Caffe|Protobuf|Gzip>");

    options.add_options("Transform: output")
            ("dotype", "Data output type (optional)", cxxopts::value<std::string>(), "<Fixed16>")
            ("otype", "Output type", cxxopts::value<std::string>(), "<Protobuf|Gzip>");

    options.add_options("Simulator: output")
            ("a,arch", "Architecture to simulate (Only fixed point)", cxxopts::value<std::string>(),
                    "<Laconic|BitPragmatic>");
         //   ("o,output", "Path to the input file/folder", cxxopts::value<std::string>(), "<File>")
         //   ("otype", "Output type", cxxopts::value<std::string>(), "<Text|csv>");

    options.parse_positional("tool");

    options.parse(argc, argv);

    return options;
}

int main(int argc, char *argv[]) {
    try {
        auto const options = parse_options(argc, argv);

        // Help
        if (options.count("h") == 1) {
            std::cout << options.help({"help", "tools", "input", "Transform: output", "Simulator: output"})
                << std::endl;
            return 0;
        }

        check_options(options);

        std::string tool = options["tool"].as<std::string>();
        std::string data_type = options["ditype"].as<std::string>();

        // Separate in order to simplify simulators instantiation
        // (not all the simulator works on all the data types allowed)
        if (data_type == "Float32") {

            core::Network<float> network;
            network = read<float>(options);

            if (tool == "Transform") {
                std::string data_conversion = options["dotype"].count() == 0 ?
                        "Not" : options["dotype"].as<std::string>();
                write<float>(network,options,data_conversion);
            }

            else if (tool == "Simulator") {
                #if defined(OUTPUT_ACTIVATIONS) || defined(BIAS) //Need both to perform and check the inference
                core::InferenceSimulator<float> DNNsim;
                DNNsim.run(network);
                #endif
            }

        } else if (data_type == "Fixed16") {
            core::Network<uint16_t> network;
            network = read<uint16_t>(options);

            // Not converting from fixed point to other values
            if (tool == "Transform") write<uint16_t>(network,options,"Not");

            else if (tool == "Simulator") {
                std::string architecture = options["a"].as<std::string>();
                if(architecture == "BitPragmatic") {
                    core::BitPragmatic<uint16_t> DNNsim;
                    //DNNsim.run(network);
                    DNNsim.memoryAccesses(network);
                } else if (architecture == "Laconic") {
                    core::Laconic<uint16_t> DNNsim;
                    //DNNsim.run(network);
                    DNNsim.workReduction(network);
                }

                //Dump statistics
                //std::string dump_type = options["otype"].as<std::string>();
                //if(dump_type == "Text") writer.dump_txt();
                /*else if (dump_type == "csv")*/
                interface::StatsWriter::dump_csv();

            }

        }

    } catch (std::exception &exception) {
        std::cout << "Error: " << exception.what() << std::endl;
        exit(1);
    }
    return 0;
}