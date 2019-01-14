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
#include <sys/Batch.h>

#include <interface/NetReader.h>
#include <interface/NetWriter.h>
#include <interface/StatsWriter.h>

#include <core/Network.h>
#include <core/InferenceSimulator.h>
#include <core/Stripes.h>
#include <core/DynamicStripes.h>
#include <core/BitPragmatic.h>
#include <core/Laconic.h>
#include <core/BitTacticalE.h>
#include <core/BitTacticalP.h>
#include <core/BitFusion.h>

template <typename T>
core::Network<T> read(const std::string &input_type, const std::string &network_name, bool activate_bias_and_out_act) {

    // Read the network
    core::Network<T> network;
    interface::NetReader<T> reader = interface::NetReader<T>(network_name, activate_bias_and_out_act);
    if (input_type == "Caffe") {
        network = reader.read_network_caffe();
        reader.read_precision(network);
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);
        if(activate_bias_and_out_act) {
            reader.read_bias_npy(network);
            reader.read_output_activations_npy(network);
        }
    } else if (input_type == "Protobuf") {
        network = reader.read_network_protobuf();
    } else {
        network = reader.read_network_gzip();
    }

    return network;

}

template <typename T>
void write(const core::Network<T> &network, const std::string &output_type, const std::string &data_conversion,
        bool activate_bias_and_out_act) {

    // Write network
    interface::NetWriter<T> writer = interface::NetWriter<T>(network.getName(),data_conversion,
            activate_bias_and_out_act);
    if (output_type == "Protobuf") {
        writer.write_network_protobuf(network);
    } else {
        writer.write_network_gzip(network);
    }
}

void check_options(const cxxopts::Options &options)
{
    if(options.count("batch") == 0) {
        throw std::runtime_error("Please provide a batch file with instructions. Examples in folder \"examples\"");
    } else {
        std::string batch_path = options["batch"].as<std::string>();
        std::ifstream file(batch_path);
        if(!file.good()) {
            throw std::runtime_error("The path " + batch_path + " does not exist.");
        }
    }
}

cxxopts::Options parse_options(int argc, char *argv[]) {
    cxxopts::Options options("DNNsim", "Deep Neural Network simulator");

    // help-related options
    options.add_options("help")("h,help", "Print this help message", cxxopts::value<bool>(), "");

    options.add_options("batch")
    ("batch", "Specify a batch file with intrusctions. Examples in folder \"examples\"",cxxopts::value<std::string>(),
            "<Prototxt file>");

    options.parse_positional("batch");

    options.parse(argc, argv);

    return options;
}

int main(int argc, char *argv[]) {
    try {
        auto const options = parse_options(argc, argv);

        // Help
        if (options.count("h") == 1) {
            std::cout << options.help({"help", "batch"}) << std::endl;
            return 0;
        }

        check_options(options);

        std::string batch_path = options["batch"].as<std::string>();
        sys::Batch batch = sys::Batch(batch_path);
        batch.read_batch();

        // Do transformation first
        for(const auto &transform : batch.getTransformations()) {
            try {
                if (transform.inputDataType == "Float32") {
                    core::Network<float> network;
                    network = read<float>(transform.inputType, transform.network, transform.activate_bias_out_act);
                    write<float>(network, transform.outputType, transform.outputDataType,
                            transform.activate_bias_out_act);
                } else if (transform.inputDataType == "Fixed16") {
                    core::Network<uint16_t> network;
                    network = read<uint16_t>(transform.inputType, transform.network, transform.activate_bias_out_act);
                    write<uint16_t>(network, transform.outputType, transform.outputDataType,
                            transform.activate_bias_out_act);
                }
            } catch (std::exception &exception) {
                std::cerr << "Transformation error: " << exception.what() << std::endl;
                #ifdef STOP_AFTER_ERROR
                exit(1);
                #endif
            }
        }

        for(const auto &simulate : batch.getSimulations()) {
            try {
                if (simulate.inputDataType == "Float32") {
                    core::Network<float> network;
                    network = read<float>(simulate.inputType, simulate.network, simulate.activate_bias_out_act);
                    core::InferenceSimulator<float> DNNsim;
                    DNNsim.run(network);
                } else if (simulate.inputDataType == "Fixed16") {
                    core::Network<uint16_t> network;
                    network = read<uint16_t>(simulate.inputType, simulate.network, simulate.activate_bias_out_act);
                    for(const auto &experiment : simulate.experiments) {
                        if(experiment.architecture == "BitPragmatic") {
                            core::BitPragmatic<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.bits_first_stage);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);
                            else if (experiment.task == "MemAccesses") DNNsim.memoryAccesses(network);

                        } else if(experiment.architecture == "Stripes") {
                            core::Stripes<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);
                            else if (experiment.task == "MemAccesses") DNNsim.memoryAccesses(network);

                        } else if(experiment.architecture == "DynamicStripes") {
                            core::DynamicStripes<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.precision_granularity);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);
                            else if (experiment.task == "MemAccesses") DNNsim.memoryAccesses(network);

                        } else if (experiment.architecture == "Laconic") {
                            core::Laconic<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "BitTacticalP") {
                            core::BitTacticalP<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.lookahead_h, experiment.lookaside_d, experiment.search_shape,
                                    experiment.precision_granularity);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "BitTacticalE") {
                            core::BitTacticalE<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.lookahead_h, experiment.lookaside_d, experiment.search_shape,
                                    experiment.bits_first_stage);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "BitFusion") {
                            core::BitFusion<uint16_t> DNNsim;
                            if(experiment.task == "Cycles") DNNsim.run(network);
                        }
                    }
                }
            } catch (std::exception &exception) {
                std::cerr << "Simulation error: " << exception.what() << std::endl;
                #ifdef STOP_AFTER_ERROR
                exit(1);
                #endif
            }
        }

        //Dump statistics
        interface::StatsWriter::dump_csv();

    } catch (std::exception &exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        exit(1);
    }
    return 0;
}