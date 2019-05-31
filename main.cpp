/*
Copyright (c) 2018-ETERNITY Isak Edo Vivancos, Andreas Moshovos
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
#include <core/Inference.h>
#include <core/Stripes.h>
#include <core/DynamicStripes.h>
#include <core/DynamicStripesFP.h>
#include <core/Loom.h>
#include <core/BitPragmatic.h>
#include <core/Laconic.h>
#include <core/BitTacticalP.h>
#include <core/BitTacticalE.h>
#include <core/SCNN.h>
#include <core/SCNNp.h>
#include <core/SCNNe.h>
#include <core/BitFusion.h>

template <typename T>
core::Network<T> read_training(const std::string &network_name, int batch, int epoch, int decoder_states,
        int traces_mode) {

    // Read the network
    core::Network<T> network;
    interface::NetReader<T> reader = interface::NetReader<T>(network_name, false, batch, epoch, false);
	network = reader.read_network_trace_params();
	if(decoder_states > 0) network.duplicate_decoder_layers(decoder_states);

	bool forward = (traces_mode & 0x1) != 0;
	bool backward = (traces_mode & 0x2) != 0;
	network.setForkward(forward);
	network.setBackward(backward);

	// Forward traces
	if(forward) {
		reader.read_training_weights_npy(network);
		reader.read_training_activations_npy(network);
		reader.read_training_bias_npy(network);
	}

	// Backward traces
	if(backward) {
		reader.read_training_weight_gradients_npy(network);
		reader.read_training_activation_gradients_npy(network);
		reader.read_training_bias_gradients_npy(network);
		reader.read_training_output_activation_gradients_npy(network);
	}
    return network;

}

template <typename T>
core::Network<T> read(const std::string &input_type, const std::string &network_name, bool bias_and_out_act,
        int batch, bool tensorflow_8b) {

    // Read the network
    core::Network<T> network;
    interface::NetReader<T> reader = interface::NetReader<T>(network_name, bias_and_out_act, batch,0,
            tensorflow_8b);
    if (input_type == "Caffe") {
        network = reader.read_network_caffe();
        reader.read_precision(network);
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);
        if(bias_and_out_act) {
            reader.read_bias_npy(network);
            reader.read_output_activations_npy(network);
        }
    } else if (input_type == "Trace") {
        network = reader.read_network_trace_params();
        reader.read_precision(network);
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);
        if(bias_and_out_act) {
            reader.read_bias_npy(network);
            reader.read_output_activations_npy(network);
        }
    } else if (input_type == "CParams") {
        network = reader.read_network_conv_params();
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);
        if(bias_and_out_act) {
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
        bool bias_and_out_act, bool OVERWRITE, bool tensorflow_8b) {

    // Write network
    interface::NetWriter<T> writer = interface::NetWriter<T>(network.getName(),data_conversion,
            bias_and_out_act,OVERWRITE,tensorflow_8b);
    if (output_type == "Protobuf") {
        writer.write_network_protobuf(network);
    } else {
        writer.write_network_gzip(network);
    }
}

template <typename T>
std::vector<schedule> read_schedule(const std::string &network_name, const std::string &arch,
        const sys::Batch::Simulate::Experiment &experiment) {

    interface::NetReader<T> reader = interface::NetReader<T>(network_name, false,0,0,false);
    int mux_entries = experiment.lookahead_h + experiment.lookaside_d + 1;
    std::string schedule_type = arch + "_" + experiment.search_shape + std::to_string(mux_entries) + "("
            + std::to_string(experiment.lookahead_h) + "-" + std::to_string(experiment.lookaside_d) + ")";
    return reader.read_schedule_protobuf(schedule_type);
}

template <typename T>
void write_schedule(const core::Network<T> &network, core::BitTactical<T> &DNNsim, const std::string &arch,
        const sys::Batch::Simulate::Experiment &experiment, bool OVERWRITE) {
    const auto &network_schedule = DNNsim.network_scheduler(network);
    interface::NetWriter<uint16_t> writer = interface::NetWriter<uint16_t>(network.getName(),"Not", false, OVERWRITE,
            false);
    int mux_entries = experiment.lookahead_h + experiment.lookaside_d + 1;
    std::string schedule_type = arch + "_" + experiment.search_shape + std::to_string(mux_entries) + "("
            + std::to_string(experiment.lookahead_h) + "-" + std::to_string(experiment.lookaside_d) + ")";
    writer.write_schedule_protobuf(network_schedule,schedule_type);
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

    if(options.count("threads") == 1 && options["threads"].as<uint16_t>() < 1) {
        throw std::runtime_error("The number of threads must be at least one.");
    }

}

cxxopts::Options parse_options(int argc, char *argv[]) {
    cxxopts::Options options("DNNsim", "Deep Neural Network simulator");

    // help-related options
    options.add_options("help")("h,help", "Print this help message", cxxopts::value<bool>(), "");

    options.add_options("batch")
    ("batch", "Specify a batch file with instructions. Examples in folder \"examples\"",cxxopts::value<std::string>(),
            "<Prototxt file>");

    options.add_options("simulation")
    ("t,threads", "Specify the number of threads",cxxopts::value<uint16_t>(),"<Positive number>")
    ("fast_mode", "Enable fast mode: simulate only one image",cxxopts::value<bool>(),"<Boolean>")
    ("overwrite", "Overwrite the intermediate files (Protobuf,Gzip,Schedule)",cxxopts::value<bool>(),"<Boolean>");

    options.parse_positional("batch");

    options.parse(argc, argv);

    return options;
}

int main(int argc, char *argv[]) {
    try {
        auto const options = parse_options(argc, argv);

        // Help
        if (options.count("h") == 1) {
            std::cout << options.help({"help", "batch", "simulation"}) << std::endl;
            return 0;
        }

        check_options(options);

        uint8_t N_THREADS = options.count("threads") == 0 ? (uint8_t)1 : (uint8_t)options["threads"].as<uint16_t>();
        bool FAST_MODE = options.count("fast_mode") == 0 ? false : options["fast_mode"].as<bool>();
        bool OVERWRITE = options.count("overwrite") == 0 ? false : options["overwrite"].as<bool>();
        std::string batch_path = options["batch"].as<std::string>();
        sys::Batch batch = sys::Batch(batch_path);
        batch.read_batch();

        // Do transformation first
        for(const auto &transform : batch.getTransformations()) {
            try {
                if (transform.inputDataType == "Float32") {
                    core::Network<float> network;
                    network = read<float>(transform.inputType, transform.network, transform.bias_and_out_act,
                            transform.batch, transform.tensorflow_8b);
                    write<float>(network, transform.outputType, transform.outputDataType,
                            transform.bias_and_out_act, OVERWRITE, transform.tensorflow_8b);
                } else if (transform.inputDataType == "Fixed16") {
                    core::Network<uint16_t> network;
                    network = read<uint16_t>(transform.inputType, transform.network, transform.bias_and_out_act,
                            transform.batch, transform.tensorflow_8b);
                    write<uint16_t>(network, transform.outputType, transform.outputDataType,
                            transform.bias_and_out_act, OVERWRITE, transform.tensorflow_8b);
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
				// Training traces
				if(simulate.training) {

					int traces_mode = 0;
					if(simulate.only_forward) traces_mode = 1;
					else if(simulate.only_backward) traces_mode = 2;
					else traces_mode = 3;
						
					int epochs = simulate.epochs;
					for(const auto &experiment : simulate.experiments) {

						// Initialize statistics
						sys::Statistics::Stats stats;
						sys::Statistics::initialize(stats);

						for (int epoch = 0; epoch < epochs; epoch++) {
							core::Network<float> network;
							network = read_training<float>(simulate.network, simulate.batch, epoch,
							        simulate.decoder_states, traces_mode);

				        	if(experiment.architecture == "None") {
				        		core::Simulator<float> DNNsim(N_THREADS,FAST_MODE);
				                if (experiment.task == "Sparsity") DNNsim.training_sparsity(network,stats,epoch,
				                        epochs);
				                else if (experiment.task == "ExpBitSparsity") DNNsim.training_bit_sparsity(network,
				                        stats, epoch, epochs, false);
				                else if (experiment.task == "MantBitSparsity") DNNsim.training_bit_sparsity(network,
				                        stats, epoch, epochs, true);
				                else if (experiment.task == "ExpDistr") DNNsim.training_distribution(network, stats,
				                        epoch, epochs, false);
				                else if (experiment.task == "MantDistr") DNNsim.training_distribution(network, stats,
								        epoch, epochs, true);
				            } else if (experiment.architecture == "DynamicStripesFP") {
								core::DynamicStripesFP<float> DNNsim(experiment.leading_bit,experiment.minor_bit,
                                        N_THREADS,FAST_MODE);
				                if (experiment.task == "AvgWidth") DNNsim.average_width(network,stats,epoch,epochs);
				            }
						}

						sys::Statistics::addStats(stats);

		            }

				// Inference traces
				} else {

		            if (simulate.inputDataType == "Float32") {
		                core::Network<float> network;
		                network = read<float>(simulate.inputType, simulate.network, simulate.bias_and_out_act,
		                        simulate.batch, simulate.tensorflow_8b);
		                network.setNetwork_bits(simulate.network_bits);
		                for(const auto &experiment : simulate.experiments) {
		                    if(experiment.architecture == "None") {
		                        if(experiment.task == "Inference") {
		                            core::Inference<float> DNNsim(N_THREADS,FAST_MODE);
		                            DNNsim.run(network);
		                        } else {
		                            core::Simulator<float> DNNsim(N_THREADS,FAST_MODE);
		                            if (experiment.task == "Sparsity") DNNsim.sparsity(network);
		                        }
		                    } else if (experiment.architecture == "SCNN") {
		                        core::SCNN<float> DNNsim(experiment.Wt, experiment.Ht, experiment.I, experiment.F,
		                                experiment.out_acc_size, experiment.banks, N_THREADS, FAST_MODE);
		                        if (experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);
		                    }
		                }
		            } else if (simulate.inputDataType == "Fixed16") {
		                core::Network<uint16_t> network;
		                network = read<uint16_t>(simulate.inputType, simulate.network, simulate.bias_and_out_act,
		                        simulate.batch, simulate.tensorflow_8b);
		                network.setNetwork_bits(simulate.network_bits);
		                for(const auto &experiment : simulate.experiments) {
		                    if(experiment.architecture == "None") {
		                        core::Simulator<uint16_t> DNNsim(N_THREADS,FAST_MODE);
		                        if(experiment.task == "Sparsity") DNNsim.sparsity(network);
		                        else if(experiment.task == "BitSparsity") DNNsim.bit_sparsity(network);

		                    } else if(experiment.architecture == "BitPragmatic") {
		                        core::BitPragmatic<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
		                                experiment.bits_first_stage,experiment.column_registers,experiment.diffy,
		                                N_THREADS,FAST_MODE);
		                        if(experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if(experiment.architecture == "Stripes") {
		                        core::Stripes<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,experiment.bits_pe,
		                                N_THREADS,FAST_MODE);
		                        if(experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if(experiment.architecture == "DynamicStripes") {
		                        core::DynamicStripes<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
		                                experiment.precision_granularity, experiment.column_registers,
		                                experiment.bits_pe, experiment.leading_bit, experiment.diffy, 
                                        N_THREADS,FAST_MODE);
		                        if(experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);
		                        else if (experiment.task == "AvgWidth") DNNsim.average_width(network);

		                    } else if(experiment.architecture == "Loom") {
		                        core::Loom<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
		                                experiment.precision_granularity, experiment.pe_serial_bits,
		                                experiment.leading_bit, experiment.dynamic_weights, N_THREADS,FAST_MODE);
		                        if(experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if (experiment.architecture == "Laconic") {
		                        core::Laconic<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,N_THREADS,
									FAST_MODE);
		                        if(experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if (experiment.architecture == "BitTacticalP") {
		                        core::BitTacticalP<uint16_t> DNNsim(experiment.n_columns, experiment.n_rows,
		                                experiment.precision_granularity, experiment.column_registers,
		                                experiment.lookahead_h, experiment.lookaside_d, experiment.search_shape,
		                                experiment.leading_bit, N_THREADS, FAST_MODE);
		                        if(experiment.task == "Cycles" && experiment.read_schedule_from_proto) {
		                            auto dense_schedule = read_schedule<uint16_t>(network.getName(),"BitTactical",
		                                    experiment);
		                            DNNsim.run(network, dense_schedule);
		                        }
		                        else if (experiment.task == "Schedule")
		                            write_schedule<uint16_t>(network,DNNsim,"BitTactical",experiment,OVERWRITE);
		                        else if (experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if (experiment.architecture == "BitTacticalE") {
		                        core::BitTacticalE<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
		                                experiment.bits_first_stage, experiment.column_registers, experiment.lookahead_h,
		                                experiment.lookaside_d, experiment.search_shape,N_THREADS,FAST_MODE);
		                        if(experiment.task == "Cycles" && experiment.read_schedule_from_proto) {
		                            auto dense_schedule = read_schedule<uint16_t>(network.getName(),"BitTactical",
		                                    experiment);
		                            DNNsim.run(network, dense_schedule);
		                        }
		                        else if (experiment.task == "Schedule")
		                            write_schedule<uint16_t>(network,DNNsim,"BitTactical",experiment,OVERWRITE);
		                        else if (experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if (experiment.architecture == "SCNN") {
		                        core::SCNN<uint16_t> DNNsim(experiment.Wt, experiment.Ht, experiment.I, experiment.F,
		                                experiment.out_acc_size, experiment.banks, N_THREADS, FAST_MODE);
		                        if (experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if (experiment.architecture == "SCNNp") {
		                        core::SCNNp<uint16_t> DNNsim(experiment.Wt, experiment.Ht, experiment.I, experiment.F,
		                                experiment.out_acc_size, experiment.banks, experiment.pe_serial_bits,
		                                N_THREADS, FAST_MODE);
		                        if (experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);

		                    } else if (experiment.architecture == "SCNNe") {
		                        core::SCNNe<uint16_t> DNNsim(experiment.Wt, experiment.Ht, experiment.I, experiment.F,
		                                experiment.out_acc_size, experiment.banks, experiment.pe_serial_bits, N_THREADS,
		                                FAST_MODE);
		                        if (experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);
		                    } else if (experiment.architecture == "BitFusion") {
		                        core::BitFusion<uint16_t> DNNsim(experiment.M, experiment.N, experiment.pmax,
		                                experiment.pmin, N_THREADS, FAST_MODE);
		                        if (experiment.task == "Cycles") DNNsim.run(network);
		                        else if (experiment.task == "Potentials") DNNsim.potentials(network);
		                    }
		                    sys::Statistics::updateFlagsLastStat(simulate.tensorflow_8b);
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
