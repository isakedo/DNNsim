/*
Copyright (c) 2018-Present Isak Edo Vivancos, Andreas Moshovos
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

#include <sys/cxxopts.h>
#include <sys/Batch.h>

#include <base/NetReader.h>
#include <base/Network.h>

#include <core/Simulator.h>
#include <core/DRAM.h>
#include <core/GlobalBuffer.h>
#include <core/DaDianNao.h>
#include <core/Stripes.h>
#include <core/ShapeShifter.h>
#include <core/Loom.h>
#include <core/BitPragmatic.h>
#include <core/Laconic.h>
#include <core/SCNN.h>
#include <core/BitTactical.h>
#include <core/WindowFirstOutS.h>

template <typename T>
base::Network<T> read(const sys::Batch::Simulate &simulate, bool QUIET) {

    // Read the network
    base::Network<T> network;
    base::NetReader<T> reader = base::NetReader<T>(simulate.network, simulate.batch, 0, QUIET);
    if (simulate.model == "Caffe") {
        network = reader.read_network_caffe();
        network.setQuantised(simulate.quantised);
        network.setNetworkWidth(simulate.data_width);
        reader.read_precision(network);
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);

    } else if (simulate.model == "CSV") {
        network = reader.read_network_csv();
        network.setQuantised(simulate.quantised);
        network.setNetworkWidth(simulate.data_width);
        reader.read_precision(network);
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);

    } else {
		throw std::runtime_error("Input model option not recognized");
	}

    return network;

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
    ("batch", "Specify a batch file with instructions. Examples in folder \"examples\"",cxxopts::value<std::string>(),
            "<Prototxt file>");

    options.add_options("simulation")
    ("q,quiet", "Don't show stdout progress messages",cxxopts::value<bool>(),"<Boolean>")
    ("fast_mode", "Enable fast mode: simulate only one sample",cxxopts::value<bool>(),"<Boolean>")
    ("check_values", "Check the correctness of the output values of the simulations.", cxxopts::value<bool>(),
            "<Boolean>");

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

        bool QUIET = options.count("quiet") == 0 ? false : options["quiet"].as<bool>();
        bool FAST_MODE = options.count("fast_mode") == 0 ? false : options["fast_mode"].as<bool>();
        bool CHECK = options.count("check_values") == 0 ? false : options["check_values"].as<bool>();
        std::string batch_path = options["batch"].as<std::string>();
        sys::Batch batch = sys::Batch(batch_path);
        batch.read_batch();

        for(const auto &simulate : batch.getSimulations()) {

            if(!QUIET) std::cout << "Network: " << simulate.network << std::endl;

            try {

                // Inference traces
                if (simulate.data_type == "Float") {
                    auto network = read<float>(simulate, QUIET);
                    for(const auto &experiment : simulate.experiments) {

                        auto tracked_data = std::make_shared<std::map<uint64_t, uint64_t>>();
                        auto act_addresses = std::make_shared<core::AddressRange>();
                        auto wgt_addresses = std::make_shared<core::AddressRange>();

                        auto dram = std::make_shared<core::DRAM<float>>(tracked_data, act_addresses, wgt_addresses,
                                experiment.dram_size, simulate.data_width, experiment.cpu_clock_freq,
                                experiment.dram_start_act_address, experiment.dram_start_wgt_address,
                                "ini/DDR4_3200.ini", "system.ini", network.getName());

                        auto gbuffer = std::make_shared<core::GlobalBuffer<float>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.gbuffer_act_size, experiment.gbuffer_wgt_size,
                                experiment.gbuffer_act_banks, experiment.gbuffer_wgt_banks,
                                experiment.gbuffer_bank_width, experiment.gbuffer_read_delay,
                                experiment.gbuffer_write_delay);

                        auto abuffer = std::make_shared<core::LocalBuffer<float>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.abuffer_rows, experiment.abuffer_read_delay,
                                core::NULL_DELAY);

                        auto pbuffer = std::make_shared<core::LocalBuffer<float>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.pbuffer_rows, experiment.pbuffer_read_delay,
                                core::NULL_DELAY);

                        auto wbuffer = std::make_shared<core::LocalBuffer<float>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.wbuffer_rows, experiment.wbuffer_read_delay,
                                core::NULL_DELAY);

                        auto obuffer = std::make_shared<core::LocalBuffer<float>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.obuffer_rows, core::NULL_DELAY,
                                experiment.obuffer_write_delay);

                        auto composer = std::make_shared<core::Composer<float>>(experiment.composer_inputs,
                                experiment.composer_delay);

                        auto ppu = std::make_shared<core::PPU<float>>(experiment.ppu_inputs, experiment.ppu_delay);

                        auto scheduler = std::make_shared<core::BitTactical<float>>(experiment.lookahead_h,
                                experiment.lookaside_d, experiment.search_shape.c_str()[0]);

                        std::shared_ptr<core::Control<float>> control;
                        if (experiment.dataflow == "WindowFirstOutS")
                            control = std::make_shared<core::WindowFirstOutS<float>>(scheduler, dram, gbuffer, abuffer,
                                    pbuffer, wbuffer, obuffer, composer, ppu);

                        core::Simulator<float> DNNsim(FAST_MODE, QUIET, CHECK);

                        if (experiment.architecture == "SCNN") {
                            std::shared_ptr<core::Architecture<float>> arch =
                                    std::make_shared<core::SCNN<float>>(experiment.Wt, experiment.Ht, experiment.I,
                                    experiment.F, experiment.out_acc_size, experiment.banks, FAST_MODE, QUIET);

                            if (experiment.task == "Cycles")
                                std::static_pointer_cast<core::SCNN<float>>(arch)->run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network, arch);

                        } else if (experiment.architecture == "DaDianNao") {
                            std::shared_ptr<core::Architecture<float>> arch =
                                    std::make_shared<core::DaDianNao<float>>(experiment.lanes, experiment.columns,
                                    experiment.rows, experiment.tiles, experiment.pe_width,  experiment.tactical);

                            if (experiment.task == "Cycles") {
                                control->setArch(arch);
                                DNNsim.run(network, control);
                            } else if (experiment.task == "Potentials")
                                DNNsim.potentials(network, arch);
                        }
                    }

                } else if (simulate.data_type == "Fixed") {
                    base::Network<uint16_t> network;
                    {
                        base::Network<float> tmp_network;
                        tmp_network = read<float>(simulate, QUIET);
                        network = tmp_network.fixed_point();
                    }

                    for (const auto &experiment : simulate.experiments) {

                        auto tracked_data = std::make_shared<std::map<uint64_t, uint64_t>>();
                        auto act_addresses = std::make_shared<core::AddressRange>();
                        auto wgt_addresses = std::make_shared<core::AddressRange>();

                        auto dram = std::make_shared<core::DRAM<uint16_t>>(tracked_data, act_addresses, wgt_addresses,
                                experiment.dram_size, simulate.data_width, experiment.cpu_clock_freq,
                                experiment.dram_start_act_address, experiment.dram_start_wgt_address,
                                "ini/DDR4_3200.ini", "system.ini", network.getName());

                        auto gbuffer = std::make_shared<core::GlobalBuffer<uint16_t>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.gbuffer_act_size, experiment.gbuffer_wgt_size,
                                experiment.gbuffer_act_banks, experiment.gbuffer_wgt_banks,
                                experiment.gbuffer_bank_width, experiment.gbuffer_read_delay,
                                experiment.gbuffer_write_delay);

                        auto abuffer = std::make_shared<core::LocalBuffer<uint16_t>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.abuffer_rows, experiment.abuffer_read_delay,
                                core::NULL_DELAY);

                        auto pbuffer = std::make_shared<core::LocalBuffer<uint16_t>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.pbuffer_rows, experiment.pbuffer_read_delay,
                                core::NULL_DELAY);

                        auto wbuffer = std::make_shared<core::LocalBuffer<uint16_t>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.wbuffer_rows, experiment.wbuffer_read_delay,
                                core::NULL_DELAY);

                        auto obuffer = std::make_shared<core::LocalBuffer<uint16_t>>(tracked_data, act_addresses,
                                wgt_addresses, experiment.obuffer_rows, core::NULL_DELAY,
                                experiment.obuffer_write_delay);

                        auto composer = std::make_shared<core::Composer<uint16_t>>(experiment.composer_inputs,
                                experiment.composer_delay);

                        auto ppu = std::make_shared<core::PPU<uint16_t>>(experiment.ppu_inputs, experiment.ppu_delay);

                        auto scheduler = std::make_shared<core::BitTactical<uint16_t>>(experiment.lookahead_h,
                                experiment.lookaside_d, experiment.search_shape.c_str()[0]);

                        std::shared_ptr<core::Control<uint16_t>> control;
                        if (experiment.dataflow == "WindowFirstOutS")
                            control = std::make_shared<core::WindowFirstOutS<uint16_t>>(scheduler, dram, gbuffer,
                                    abuffer, pbuffer, wbuffer, obuffer, composer, ppu);

                        core::Simulator<uint16_t> DNNsim(FAST_MODE, QUIET, CHECK);

                        if (experiment.architecture == "SCNN") {
                            std::shared_ptr<core::Architecture<uint16_t>> arch =
                                    std::make_shared<core::SCNN<uint16_t>>(experiment.Wt, experiment.Ht, experiment.I,
                                    experiment.F, experiment.out_acc_size, experiment.banks, FAST_MODE, QUIET);

                            if (experiment.task == "Cycles")
                                std::static_pointer_cast<core::SCNN<uint16_t>>(arch)->run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network, arch);

                        } else if (experiment.architecture == "DaDianNao") {
                            std::shared_ptr<core::Architecture<uint16_t>> arch =
                                    std::make_shared<core::DaDianNao<uint16_t>>(experiment.lanes,
                                    experiment.columns, experiment.rows, experiment.tiles, experiment.pe_width,
                                    experiment.tactical);

                            if (experiment.task == "Cycles") {
                                control->setArch(arch);
                                DNNsim.run(network, control);
                            } else if (experiment.task == "Potentials")
                                DNNsim.potentials(network, arch);

                        } else if (experiment.architecture == "Stripes") {
                            std::shared_ptr<core::Architecture<uint16_t>> arch =
                                    std::make_shared<core::Stripes<uint16_t>>(experiment.lanes, experiment.columns,
                                    experiment.rows, experiment.tiles, experiment.pe_width);

                            if (experiment.task == "Cycles") {
                                control->setArch(arch);
                                DNNsim.run(network, control);
                            } else if (experiment.task == "Potentials")
                                DNNsim.potentials(network, arch);

                        } else if (experiment.architecture == "ShapeShifter") {
                            std::shared_ptr<core::Architecture<uint16_t>> arch =
                                    std::make_shared<core::ShapeShifter<uint16_t>>(experiment.lanes,
                                    experiment.columns, experiment.rows, experiment.tiles, experiment.pe_width,
                                    experiment.group_size, experiment.column_registers, experiment.minor_bit,
                                    experiment.diffy, experiment.tactical);

                            if (experiment.task == "Cycles") {
                                control->setArch(arch);
                                DNNsim.run(network, control);
                            } else if (experiment.task == "Potentials")
                                DNNsim.potentials(network, arch);

                        } else if (experiment.architecture == "Loom") {
                            std::shared_ptr<core::Architecture<uint16_t>> arch =
                                    std::make_shared<core::Loom<uint16_t>>(experiment.lanes, experiment.columns,
                                    experiment.rows, experiment.tiles, experiment.pe_width, experiment.group_size,
                                    experiment.pe_serial_bits, experiment.minor_bit, experiment.dynamic_weights);

                            if (experiment.task == "Cycles") {
                                control->setArch(arch);
                                DNNsim.run(network, control);
                            } else if (experiment.task == "Potentials")
                                DNNsim.potentials(network, arch);

                        } else if (experiment.architecture == "BitPragmatic") {
                            std::shared_ptr<core::Architecture<uint16_t>> arch =
                                    std::make_shared<core::BitPragmatic<uint16_t>>(experiment.lanes,
                                    experiment.columns, experiment.rows, experiment.tiles, experiment.pe_width,
                                    experiment.bits_first_stage, experiment.column_registers, experiment.booth,
                                    experiment.diffy, experiment.tactical);

                            if (experiment.task == "Cycles") {
                                control->setArch(arch);
                                DNNsim.run(network, control);
                            } else if (experiment.task == "Potentials")
                                DNNsim.potentials(network, arch);

                        } else if (experiment.architecture == "Laconic") {
                            std::shared_ptr<core::Architecture<uint16_t>> arch =
                                    std::make_shared<core::Laconic<uint16_t>>(experiment.lanes, experiment.columns,
                                    experiment.rows, experiment.tiles, experiment.pe_width, experiment.booth);

                            if (experiment.task == "Cycles") {
                                control->setArch(arch);
                                DNNsim.run(network, control);
                            } else if (experiment.task == "Potentials")
                                DNNsim.potentials(network, arch);

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

    } catch (std::exception &exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        exit(1);
    }
    return 0;
}
