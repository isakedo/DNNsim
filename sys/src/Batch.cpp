
#include <sys/Batch.h>
#include <regex>

namespace sys {

    bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto) {
        int fd = open(filename, O_RDONLY);
        auto input = new google::protobuf::io::FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }

    uint64_t parse_frequency(const std::string &freq) {

        if (std::regex_match(freq, std::regex("0*(GHz|MHz|KHz|Hz)$")))
            return 0;

        int base, power;

        if (std::regex_match(freq, std::regex("[0-9]+GHz$")))
            base = 10, power = 9;
        else if (std::regex_match(freq, std::regex("[0-9]+MHz$")))
            base = 10, power = 6;
        else if (std::regex_match(freq, std::regex("[0-9]+KHz$")))
            base = 10, power = 3;
        else if (std::regex_match(freq, std::regex("[0-9]+Hz$")))
            base = 10, power = 0;
        else
            throw std::exception();

        auto npos = freq.find_first_of("GMKBi");
        return std::stoul(freq.substr(0, npos)) * (uint64_t)pow(base, power);

    }

    uint64_t parse_memory_size(const std::string &size) {

        if (std::regex_match(size, std::regex("0*(GB|GiB|MB|MiB|KB|KiB|B)$")))
            return 0;

        int base, power;

        if (std::regex_match(size, std::regex("[0-9]+GB$")))
            base = 10, power = 9;
        else if (std::regex_match(size, std::regex("[0-9]+GiB$")))
            base = 2, power = 30;
        else if (std::regex_match(size, std::regex("[0-9]+MB$")))
            base = 10, power = 6;
        else if (std::regex_match(size, std::regex("[0-9]+MiB$")))
            base = 2, power = 20;
        else if (std::regex_match(size, std::regex("[0-9]+KB$")))
            base = 10, power = 3;
        else if (std::regex_match(size, std::regex("[0-9]+KiB$")))
            base = 2, power = 10;
        else if (std::regex_match(size, std::regex("[0-9]+B$")))
            base = 10, power = 0;
        else
            throw std::exception();

        auto npos = size.find_first_of("GMKBi");
        return std::stoul(size.substr(0, npos)) * (uint64_t)pow(base, power);

    }

    Batch::Simulate Batch::read_inference_simulation(const protobuf::Batch_Simulate &simulate_proto) {

        Batch::Simulate simulate;
        simulate.network = simulate_proto.network();
        simulate.batch = simulate_proto.batch();

        const auto &model = simulate_proto.model();
        if(model  != "Caffe" && model != "CSV")
            throw std::runtime_error("Model configuration must be <Caffe|CSV>.");
        else
            simulate.model = simulate_proto.model();

        const auto &dtype = simulate_proto.data_type();
        if(dtype  != "Float" && dtype != "Fixed")
            throw std::runtime_error("Input data type configuration must be <Float|Fixed>.");
        else
            simulate.data_type = simulate_proto.data_type();

        if (dtype == "Float") simulate.data_width = 32;
        else simulate.data_width = simulate_proto.data_width() < 1 ? 16 : simulate_proto.data_width();

        if (simulate.data_width > 16)
            throw std::runtime_error("Maximum data width allowed for Fixed data type is 16");

        for(const auto &experiment_proto : simulate_proto.experiment()) {

            Batch::Simulate::Experiment experiment;

            // Core parameters
            try {
                if (experiment_proto.cpu_clock_freq().empty()) experiment.cpu_clock_freq = (uint64_t)pow(10, 9);
                else experiment.cpu_clock_freq = parse_frequency(experiment_proto.cpu_clock_freq());
            } catch (const std::exception &e) {
                throw std::runtime_error("Core frequency not recognised.");
            }

            // Memory parameters
            try {
                if (experiment_proto.dram_size().empty()) experiment.dram_size = (uint64_t)pow(2, 14);
                else {
                    experiment.dram_size = parse_memory_size(experiment_proto.dram_size());
                    if (experiment.dram_size < pow(2, 20)) experiment.dram_size = 1;
                    else experiment.dram_size /= pow(2, 20);
                }
            } catch (const std::exception &e) {
                throw std::runtime_error("DRAM size not recognised.");
            }

            experiment.dram_start_act_address = experiment_proto.dram_start_act_address() < 1 ? 0x80000000 :
                    experiment_proto.dram_start_act_address();
            experiment.dram_start_wgt_address = experiment_proto.dram_start_wgt_address() < 1 ? 0x00000000 :
                    experiment_proto.dram_start_wgt_address();

            auto dram_range = log2(experiment.dram_size * pow(2, 20));

            auto dram_act_addr_range = log2(experiment.dram_start_act_address);
            if (dram_range <= dram_act_addr_range)
                throw std::runtime_error("DRAM start activation addresses out of range.");

            auto dram_wgt_addr_range = log2(experiment.dram_start_wgt_address);
            if (dram_range <= dram_wgt_addr_range)
                throw std::runtime_error("DRAM start weight addresses out of range.");

            try {
                if (experiment_proto.gbuffer_act_size().empty())
                    experiment.gbuffer_act_size = (uint64_t)pow(10, 9);
                else experiment.gbuffer_act_size = parse_memory_size(experiment_proto.gbuffer_act_size());
            } catch (const std::exception &e) {
                throw std::runtime_error("Global Buffer activation size not recognised.");
            }

            try {
                if (experiment_proto.gbuffer_wgt_size().empty())
                    experiment.gbuffer_wgt_size = (uint64_t)pow(10, 9);
                else experiment.gbuffer_wgt_size = parse_memory_size(experiment_proto.gbuffer_wgt_size());
            } catch (const std::exception &e) {
                throw std::runtime_error("Global Buffer weights size not recognised.");
            }

            experiment.gbuffer_act_banks = experiment_proto.gbuffer_act_banks() < 1 ? 16 :
                    experiment_proto.gbuffer_act_banks();
            experiment.gbuffer_wgt_banks = experiment_proto.gbuffer_wgt_banks() < 1 ? 256 :
                    experiment_proto.gbuffer_wgt_banks();
            experiment.gbuffer_out_banks = experiment_proto.gbuffer_out_banks() < 1 ? 16 :
                    experiment_proto.gbuffer_out_banks();
            experiment.gbuffer_bank_width = experiment_proto.gbuffer_bank_width() < 1 ? 256 :
                    experiment_proto.gbuffer_bank_width();
            experiment.gbuffer_read_delay = experiment_proto.gbuffer_read_delay() < 1 ? 1 :
                    experiment_proto.gbuffer_read_delay();
            experiment.gbuffer_write_delay = experiment_proto.gbuffer_write_delay() < 1 ? 1 :
                    experiment_proto.gbuffer_write_delay();

            experiment.abuffer_rows = experiment_proto.abuffer_rows() < 1 ? 2 :
                    experiment_proto.abuffer_rows();
            experiment.abuffer_read_delay = experiment_proto.abuffer_read_delay() < 1 ? 1 :
                    experiment_proto.abuffer_read_delay();

            experiment.wbuffer_rows = experiment_proto.wbuffer_rows() < 1 ? 2 :
                    experiment_proto.wbuffer_rows();
            experiment.wbuffer_read_delay = experiment_proto.wbuffer_read_delay() < 1 ? 1 :
                    experiment_proto.wbuffer_read_delay();

            experiment.obuffer_rows = experiment_proto.obuffer_rows() < 1 ? 2 :
                    experiment_proto.obuffer_rows();
            experiment.obuffer_write_delay = experiment_proto.obuffer_write_delay() < 1 ? 1 :
                    experiment_proto.obuffer_write_delay();

            experiment.composer_inputs = experiment_proto.composer_inputs() < 1 ? 256 :
                    experiment_proto.composer_inputs();
            experiment.composer_delay = experiment_proto.composer_delay() < 1 ? 1 :
                    experiment_proto.composer_delay();

            experiment.ppu_inputs = experiment_proto.ppu_inputs() < 1 ? 16 : experiment_proto.ppu_inputs();
            experiment.ppu_delay = experiment_proto.ppu_delay() < 1 ? 1 : experiment_proto.ppu_delay();

            // Generic parameters
            experiment.lanes = experiment_proto.lanes() < 1 ? 16 : experiment_proto.lanes();
            experiment.columns = experiment_proto.columns() < 1 ? 16 : experiment_proto.columns();
            experiment.rows = experiment_proto.rows() < 1 ? 16 : experiment_proto.rows();
            experiment.tiles = experiment_proto.tiles() < 1 ? 16 : experiment_proto.tiles();
            experiment.column_registers = experiment_proto.column_registers();
            experiment.pe_width = experiment_proto.pe_width() < 1 ? 16 : experiment_proto.pe_width();

            // BitPragmatic-Laconic
            experiment.booth = experiment_proto.booth_encoding();
            experiment.bits_first_stage = experiment_proto.bits_first_stage();

            // ShapeShifter-Loom
            experiment.group_size = experiment_proto.group_size() < 1 ? 1 : experiment_proto.group_size();
            experiment.minor_bit = experiment_proto.minor_bit();

            if((experiment_proto.architecture() == "ShapeShifter" || experiment_proto.architecture() == "Loon") &&
                    (experiment.columns % experiment.group_size != 0))
                throw std::runtime_error("Group size on network must be divisor of the columns.");

            // Loom
            experiment.dynamic_weights = experiment_proto.dynamic_weights();
            experiment.pe_serial_bits = experiment_proto.pe_serial_bits() < 1 ? 1 :
                    experiment_proto.pe_serial_bits();

            if(experiment_proto.architecture() == "Loom" &&
                    (experiment.rows % experiment.group_size != 0))
                throw std::runtime_error("Group size on network must be divisor of the rows.");

            // BitTactical
            experiment.lookahead_h = experiment_proto.lookahead_h() < 1 ? 2 : experiment_proto.lookahead_h();
            experiment.lookaside_d = experiment_proto.lookaside_d() < 1 ? 5 : experiment_proto.lookaside_d();
            experiment.search_shape = experiment_proto.search_shape().empty() ? "T" :
                    experiment_proto.search_shape();

            const auto &search_shape = experiment.search_shape;
            if(search_shape != "L" && search_shape != "T")
                throw std::runtime_error("BitTactical search shape on network must be <L|T>.");
            if(search_shape == "T" && (experiment.lookahead_h != 2 || experiment.lookaside_d != 5))
                throw std::runtime_error("BitTactical search T-shape on network must be lookahead of 2, and "
                                         "lookaside of 5.");

            // SCNN
            experiment.Wt = experiment_proto.wt() < 1 ? 8 : experiment_proto.wt();
            experiment.Ht = experiment_proto.ht() < 1 ? 8 : experiment_proto.ht();
            experiment.I = experiment_proto.i() < 1 ? 4 : experiment_proto.i();
            experiment.F = experiment_proto.f() < 1 ? 4 : experiment_proto.f();
            experiment.out_acc_size = experiment_proto.out_acc_size() < 1 ?
                    6144 : experiment_proto.out_acc_size();
            experiment.banks = experiment_proto.banks() < 1 ? 32 : experiment_proto.banks();

            if(experiment.banks > 32)
                throw std::runtime_error("Banks for SCNN on network must be from 1 to 32");

            // On top architectures
            experiment.diffy = experiment_proto.diffy();
            experiment.tactical = experiment_proto.tactical();

            // Sanity check
            const auto &task = experiment_proto.task();
            if(task  != "Cycles" && task != "Potentials")
                throw std::runtime_error("Task must be <Cycles|Potentials>.");

            if (task == "Potentials" && experiment.diffy)
                throw std::runtime_error("Diffy simulation on network is only allowed for <Cycles>.");

            const auto &arch = experiment_proto.architecture();
            if (dtype == "Fixed" && arch != "DaDianNao" && arch != "Stripes" && arch != "ShapeShifter" &&
                    arch != "Loom" && arch != "BitPragmatic" && arch != "Laconic" && arch != "SCNN")
                throw std::runtime_error("Architectures allowed for Fixed are <DaDianNao|Stripes|ShapeShifter|"
                                         "Loom|BitPragmatic|Laconic|SCNN>.");
            else  if (dtype == "Float" && arch != "DaDianNao" && arch != "SCNN")
                throw std::runtime_error("Architectures allowed for Float data type are <DaDianNao|SCNN>.");

            if (dtype == "Fixed" && arch != "DaDianNao" && arch != "ShapeShifter" && arch != "BitPragmatic"
                    && experiment.tactical)
                throw std::runtime_error("Tactical simulation for Fixed data type is only allowed for backends "
                                         "<DaDianNao|ShapeShifter|BitPragmatic>");
            else if (dtype == "Float" && arch != "DaDianNao" && experiment.tactical)
                throw std::runtime_error("Tactical simulation for Float data type is only allowed for backends "
                                         "<DaDianNao>");

            if (dtype == "Float" && experiment.diffy)
                throw std::runtime_error("Diffy simulation is not allowed for Float data type");

            if (arch != "ShapeShifter" && arch != "BitPragmatic" && experiment.diffy)
                throw std::runtime_error("Diffy simulation for Fixed data type is only allowed for backends "
                                         "<ShapeShifter|BitPragmatic>");

            if (experiment.tactical && experiment.diffy)
                throw std::runtime_error("Both Tactical and Diffy simulation are not allowed on the same experiment");

            const auto &dataflow = experiment_proto.dataflow();
            if (task == "Cycles" && arch != "SCNN" && dataflow != "WindowFirstOutS")
                throw std::runtime_error("Dataflow on network " + simulate.network + " must be <WindowFirstOutS>.");

            experiment.architecture = experiment_proto.architecture();
            experiment.task = experiment_proto.task();
            experiment.dataflow = experiment_proto.dataflow();
            simulate.experiments.emplace_back(experiment);
        }

        return simulate;
    }

    void Batch::read_batch() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        protobuf::Batch batch;

        if (!ReadProtoFromTextFile(this->path.c_str(),&batch)) {
            throw std::runtime_error("Failed to read prototxt");
        }

        for(const auto &simulate : batch.simulate()) {
            try {
                this->simulations.emplace_back(read_inference_simulation(simulate));
            } catch (std::exception &exception) {
                std::cerr << "Prototxt simulation error: " << exception.what() << std::endl;
                #ifdef STOP_AFTER_ERROR
                exit(1);
                #endif
            }
        }

    }

    /* Getters */
    const std::vector<Batch::Simulate> &Batch::getSimulations() const { return simulations; }

}
