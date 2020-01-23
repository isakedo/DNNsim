
#include <sys/Batch.h>

namespace sys {

    bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto) {
        int fd = open(filename, O_RDONLY);
        auto input = new google::protobuf::io::FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }

    Batch::Simulate Batch::read_inference_simulation(const protobuf::Batch_Simulate &simulate_proto) {

        Batch::Simulate simulate;
        simulate.network = simulate_proto.network();
        simulate.batch = simulate_proto.batch();
        simulate.tensorflow_8b = simulate_proto.tensorflow_8b();
        simulate.intel_inq = simulate_proto.intel_inq();
        simulate.network_bits = simulate_proto.network_bits() < 1 ? 16 : simulate_proto.network_bits();
        if(simulate.tensorflow_8b) simulate.network_bits = 8;

        const auto &model = simulate_proto.model();
        if(model  != "Caffe" && model != "CSV")
            throw std::runtime_error("Model configuration on network " + simulate.network +
                                     " must be <Caffe|CSV>.");
        else
            simulate.model = simulate_proto.model();

        const auto &dtype = simulate_proto.data_type();
        if(dtype  != "Float32" && dtype != "Fixed16")
            throw std::runtime_error("Input data type configuration on network " + simulate.network +
                                     " must be <Float32|Fixed16>.");
        else
            simulate.data_type = simulate_proto.data_type();

        for(const auto &experiment_proto : simulate_proto.experiment()) {

            Batch::Simulate::Experiment experiment;
            experiment.n_lanes = experiment_proto.n_lanes() < 1 ? 16 : experiment_proto.n_lanes();
            experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
            experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
            experiment.n_tiles = experiment_proto.n_tiles() < 1 ? 16 : experiment_proto.n_tiles();
            experiment.column_registers = experiment_proto.column_registers();
            experiment.bits_pe = experiment_proto.bits_pe() < 1 ? 16 : experiment_proto.bits_pe();

            // BitPragmatic-Laconic
            experiment.booth = experiment_proto.booth_encoding();
            experiment.bits_first_stage = experiment_proto.bits_first_stage();

            // ShapeShifter-Loom
            experiment.group_size = experiment_proto.group_size() < 1 ? 1 : experiment_proto.group_size();
            experiment.minor_bit = experiment_proto.minor_bit();

            if((experiment_proto.architecture() == "ShapeShifter" || experiment_proto.architecture() == "Loon") &&
                    (experiment.n_columns % experiment.group_size != 0))
                throw std::runtime_error("Group size on network " + simulate.network +
                        " must be divisor of the columns.");

            // Loom
            experiment.dynamic_weights = experiment_proto.dynamic_weights();
            experiment.pe_serial_bits = experiment_proto.pe_serial_bits() < 1 ? 1 :
                    experiment_proto.pe_serial_bits();

            if(experiment_proto.architecture() == "Loom" &&
                    (experiment.n_rows % experiment.group_size != 0))
                throw std::runtime_error("Group size on network " + simulate.network +
                        " must be divisor of the rows.");

            // BitTactical
            experiment.lookahead_h = experiment_proto.lookahead_h() < 1 ? 2 : experiment_proto.lookahead_h();
            experiment.lookaside_d = experiment_proto.lookaside_d() < 1 ? 5 : experiment_proto.lookaside_d();
            experiment.search_shape = experiment_proto.search_shape().empty() ? "T" :
                    experiment_proto.search_shape();
            experiment.read_schedule = experiment_proto.read_schedule();

            const auto &search_shape = experiment.search_shape;
            if(search_shape != "L" && search_shape != "T")
                throw std::runtime_error("BitTactical search shape on network " + simulate.network +
                                         " must be <L|T>.");
            if(search_shape == "T" && (experiment.lookahead_h != 2 || experiment.lookaside_d != 5))
                throw std::runtime_error("BitTactical search T-shape on network " + simulate.network +
                                         " must be lookahead of 2, and lookaside of 5.");

            // SCNN
            experiment.Wt = experiment_proto.wt() < 1 ? 8 : experiment_proto.wt();
            experiment.Ht = experiment_proto.ht() < 1 ? 8 : experiment_proto.ht();
            experiment.I = experiment_proto.i() < 1 ? 4 : experiment_proto.i();
            experiment.F = experiment_proto.f() < 1 ? 4 : experiment_proto.f();
            experiment.out_acc_size = experiment_proto.out_acc_size() < 1 ?
                    6144 : experiment_proto.out_acc_size();
            experiment.banks = experiment_proto.banks() < 1 ? 32 : experiment_proto.banks();

            if(experiment.banks > 32)
                throw std::runtime_error("Banks for SCNN on network " + simulate.network +
                                         " must be from 1 to 32");

            // On top architectures
            experiment.diffy = experiment_proto.diffy();
            experiment.tactical = experiment_proto.tactical();

            // Sanity check
            const auto &task = experiment_proto.task();
            if(task  != "Cycles" && task != "Potentials")
                throw std::runtime_error("Task on network " + simulate.network +
                                         " in Fixed16 must be <Cycles|Potentials>.");

            if (task == "Potentials" && experiment.diffy)
                throw std::runtime_error("Diffy simulation on network " + simulate.network +
                                         " is only allowed for <Cycles>.");

            const auto &arch = experiment_proto.architecture();
            if (dtype == "Fixed16" && arch != "DaDianNao" && arch != "Stripes" && arch != "ShapeShifter" &&
                    arch != "Loom" && arch != "BitPragmatic" && arch != "Laconic" && arch != "SCNN")
                throw std::runtime_error("Architecture on network " + simulate.network +
                    " in Fixed16 must be <DaDianNao|Stripes|ShapeShifter|Loom|BitPragmatic|Laconic|SCNN>.");
            else  if (dtype == "Float32" && arch != "DaDianNao" && arch != "SCNN")
                throw std::runtime_error("Architecture on network " + simulate.network +
                                         " in Float32 must be <DaDianNao|SCNN>.");

            if (arch != "DaDianNao" && arch != "ShapeShifter" && arch != "BitPragmatic" && experiment.tactical)
                throw std::runtime_error("Tactical simulation on network " + simulate.network +
                        " in Fixed16 is only allowed for backends <DaDianNao|ShapeShifter|BitPragmatic>");

            if (arch != "ShapeShifter" && arch != "BitPragmatic" && experiment.diffy)
                throw std::runtime_error("Diffy simulation on network " + simulate.network +
                                         " in Fixed16 is only allowed for backends <ShapeShifter|BitPragmatic>");

            if (experiment.tactical && experiment.diffy)
                throw std::runtime_error("Both Tactical and Diffy simulation on network " + simulate.network +
                                         " are not allowed");

            experiment.architecture = experiment_proto.architecture();
            experiment.task = experiment_proto.task();
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
