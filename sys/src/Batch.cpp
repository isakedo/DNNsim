
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

    Batch::Transform Batch::read_transformation(const protobuf::Batch_Transform &transform_proto) {
        Batch::Transform transform;
        std::string value;
        transform.network = transform_proto.network();
        transform.activate_bias_out_act = transform_proto.activate_bias_and_out_act();
        transform.batch = transform_proto.batch();

        value = transform_proto.inputtype();
        if(value  != "Caffe" && value != "Trace" && value != "CParams" && value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Input type configuration for network " + transform.network +
                                     " must be <Caffe|Trace|CParams|Protobuf|Gzip>.");
        else
            transform.inputType = transform_proto.inputtype();

        value = transform_proto.inputdatatype();
        if(value  != "Float32" && value != "Fixed16")
            throw std::runtime_error("Input data type configuration for network " + transform.network +
                " must be <Float32|Fixed16>.");
        else
            transform.inputDataType = transform_proto.inputdatatype();

        value = transform_proto.outputtype();
        if(value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Output type configuration for network " + transform.network +
                                     " must be <Protobuf|Gzip>.");
        else
            transform.outputType = transform_proto.outputtype();

        value = transform_proto.outputdatatype();
        if(value  != "Float32" && value != "Fixed16")
            throw std::runtime_error("Output data type configuration for network " + transform.network +
                                     " must be <Float32|Fixed16>.");
        else {
            // Only allow conversion from float32 to fixed16
            std::string data_conversion = transform_proto.inputdatatype() == "Float32" &&
                    transform_proto.outputdatatype() == "Fixed16" ? transform_proto.outputdatatype() : "Not";
            transform.outputDataType = data_conversion;
        }

        return transform;
    }

    Batch::Simulate Batch::read_simulation(const protobuf::Batch_Simulate &simulate_proto) {
        Batch::Simulate simulate;
        std::string value;
        simulate.network = simulate_proto.network();
        simulate.activate_bias_out_act = simulate_proto.activate_bias_and_out_act();
        simulate.batch = simulate_proto.batch();

        value = simulate_proto.inputtype();
        if(value  != "Caffe" && value != "Trace" && value != "CParams" && value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Input type configuration for network " + simulate.network +
                                     " must be <Caffe|Trace|CParams|Protobuf|Gzip>.");
        else
            simulate.inputType = simulate_proto.inputtype();

        value = simulate_proto.inputdatatype();
        if(value  != "Float32" && value != "Fixed16")
            throw std::runtime_error("Input data type configuration for network " + simulate.network +
                                     " must be <Float32|Fixed16>.");
        else
            simulate.inputDataType = simulate_proto.inputdatatype();

        if (simulate.inputDataType == "Fixed16") {
            for(const auto &experiment_proto : simulate_proto.experiment()) {

                Batch::Simulate::Experiment experiment;
                if(experiment_proto.architecture() == "BitPragmatic") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
                    experiment.column_registers = experiment_proto.column_registers();
                    experiment.bits_first_stage = experiment_proto.bits_first_stage();

                } else if(experiment_proto.architecture() == "Stripes") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
                    experiment.bits_pe = experiment_proto.bits_pe() < 1 ? 16 : experiment_proto.bits_pe();

                } else if(experiment_proto.architecture() == "DynamicStripes") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
                    experiment.column_registers = experiment_proto.column_registers();
                    experiment.precision_granularity = experiment_proto.precision_granularity().empty() ? "Tile" :
                            experiment_proto.precision_granularity();
                    value = experiment.precision_granularity;
                    if(value != "Tile" && value != "SIP")
                        throw std::runtime_error("Dynamic-Stripes per precision granularity specification for network "
                                                + simulate.network + " must be <Tile|SIP>.");

                } else if (experiment_proto.architecture() == "Laconic") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 8 : experiment_proto.n_rows();

                } else if (experiment_proto.architecture() == "BitTacticalP") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
                    experiment.column_registers = experiment_proto.column_registers();
                    experiment.precision_granularity = experiment_proto.precision_granularity().empty() ? "Tile" :
                            experiment_proto.precision_granularity();
                    experiment.lookahead_h = experiment_proto.lookahead_h() < 1 ? 2 : experiment_proto.lookahead_h();
                    experiment.lookaside_d = experiment_proto.lookaside_d() < 1 ? 5 : experiment_proto.lookaside_d();
                    experiment.search_shape = experiment_proto.search_shape().empty() ? 'L' :
                            experiment_proto.search_shape().c_str()[0];
                    experiment.read_schedule_from_proto = experiment_proto.read_schedule_from_proto();
                    value = experiment.search_shape;
                    if(value != "L" && value != "T")
                        throw std::runtime_error("BitTactical search shape for network " + simulate.network +
                                                 " must be <L|T>.");
                    if(value == "T" && (experiment.lookahead_h != 2 || experiment.lookaside_d != 5))
                        throw std::runtime_error("BitTactical search T-shape for network " + simulate.network +
                                                 " must be lookahead of 2, and lookaside of 5.");
                    value = experiment.precision_granularity;
                    if(value != "Tile" && value != "SIP")
                        throw std::runtime_error("BitTacticalP per precision granularity specification for network "
                                                 + simulate.network + " must be <Tile|SIP>.");

                } else if (experiment_proto.architecture() == "BitTacticalE") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
                    experiment.column_registers = experiment_proto.column_registers();
                    experiment.bits_first_stage = experiment_proto.bits_first_stage();
                    experiment.lookahead_h = experiment_proto.lookahead_h() < 1 ? 2 : experiment_proto.lookahead_h();
                    experiment.lookaside_d = experiment_proto.lookaside_d() < 1 ? 5 : experiment_proto.lookaside_d();
                    experiment.search_shape = experiment_proto.search_shape().empty() ? 'L' :
                                              experiment_proto.search_shape().c_str()[0];
                    experiment.read_schedule_from_proto = experiment_proto.read_schedule_from_proto();
                    value = experiment.search_shape;
                    if(value != "L" && value != "T")
                        throw std::runtime_error("BitTactical search shape for network " + simulate.network +
                                                 " must be <L|T>.");
                    if(value == "T" && (experiment.lookahead_h != 2 || experiment.lookaside_d != 5))
                        throw std::runtime_error("BitTactical search T-shape for network " + simulate.network +
                                                 " must be lookahead of 2, and lookaside of 5.");

                } else if (experiment_proto.architecture() == "SCNN") {
                    experiment.Wt = experiment_proto.wt() < 1 ? 8 : experiment_proto.wt();
                    experiment.Ht = experiment_proto.ht() < 1 ? 8 : experiment_proto.ht();
                    experiment.I = experiment_proto.i() < 1 ? 4 : experiment_proto.i();
                    experiment.F = experiment_proto.f() < 1 ? 4 : experiment_proto.f();
                    experiment.out_acc_size = experiment_proto.out_acc_size() < 1 ?
                            1024 : experiment_proto.out_acc_size();
                    experiment.banks = experiment_proto.banks() < 1 ? 32 : experiment_proto.banks();
                    if(experiment.banks > 32)
                        throw std::runtime_error("Banks for SCNN in network " + simulate.network +
                                                 " must be from 1 to 32");

                } else if (experiment_proto.architecture() == "SCNNp") {
                    experiment.Wt = experiment_proto.wt() < 1 ? 32 : experiment_proto.wt();
                    experiment.Ht = experiment_proto.ht() < 1 ? 32 : experiment_proto.ht();
                    experiment.I = experiment_proto.i() < 1 ? 4 : experiment_proto.i();
                    experiment.F = experiment_proto.f() < 1 ? 4 : experiment_proto.f();
                    experiment.out_acc_size = experiment_proto.out_acc_size() < 1 ?
                            1024 : experiment_proto.out_acc_size();
                    experiment.banks = experiment_proto.banks() < 1 ? 32 : experiment_proto.banks();
                    if(experiment.banks > 32)
                        throw std::runtime_error("Banks for SCNN in network " + simulate.network +
                                                 " must be from 1 to 32");

                } else if (experiment_proto.architecture() == "SCNNe") {
                    experiment.Wt = experiment_proto.wt() < 1 ? 32 : experiment_proto.wt();
                    experiment.Ht = experiment_proto.ht() < 1 ? 32 : experiment_proto.ht();
                    experiment.I = experiment_proto.i() < 1 ? 4 : experiment_proto.i();
                    experiment.F = experiment_proto.f() < 1 ? 4 : experiment_proto.f();
                    experiment.out_acc_size = experiment_proto.out_acc_size() < 1 ?
                            1024 : experiment_proto.out_acc_size();
                    experiment.banks = experiment_proto.banks() < 1 ? 32 : experiment_proto.banks();
                    if(experiment.banks > 32)
                        throw std::runtime_error("Banks for SCNN in network " + simulate.network +
                                                 " must be from 1 to 32");

                } else if(experiment_proto.architecture() == "BitFusion") {
                    experiment.num_pe = experiment_proto.num_pe() < 1 ? 512 : experiment_proto.num_pe();

                }else throw std::runtime_error("Architecture for network " + simulate.network +
                                                " in Fixed16 must be <BitPragmatic|Stripes|DynamicStripes|Laconic|"
                                                "BitTacticalP|BitTacticalE|SCNN|SCNNp|SCNNe|BitFusion>.");

                value = experiment_proto.task();
                if(value  != "Cycles" && value != "Potentials" && value != "Schedule")
                    throw std::runtime_error("Task for network " + simulate.network +
                                             " in Fixed16 must be <Cycles|Potentials|Schedule>.");

                experiment.architecture = experiment_proto.architecture();
                experiment.task = experiment_proto.task();
                simulate.experiments.emplace_back(experiment);

            }
        } else if (simulate.inputDataType == "Float32") {
            for(const auto &experiment_proto : simulate_proto.experiment()) {

                Batch::Simulate::Experiment experiment;
                if(experiment_proto.architecture() == "None") {
                    if(!simulate_proto.activate_bias_and_out_act() || experiment_proto.task() != "Inference")
                        throw std::runtime_error("Float32 None only allows \"Inference\" task, with the flag "
                                                 "\"activate_bias_and_out_act\" activated");

                } else if (experiment_proto.architecture() == "SCNN") {
                    experiment.Wt = experiment_proto.wt() < 1 ? 8 : experiment_proto.wt();
                    experiment.Ht = experiment_proto.ht() < 1 ? 8 : experiment_proto.ht();
                    experiment.I = experiment_proto.i() < 1 ? 4 : experiment_proto.i();
                    experiment.F = experiment_proto.f() < 1 ? 4 : experiment_proto.f();
                    experiment.out_acc_size = experiment_proto.out_acc_size() < 1 ?
                                              1024 : experiment_proto.out_acc_size();
                    experiment.banks = experiment_proto.banks() < 1 ? 32 : experiment_proto.banks();
                    if(experiment.banks > 32)
                        throw std::runtime_error("Banks for SCNN in network " + simulate.network +
                                                 " must be from 1 to 32");

                } else throw std::runtime_error("Architecture for network " + simulate.network +
                                                " in Float32 must be <None|SCNN>.");

                value = experiment_proto.task();
                if(value  != "Cycles" && value != "Potentials" && value != "Inference")
                    throw std::runtime_error("Task for network " + simulate.network +
                                             " in Float32 must be <Inference|Cycles|Potentials>.");

                experiment.architecture = experiment_proto.architecture();
                experiment.task = experiment_proto.task();
                simulate.experiments.emplace_back(experiment);

            }
        }

        // Allow fixed point directly from Caffe, Trace and CParams
        if(simulate.inputDataType == "Fixed16" && (simulate.inputType == "Caffe" || simulate.inputType == "Trace" ||
                simulate.inputType == "CParams")) {
            Batch::Transform transform;
            transform.network = simulate_proto.network();
            transform.inputType = simulate.inputType;
            transform.inputDataType = "Float32";
            transform.outputType = "Protobuf";
            transform.outputDataType = "Fixed16";
            this->transformations.emplace_back(transform);
            simulate.inputType = "Protobuf";
        }

        return simulate;
    }

    void Batch::read_batch() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        protobuf::Batch batch;

        if (!ReadProtoFromTextFile(this->path.c_str(),&batch)) {
            throw std::runtime_error("Failed to read prototxt");
        }

        for(const auto &transform : batch.transform()) {
            try {
                this->transformations.emplace_back(read_transformation(transform));
            } catch (std::exception &exception) {
                std::cerr << "Prototxt transformation error: " << exception.what() << std::endl;
                #ifdef STOP_AFTER_ERROR
                exit(1);
                #endif
            }
        }

        for(const auto &simulate : batch.simulate()) {
            try {
                this->simulations.emplace_back(read_simulation(simulate));
            } catch (std::exception &exception) {
                std::cerr << "Prototxt simulation error: " << exception.what() << std::endl;
                #ifdef STOP_AFTER_ERROR
                exit(1);
                #endif
            }
        }

    }

    /* Getters */
    const std::vector<Batch::Transform> &Batch::getTransformations() const { return transformations; }
    const std::vector<Batch::Simulate> &Batch::getSimulations() const { return simulations; }

}