
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

        value = transform_proto.inputtype();
        if(value  != "Caffe" && value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Input type configuration for network " + transform.network +
                " must be <Caffe|Protobuf|Gzip>.");
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
                " must be <Caffe|Protobuf|Gzip>.");
        else
            transform.outputType = transform_proto.outputtype();

        value = transform_proto.outputdatatype();
        if(value  != "Float32" && value != "Fixed16")
            throw std::runtime_error("Output data type configuration for network " + transform.network +
                                     " must be <Float32|Fixed16>.");
        else {
            // Only allow conversion from float32 to fixed16
            std::string data_conversion = transform_proto.inputdatatype() == "Float32" &&
                                          transform_proto.outputdatatype() == "Fixed16"
                                          ? transform_proto.outputdatatype() : "Not";
            transform.outputDataType = data_conversion;
        }

        return transform;
    }

    Batch::Simulate Batch::read_simulation(const protobuf::Batch_Simulate &simulate_proto) {
        Batch::Simulate simulate;
        std::string value;
        simulate.network = simulate_proto.network();
        simulate.activate_bias_out_act = simulate_proto.activate_bias_and_out_act();

        value = simulate_proto.inputtype();
        if(value  != "Caffe" && value != "Protobuf" && value != "Gzip")
            throw std::runtime_error("Input type configuration for network " + simulate.network +
                                     " must be <Caffe|Protobuf|Gzip>.");
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
                    experiment.bits_first_stage = experiment_proto.bits_first_stage();

                } else if(experiment_proto.architecture() == "Stripes") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();

                } else if (experiment_proto.architecture() == "Laconic") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 8 : experiment_proto.n_rows();

                } else if (experiment_proto.architecture() == "BitTacticalP") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
                    experiment.lookahead_h = experiment_proto.lookahead_h() < 1 ? 2 : experiment_proto.lookahead_h();
                    experiment.lookaside_d = experiment_proto.lookaside_d() < 1 ? 5 : experiment_proto.lookaside_d();
                    experiment.search_shape = experiment_proto.search_shape().empty() ? 'L' :
                            experiment_proto.search_shape().c_str()[0];
                    value = experiment.search_shape;
                    if(value != "L" && value != "T")
                        throw std::runtime_error("BitTactical search shape for network " + simulate.network +
                                                 " must be <L|T>.");
                    if(value == "T" && (experiment.lookahead_h != 2 || experiment.lookaside_d != 5))
                        throw std::runtime_error("BitTactical search T-shape for network " + simulate.network +
                                                 " must be lookahead of 2, and lookaside of 5.");

                } else if (experiment_proto.architecture() == "BitTacticalE") {
                    experiment.n_columns = experiment_proto.n_columns() < 1 ? 16 : experiment_proto.n_columns();
                    experiment.n_rows = experiment_proto.n_rows() < 1 ? 16 : experiment_proto.n_rows();
                    experiment.bits_first_stage = experiment_proto.bits_first_stage();
                    experiment.lookahead_h = experiment_proto.lookahead_h() < 1 ? 2 : experiment_proto.lookahead_h();
                    experiment.lookaside_d = experiment_proto.lookaside_d() < 1 ? 5 : experiment_proto.lookaside_d();
                    experiment.search_shape = experiment_proto.search_shape().empty() ? 'L' :
                                              experiment_proto.search_shape().c_str()[0];
                    value = experiment.search_shape;
                    if(value != "L" && value != "T")
                        throw std::runtime_error("BitTactical search shape for network " + simulate.network +
                                                 " must be <L|T>.");
                    if(value == "T" && (experiment.lookahead_h != 2 || experiment.lookaside_d != 5))
                        throw std::runtime_error("BitTactical search T-shape for network " + simulate.network +
                                                 " must be lookahead of 2, and lookaside of 5.");

                } else if (experiment_proto.architecture() == "BitFusion") {
                } else throw std::runtime_error("Architecture for network " + simulate.network +
                                                " in Fixed16 must be <BitPragmatic|Stripes|Laconic|BitTacticalP|"
                                                "BitTacticalE|BitFusion>.");

                value = experiment_proto.task();
                if(value  != "Cycles" && value != "MemAccesses" && value != "Potentials")
                    throw std::runtime_error("Simulation type for network " + simulate.network +
                                             " must be <Cycles|Potentials|MemAccesses>.");

                experiment.architecture = experiment_proto.architecture();
                experiment.task = experiment_proto.task();
                simulate.experiments.emplace_back(experiment);

            }
        } else if (simulate.inputDataType == "Float32" && !simulate_proto.activate_bias_and_out_act())
            throw std::runtime_error("Float32 only allows inference simulation, which must have the flag "
                                     "\"activate_bias_and_out_act\" activated");

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