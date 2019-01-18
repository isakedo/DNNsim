
#include <interface/NetWriter.h>

namespace interface {

    template <typename T>
    void NetWriter<T>::check_path(const std::string &path) {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    template <typename T>
    std::string NetWriter<T>::outputName() {
        std::string output_name = this->name;
        std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time
        if(type == "f4" && this->data_conversion == "Fixed16") type = "t2";
        output_name += "-" + type;
        return output_name;
    }

    /* Return value in two complement */
    static inline
    uint16_t limitPrec(float num, int mag, int prec) {
        double scale = pow(2.,(double)prec);
        double intmax = (1 << (mag + prec - 1)) - 1;
        double intmin = -1 * intmax;
        double ds = num * scale;
        if (ds > intmax) ds = intmax;
        if (ds < intmin) ds = intmin;
        auto two_comp = (int)round(ds);
        return (uint16_t)two_comp;
    }

    template <typename T>
    void NetWriter<T>::fill_layer(protobuf::Network_Layer* layer_proto, const core::Layer<T> &layer) {
        layer_proto->set_type(layer.getType());
        layer_proto->set_name(layer.getName());
        layer_proto->set_input(layer.getInput());
        layer_proto->set_nn(layer.getNn());
        layer_proto->set_kx(layer.getKx());
        layer_proto->set_ky(layer.getKy());
        layer_proto->set_stride(layer.getStride());
        layer_proto->set_padding(layer.getPadding());
        layer_proto->set_act_mag(std::get<0>(layer.getAct_precision()));
        layer_proto->set_act_prec(std::get<1>(layer.getAct_precision()));
        layer_proto->set_wgt_mag(std::get<0>(layer.getWgt_precision()));
        layer_proto->set_wgt_prec(std::get<1>(layer.getWgt_precision()));


        std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time

        // Add weights, activations, and output activations only to the desired layers
        if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {

            for (size_t length : layer.getWeights().getShape())
                layer_proto->add_wgt_shape((int) length);

            for (size_t length : layer.getActivations().getShape())
                layer_proto->add_act_shape((int) length);

            if(this->activate_bias_and_out_act) {
                for (size_t length : layer.getBias().getShape())
                    layer_proto->add_bias_shape((int) length);
                for (size_t length : layer.getOutput_activations().getShape())
                    layer_proto->add_out_act_shape((int) length);
            }

            if(type == "f4" && this->data_conversion == "Not") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_flt(layer.getWeights().get(i));

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_flt(layer.getActivations().get(i));

                if (this->activate_bias_and_out_act) {
                    for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                        layer_proto->add_bias_data_flt(layer.getBias().get(i));
                    for (unsigned long long i = 0; i < layer.getOutput_activations().getMax_index(); i++)
                        layer_proto->add_out_act_data_flt(layer.getOutput_activations().get(i));
                }

            } else if (type == "f4" && this->data_conversion == "Fixed16") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_fxd(limitPrec(layer.getWeights().get(i),
                         std::get<0>(layer.getWgt_precision()),std::get<1>(layer.getWgt_precision())));

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_fxd(limitPrec(layer.getActivations().get(i),
                        std::get<0>(layer.getAct_precision()),std::get<1>(layer.getAct_precision())));

                if (this->activate_bias_and_out_act) {
                    for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                        layer_proto->add_bias_data_fxd(limitPrec(layer.getBias().get(i),
                                std::get<0>(layer.getAct_precision()),std::get<1>(layer.getAct_precision())));
                    for (unsigned long long i = 0; i < layer.getOutput_activations().getMax_index(); i++)
                        layer_proto->add_out_act_data_fxd(limitPrec(layer.getOutput_activations().get(i),
                                std::get<0>(layer.getAct_precision()),std::get<1>(layer.getAct_precision())));
                }

            } else if (type == "t2") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_fxd(layer.getWeights().get(i));

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_fxd(layer.getActivations().get(i));

                if (this->activate_bias_and_out_act) {
                    for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                        layer_proto->add_bias_data_fxd(layer.getBias().get(i));
                    for (unsigned long long i = 0; i < layer.getOutput_activations().getMax_index(); i++)
                        layer_proto->add_out_act_data_fxd(layer.getOutput_activations().get(i));
                }

            }

        }

    }

    template <typename T>
    void NetWriter<T>::write_network_protobuf(const core::Network<T> &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        check_path("net_traces/" + this->name);
        std::string path = "net_traces/" + this->name + '/' + outputName() + ".proto";

        try {
            // If Protobuf is found, do not overwrite
            check_path(path);
            return;
        } catch (std::exception &exception) {}

        protobuf::Network network_proto;
        network_proto.set_name(network.getName());

        for(const core::Layer<T> &layer : network.getLayers())
            fill_layer(network_proto.add_layers(),layer);

        {
            // Write the new network back to disk.
            std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
            if (!network_proto.SerializeToOstream(&output)) {
                throw std::runtime_error("Failed to write protobuf");
            }

            #ifdef DEBUG
            std::cout << "Protobuf written in: " << path << std::endl;
            #endif
        }

    }

    template <typename T>
    void NetWriter<T>::write_network_gzip(const core::Network<T> &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        check_path("net_traces/" + this->name);
        std::string path = "net_traces/" + this->name + '/' + outputName();

        try {
            // If Gzip is found, do not overwrite
            check_path(path);
            return;
        } catch (std::exception &exception) {}

        protobuf::Network network_proto;
        network_proto.set_name(network.getName());

        for(const core::Layer<T> &layer : network.getLayers())
            fill_layer(network_proto.add_layers(),layer);

        // Write the new network back to disk.
        std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);

        google::protobuf::io::OstreamOutputStream outputFileStream(&output);
        google::protobuf::io::GzipOutputStream::Options options;
        options.format = google::protobuf::io::GzipOutputStream::GZIP;
        options.compression_level = 9; //Max compression

        google::protobuf::io::GzipOutputStream gzipOutputStream(&outputFileStream, options);

        if (!network_proto.SerializeToZeroCopyStream(&gzipOutputStream)) {
            throw std::runtime_error("Failed to write Gzip protobuf");
        }

        #ifdef DEBUG
        std::cout << "Gzip Protobuf written in: " << path << std::endl;
        #endif
    }

    template <typename T>
    void NetWriter<T>::fill_schedule_tuple(protobuf::Schedule_Layer_Time_Tuple* schedule_tuple_proto,
            const schedule_tuple &dense_schedule_tuple) {
        schedule_tuple_proto->set_channel(std::get<0>(dense_schedule_tuple));
        schedule_tuple_proto->set_kernel_x(std::get<1>(dense_schedule_tuple));
        schedule_tuple_proto->set_kernel_y(std::get<2>(dense_schedule_tuple));
        schedule_tuple_proto->set_wgt_bits(std::get<3>(dense_schedule_tuple));
    }

    template <typename T>
    void NetWriter<T>::write_schedule_protobuf(const std::vector<schedule> &network_schedule,
            const std::string &schedule_type) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        check_path("net_traces/" + this->name);
        std::string path = "net_traces/" + this->name + '/' + outputName() + "_" + schedule_type + "_schedule.proto";
        /*
        try {
            // If Protobuf is found, do not overwrite
            check_path(path);
            return;
        } catch (std::exception &exception) {}*/

        protobuf::Schedule network_schedule_proto;

        for(const auto &schedule : network_schedule) {
            auto layer_proto_ptr = network_schedule_proto.add_layers();
            for (const auto &schedule_time : schedule) {
                auto time_proto_ptr = layer_proto_ptr->add_times();
                for (const auto &schedule_tuple : schedule_time)
                    fill_schedule_tuple(time_proto_ptr->add_tuples(),schedule_tuple);
            }
        }

        {
            // Write the schedule back to disk.
            std::fstream output(path, std::ios::out | std::ios::trunc | std::ios::binary);
            if (!network_schedule_proto.SerializeToOstream(&output)) {
                throw std::runtime_error("Failed to write protobuf");
            }

            #ifdef DEBUG
            std::cout << "Schedule written in: " << path << std::endl;
            #endif
        }
    }

    INITIALISE_DATA_TYPES(NetWriter);

}