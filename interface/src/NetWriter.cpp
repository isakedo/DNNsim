
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
    uint16_t profiled_precision(float num, int mag, int frac) {
        double scale = pow(2.,(double)frac);
        double intmax = (1 << (mag + frac)) - 1;
        double intmin = -1 * intmax;
        double ds = num * scale;
        if (ds > intmax) ds = intmax;
        if (ds < intmin) ds = intmin;
        auto two_comp = (int)round(ds);
        return (uint16_t)two_comp;
    }

    static inline
    uint16_t tensorflow_8b_precision(float num, double scale, float min_value, int max_fixed, int min_fixed) {
        auto sign_mag = (int)(round(num * scale) - round(min_value * scale) + min_fixed);
        sign_mag = std::max(sign_mag, min_fixed);
        sign_mag = std::min(sign_mag, max_fixed);
        return (uint16_t)sign_mag;
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
        layer_proto->set_act_prec(layer.getAct_precision());
        layer_proto->set_act_mag(layer.getAct_magnitude());
        layer_proto->set_act_frac(layer.getAct_fraction());
        layer_proto->set_wgt_prec(layer.getWgt_precision());
        layer_proto->set_wgt_mag(layer.getWgt_magnitude());
        layer_proto->set_wgt_frac(layer.getWgt_fraction());

        std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time

        // Add weights, activations, and output activations only to the desired layers
        if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {

            for (size_t length : layer.getWeights().getShape())
                layer_proto->add_wgt_shape((int) length);

            for (size_t length : layer.getActivations().getShape())
                layer_proto->add_act_shape((int) length);

            if(this->bias_and_out_act) {
                for (size_t length : layer.getBias().getShape())
                    layer_proto->add_bias_shape((int) length);
                for (size_t length : layer.getOutputActivations().getShape())
                    layer_proto->add_out_act_shape((int) length);
            }

            if(type == "f4" && this->data_conversion == "Not") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_flt(layer.getWeights().get(i));

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_flt(layer.getActivations().get(i));

                if (this->bias_and_out_act) {
                    for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                        layer_proto->add_bias_data_flt(layer.getBias().get(i));
                    for (unsigned long long i = 0; i < layer.getOutputActivations().getMax_index(); i++)
                        layer_proto->add_out_act_data_flt(layer.getOutputActivations().get(i));
                }

            } else if (TENSORFLOW_8b && type == "f4" && this->data_conversion == "Fixed16") {

                const int NUM_BITS = 8;
                const int max_fixed = 127;
                const int min_fixed = -128;
                const int num_discrete_values = 1 << NUM_BITS;
                const auto range_adjust = num_discrete_values / (num_discrete_values - 1.0);

                auto max_wgt = layer.getWeights().max();
                auto min_wgt = layer.getWeights().min();
                auto range_wgt = (max_wgt - min_wgt) * range_adjust;
                auto scale_wgt = num_discrete_values / range_wgt;

                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_fxd(tensorflow_8b_precision(layer.getWeights().get(i),scale_wgt,
                            min_wgt,max_fixed,min_fixed));

                auto max_input_act = layer.getActivations().max();
                auto min_input_act = layer.getActivations().min();
                auto range_input_act = (max_input_act - min_input_act) * range_adjust;
                auto scale_input_act = num_discrete_values / range_input_act;

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_fxd(tensorflow_8b_precision(layer.getActivations().get(i),scale_input_act,
                            min_input_act,max_fixed,min_fixed));

                if (this->bias_and_out_act) {

                    auto max_bias = layer.getBias().max();
                    auto min_bias = layer.getBias().min();
                    auto range_bias = (max_bias - min_bias) * range_adjust;
                    auto scale_bias = num_discrete_values / range_bias;

                    for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                        layer_proto->add_bias_data_fxd(tensorflow_8b_precision(layer.getBias().get(i),scale_bias,
                                min_bias,max_fixed,min_fixed));

                    auto max_output_act = layer.getOutputActivations().max();
                    auto min_output_act = layer.getOutputActivations().min();
                    auto range_output_act = (max_output_act - min_output_act) * range_adjust;
                    auto scale_output_act = num_discrete_values / range_output_act;

                    for (unsigned long long i = 0; i < layer.getOutputActivations().getMax_index(); i++)
                        layer_proto->add_out_act_data_fxd(tensorflow_8b_precision(layer.getOutputActivations().get(i),
                                scale_output_act,min_output_act,max_fixed,min_fixed));
                }

            } else if (type == "f4" && this->data_conversion == "Fixed16") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_fxd(profiled_precision(layer.getWeights().get(i),
                             layer.getWgt_magnitude(),layer.getWgt_fraction()));

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_fxd(profiled_precision(layer.getActivations().get(i),
                            layer.getAct_magnitude(),layer.getAct_fraction()));

                if (this->bias_and_out_act) {
                    for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                        layer_proto->add_bias_data_fxd(profiled_precision(layer.getBias().get(i),1 + 0,15));
                    for (unsigned long long i = 0; i < layer.getOutputActivations().getMax_index(); i++)
                        layer_proto->add_out_act_data_fxd(profiled_precision(layer.getOutputActivations().get(i),
                                1 + 13,2));
                }

            } else if (type == "t2") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_fxd(layer.getWeights().get(i));

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_fxd(layer.getActivations().get(i));

                if (this->bias_and_out_act) {
                    for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                        layer_proto->add_bias_data_fxd(layer.getBias().get(i));
                    for (unsigned long long i = 0; i < layer.getOutputActivations().getMax_index(); i++)
                        layer_proto->add_out_act_data_fxd(layer.getOutputActivations().get(i));
                }

            }

        }

    }

    template <typename T>
    void NetWriter<T>::write_network_protobuf(const core::Network<T> &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        check_path("net_traces/" + this->name);
        std::string name = TENSORFLOW_8b ? outputName() + "-TF" : outputName();
        std::string path = "net_traces/" + this->name + '/' + name + ".proto";

        if(!OVERWRITE) {
            try {
                // If Protobuf is found, do not overwrite
                check_path(path);
                return;
            } catch (std::exception &exception) {}
        }

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
        std::string name = TENSORFLOW_8b ? outputName() + "-TF" : outputName();
        std::string path = "net_traces/" + this->name + '/' + name + ".gz";

        if(!OVERWRITE) {
            try {
                // If Gzip is found, do not overwrite
                check_path(path);
                return;
            } catch (std::exception &exception) {}
        }

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

        if(!OVERWRITE) {
            try {
                // If Protobuf is found, do not overwrite
                check_path(path);
                return;
            } catch (std::exception &exception) {}
        }

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