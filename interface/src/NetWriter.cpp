
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
    void NetWriter<T>::fill_layer(protobuf::Network_Layer* layer_proto, const core::Layer<T> &layer) {
        layer_proto->set_type(layer.getType());
        layer_proto->set_name(layer.getName());
        layer_proto->set_input(layer.getInput());
        layer_proto->set_nn(layer.getNn());
        layer_proto->set_kx(layer.getKx());
        layer_proto->set_ky(layer.getKy());
        layer_proto->set_stride(layer.getStride());
        layer_proto->set_padding(layer.getPadding());
        layer_proto->set_act_prec(layer.getActPrecision());
        layer_proto->set_act_mag(layer.getActMagnitude());
        layer_proto->set_act_frac(layer.getActFraction());
        layer_proto->set_wgt_prec(layer.getWgtPrecision());
        layer_proto->set_wgt_mag(layer.getWgtMagnitude());
        layer_proto->set_wgt_frac(layer.getWgtFraction());

        for (size_t length : layer.getWeights().getShape())
            layer_proto->add_wgt_shape((int) length);

        for (size_t length : layer.getActivations().getShape())
            layer_proto->add_act_shape((int) length);

        for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
            layer_proto->add_wgt_data_fxd(layer.getWeights().get(i));

        for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
            layer_proto->add_act_data_fxd(layer.getActivations().get(i));


    }

    template <typename T>
    void NetWriter<T>::write_network_protobuf(const core::Network<T> &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        check_path("net_traces/" + this->name);
        std::string path = "net_traces/" + this->name + "/model.proto";

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
        std::string path = "net_traces/" + this->name + "/schedule_" + schedule_type + ".proto";

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