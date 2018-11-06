
#include <interface/NetWriter.h>

namespace interface {

    template <typename T>
    std::string NetWriter<T>::outputName() {
        std::string output_name = this->name;
        std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time
        if(type == "f4" && this->data_conversion == "Fixed16") type = "t2";
        output_name += "-" + type;
        return output_name;
    }

    static inline
    uint16_t limitPrec(float num, int mag, int prec) {
        double scale = pow(2.,(double)prec);
        double intmax = (1 << (mag + prec - 1)) - 1;
        double intmin = -1 * intmax;
        double ds = num * scale;
        if (ds > intmax) ds = intmax;
        if (ds < intmin) ds = intmin;
        auto result = (uint16_t)round(ds);
        return result;
    }

    template <typename T>
    void NetWriter<T>::fillLayer(protobuf::Network_Layer* layer_proto, const core::Layer<T> &layer) {
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

            #ifdef BIAS
            for (size_t length : layer.getBias().getShape())
                layer_proto->add_bias_shape((int) length);
            #endif

            for (size_t length : layer.getActivations().getShape())
                layer_proto->add_act_shape((int) length);

            #ifdef OUTPUT_ACTIVATIONS
            for (size_t length : layer.getOutput_activations().getShape())
                layer_proto->add_out_act_shape((int) length);
            #endif

            if(type == "f4" && this->data_conversion == "Not") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_flt(layer.getWeights().get(i));

                #ifdef BIAS
                for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                    layer_proto->add_bias_data_flt(layer.getBias().get(i));
                #endif

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_flt(layer.getActivations().get(i));

                #ifdef OUTPUT_ACTIVATIONS
                for (unsigned long long i = 0; i < layer.getOutput_activations().getMax_index(); i++)
                    layer_proto->add_out_act_data_flt(layer.getOutput_activations().get(i));
                #endif

            } else if (type == "f4" && this->data_conversion == "Fixed16") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_fxd(limitPrec(layer.getWeights().get(i),
                         std::get<0>(layer.getWgt_precision()),std::get<1>(layer.getWgt_precision())));

                #ifdef BIAS
                for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                    layer_proto->add_bias_data_fxd(limitPrec(layer.getBias().get(i),
                        std::get<0>(layer.getAct_precision()),std::get<1>(layer.getAct_precision())));
                #endif

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_fxd(limitPrec(layer.getActivations().get(i),
                        std::get<0>(layer.getAct_precision()),std::get<1>(layer.getAct_precision())));

                #ifdef OUTPUT_ACTIVATIONS
                for (unsigned long long i = 0; i < layer.getOutput_activations().getMax_index(); i++)
                    layer_proto->add_out_act_data_fxd(limitPrec(layer.getOutput_activations().get(i),
                        std::get<0>(layer.getAct_precision()),std::get<1>(layer.getAct_precision())));
                #endif

            } else if (type == "t2") {
                for (unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
                    layer_proto->add_wgt_data_fxd(layer.getWeights().get(i));

                #ifdef BIAS
                for (unsigned long long i = 0; i < layer.getBias().getMax_index(); i++)
                    layer_proto->add_bias_data_fxd(layer.getBias().get(i));
                #endif

                for (unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
                    layer_proto->add_act_data_fxd(layer.getActivations().get(i));

                #ifdef OUTPUT_ACTIVATIONS
                for (unsigned long long i = 0; i < layer.getOutput_activations().getMax_index(); i++)
                    layer_proto->add_out_act_data_fxd(layer.getOutput_activations().get(i));
                #endif

            }

        }

    }

    template <typename T>
    void NetWriter<T>::write_network_protobuf(const core::Network<T> &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        protobuf::Network network_proto;
        network_proto.set_name(network.getName());

        for(const core::Layer<T> &layer : network.getLayers())
            fillLayer(network_proto.add_layers(),layer);

        {
            // Write the new network back to disk.
            std::fstream output("net_traces/" + this->name + '/' + outputName() + ".proto",
                    std::ios::out | std::ios::trunc | std::ios::binary);
            if (!network_proto.SerializeToOstream(&output)) {
                throw std::runtime_error("Failed to write protobuf");
            }
        }

        google::protobuf::ShutdownProtobufLibrary();
    }

    template <typename T>
    void NetWriter<T>::write_network_gzip(const core::Network<T> &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        protobuf::Network network_proto;
        network_proto.set_name(network.getName());

        for(const core::Layer<T> &layer : network.getLayers())
            fillLayer(network_proto.add_layers(),layer);

        // Write the new network back to disk.
        std::fstream output("net_traces/" + this->name + '/' + outputName(),
                std::ios::out | std::ios::trunc | std::ios::binary);

        google::protobuf::io::OstreamOutputStream outputFileStream(&output);
        google::protobuf::io::GzipOutputStream::Options options;
        options.format = google::protobuf::io::GzipOutputStream::GZIP;
        options.compression_level = 9; //Max compression

        google::protobuf::io::GzipOutputStream gzipOutputStream(&outputFileStream, options);

        if (!network_proto.SerializeToZeroCopyStream(&gzipOutputStream)) {
            throw std::runtime_error("Failed to write Gzip protobuf");
        }

        google::protobuf::ShutdownProtobufLibrary();
    }

    INITIALISE_DATA_TYPES(NetWriter);

}