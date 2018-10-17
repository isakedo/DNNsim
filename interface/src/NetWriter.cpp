
#include <interface/NetWriter.h>

namespace interface {

    void NetWriter::fillLayer(protobuf::Network_Layer* layer_proto, const core::Layer &layer) {
        layer_proto->set_type((protobuf::Network_Layer_Type) layer.getType());
        layer_proto->set_name(layer.getName());
        layer_proto->set_input(layer.getInput());
        layer_proto->set_nn(layer.getNn());
        layer_proto->set_kx(layer.getKx());
        layer_proto->set_ky(layer.getKy());
        layer_proto->set_stride(layer.getStride());
        layer_proto->set_padding(layer.getPadding());

        for(size_t length : layer.getWeights().getShape())
            layer_proto->add_wgt_shape((int)length);
        for(unsigned long long i = 0; i < layer.getWeights().getMax_index(); i++)
            layer_proto->add_wgt_data(layer.getWeights().get(i));

        for(size_t length : layer.getActivations().getShape())
            layer_proto->add_act_shape((int)length);
        for(unsigned long long i = 0; i < layer.getActivations().getMax_index(); i++)
            layer_proto->add_act_data(layer.getActivations().get(i));

        for(size_t length : layer.getOutput_activations().getShape())
            layer_proto->add_out_act_shape((int)length);
        for(unsigned long long i = 0; i < layer.getOutput_activations().getMax_index(); i++)
            layer_proto->add_out_act_data(layer.getOutput_activations().get(i));

    }

    void NetWriter::write_network_protobuf(const core::Network &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        protobuf::Network network_proto;
        network_proto.set_name(network.getName());

        for(const core::Layer &layer : network.getLayers())
            fillLayer(network_proto.add_layers(),layer);

        {
            // Write the new network back to disk.
            std::fstream output(this->path, std::ios::out | std::ios::trunc | std::ios::binary);
            if (!network_proto.SerializeToOstream(&output)) {
                throw std::runtime_error("Failed to write protobuf");
            }
        }

        google::protobuf::ShutdownProtobufLibrary();
    }

     void NetWriter::write_network_gzip(const core::Network &network) {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        protobuf::Network network_proto;
        network_proto.set_name(network.getName());

        for(const core::Layer &layer : network.getLayers())
            fillLayer(network_proto.add_layers(),layer);

        // Write the new network back to disk.
        std::fstream output(this->path, std::ios::out | std::ios::trunc | std::ios::binary);

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


}