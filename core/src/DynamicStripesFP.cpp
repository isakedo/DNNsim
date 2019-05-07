
#include <core/DynamicStripesFP.h>

namespace core {

    /* AVERAGE WIDTH */

    template <typename T>
    void DynamicStripesFP<T>::computeAvgWidthSecondDim(const cnpy::Array<T> &data, double &avg_width,
            uint64_t &bits_baseline, uint64_t &bits_datawidth) {
        
        const int field_bits = EXPONENT ? 8 : 7;

        const std::vector<size_t> &data_shape = data.getShape();
        
        int first_dim = data_shape[0];
        int second_dim = data_shape[1];
        int third_dim = data_shape[2];
        int fourth_dim = data_shape[3];

        std::vector<double> data_width;
        uint64_t data_bits_datawidth = 0;
        for(int i = 0; i < first_dim; i++) {
            for (int j = 0; j < second_dim; j += 16) {
                for (int k = 0; k < third_dim; k++) {
                    for (int l = 0; l < fourth_dim; l++) {

                        uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                        for(int jj = j; jj < std::min(j + 16,second_dim); jj++) {
                            
                            auto data_float = data.get(i, jj, k, l);
                            auto data_bfloat = this->split_bfloat16(data_float);
                            auto field = EXPONENT ? std::get<1>(data_bfloat) : std::get<2>(data_bfloat);

                            if(field != 0) non_zeroes++;

                            const auto &min_max_data_bits = this->minMax(field);

                            auto min_data_bit = std::get<0>(min_max_data_bits);
                            auto max_data_bit = std::get<1>(min_max_data_bits);

                            if(min_data_bit < min_bit) min_bit = min_data_bit;
                            if(max_data_bit > max_bit) max_bit = max_data_bit;

                        }

                        int width;
                        if(LEADING_BIT) width = (min_bit > max_bit) ? 0 : max_bit + 1;
                        else if(MINOR_BIT) width = (min_bit > max_bit) ? 0 : field_bits - min_bit;
                        else width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                        data_bits_datawidth = data_bits_datawidth + (width * non_zeroes);
                        data_width.push_back(width);

                    }
                }
            }

            auto num_data = first_dim * second_dim * third_dim * fourth_dim;
            auto overhead = (uint64_t)((16 + ceil(log2(field_bits))) * ceil(num_data / 16.));

            avg_width = num_data * field_bits;
            bits_baseline = overhead + data_bits_datawidth;
            bits_datawidth = accumulate(data_width.begin(), data_width.end(), 0.0) / data_width.size();
        }

    }

    template <typename T>
    void DynamicStripesFP<T>::computeAvgWidthLayer(const Network<T> &network, int layer_it,
            sys::Statistics::Stats &stats, const int network_bits, const int epoch, const int epochs) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        const Layer<T> &layer = network.getLayers()[layer_it];

        // Stats
        if(epoch == 0) {
            stats.layers.push_back(layer.getName());
            stats.training_time.emplace_back(std::vector<std::chrono::duration<double>>
                    (epochs,std::chrono::duration<double>()));

            stats.fw_act_avg_width.emplace_back(std::vector<double>(epochs,0));
            stats.fw_act_bits_baseline.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.fw_act_bits_datawidth.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.fw_wgt_avg_width.emplace_back(std::vector<double>(epochs,0));
            stats.fw_wgt_bits_baseline.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.fw_wgt_bits_datawidth.emplace_back(std::vector<uint64_t>(epochs,0));

            stats.bw_in_grad_avg_width.emplace_back(std::vector<double>(epochs,0));
            stats.bw_in_grad_bits_baseline.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.bw_in_grad_bits_datawidth.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.bw_wgt_grad_avg_width.emplace_back(std::vector<double>(epochs,0));
            stats.bw_wgt_grad_bits_baseline.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.bw_wgt_grad_bits_datawidth.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.bw_out_grad_avg_width.emplace_back(std::vector<double>(epochs,0));
            stats.bw_out_grad_bits_baseline.emplace_back(std::vector<uint64_t>(epochs,0));
            stats.bw_out_grad_bits_datawidth.emplace_back(std::vector<uint64_t>(epochs,0));
        }

        // Forward traces: Activations
        cnpy::Array<T> act = layer.getActivations();
        if(layer.getType() == "InnerProduct") {
            if(act.getDimensions() == 4) act.reshape_to_2D();
            act.reshape_to_4D();
        }
        computeAvgWidthSecondDim(act,stats.fw_act_avg_width[layer_it][epoch],
            stats.fw_act_bits_baseline[layer_it][epoch], stats.fw_act_bits_datawidth[layer_it][epoch]);
        
        // Forward traces: Weights

        // Backward traces: Input Activations

        // Backward traces: Weight Gradients

        // Backward traces: Output Gradients

        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        stats.training_time[layer_it][epoch] = time_span;

    }

    template <typename T>
    void DynamicStripesFP<T>::average_width(const Network<T> &network, sys::Statistics::Stats &stats, int epoch,
            int epochs) {

        if(epoch == 0) {
            stats.task_name = "average_width";
            stats.net_name = network.getName();
            stats.arch = "DynamicStripesFP";
        }

        for(int layer_it = 0; layer_it < network.getLayers().size(); layer_it++) {
            const Layer<T> &layer = network.getLayers()[layer_it];
            if(layer.getType() == "Convolution" || layer.getType() == "InnerProduct")
                computeAvgWidthLayer(network, layer_it, stats, network.getNetwork_bits(), epoch, epochs);
        }

    }

    template class DynamicStripesFP<float>;

}
