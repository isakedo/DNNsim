
#include <core/DynamicStripesFP.h>

namespace core {

    /* AVERAGE WIDTH */

    template <typename T>
    void DynamicStripesFP<T>::computeAvgWidthDataFirstDim(const base::Array<T> &data, double &avg_width,
            uint64_t &bits_baseline, uint64_t &bits_datawidth) {

        const uint16_t mask = 0x80;

        const std::vector<size_t> &data_shape = data.getShape();
        
        auto first_dim = data_shape[0];
        auto second_dim = data_shape[1];
        auto third_dim = data_shape[2];
        auto fourth_dim = data_shape[3];

        std::vector<double> data_width;
        uint64_t data_bits_datawidth = 0;
        for(int i = 0; i < first_dim; i += 16) {
            for (int j = 0; j < second_dim; j++) {
                for (int k = 0; k < third_dim; k++) {
                    for (int l = 0; l < fourth_dim; l++) {

                        uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                        for(int dstr = i; dstr < std::min(i + 16,(int)first_dim); dstr++) {
                            
                            auto data_float = data.get(dstr, j, k, l);

                            if(data_float == 0) continue;

                            auto data_bfloat = this->split_bfloat16(data_float);
                            auto biased_exponent = std::get<1>(data_bfloat);

                            int unbiased_exponent = biased_exponent - 127;
                            uint16_t exponent = this->sign_magnitude((short)unbiased_exponent, mask);

                            bool neg = false;
                            if((exponent & mask) != 0) {
                                exponent = exponent & ~mask;
                                neg = true;
                            }

                            if(exponent != 0) non_zeroes++;

                            const auto &min_max_data_bits = this->minMax(exponent);

                            auto min_data_bit = std::get<0>(min_max_data_bits);
                            auto max_data_bit = std::get<1>(min_max_data_bits);

                            if(neg) max_data_bit += 1;

                            if(min_data_bit < min_bit) min_bit = min_data_bit;
                            if(max_data_bit > max_bit) max_bit = max_data_bit;

                        }

                        int width;
                        if(LEADING_BIT) width = (min_bit > max_bit) ? 0 : max_bit + 1;
                        else if(MINOR_BIT) width = (min_bit > max_bit) ? 0 : 8 - min_bit;
                        else width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                        data_bits_datawidth = data_bits_datawidth + (width * non_zeroes);
                        data_width.push_back(width);
                    }
                }
            }

        }

        auto num_data = first_dim * second_dim * third_dim * fourth_dim;
        auto overhead = (uint64_t)((16 + 4) * ceil(num_data / 16.));

        avg_width = accumulate(data_width.begin(), data_width.end(), 0.0) / data_width.size();
        bits_baseline = (uint64_t)num_data * 8;
        bits_datawidth = overhead + data_bits_datawidth;

    }

    template <typename T>
    void DynamicStripesFP<T>::computeAvgWidthDataSecondDim(const base::Array<T> &data, double &avg_width,
            uint64_t &bits_baseline, uint64_t &bits_datawidth) {

        const uint16_t mask = 0x80;

        const std::vector<size_t> &data_shape = data.getShape();

        auto first_dim = data_shape[0];
        auto second_dim = data_shape[1];
        auto third_dim = data_shape[2];
        auto fourth_dim = data_shape[3];

        std::vector<double> data_width;
        uint64_t data_bits_datawidth = 0;
        for(int i = 0; i < first_dim; i++) {
            for (int j = 0; j < second_dim; j += 16) {
                for (int k = 0; k < third_dim; k++) {
                    for (int l = 0; l < fourth_dim; l++) {

                        uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                        for(int dstr = j; dstr < std::min(j + 16,(int)second_dim); dstr++) {

                            auto data_float = data.get(i, dstr, k, l);

                            if(data_float == 0) continue;

                            auto data_bfloat = this->split_bfloat16(data_float);
                            auto biased_exponent = std::get<1>(data_bfloat);

                            int unbiased_exponent = biased_exponent - 127;
                            uint16_t exponent = this->sign_magnitude((short)unbiased_exponent, mask);

                            bool neg = false;
                            if((exponent & mask) != 0) {
                                exponent = exponent & ~mask;
                                neg = true;
                            }

                            if(exponent != 0) non_zeroes++;

                            const auto &min_max_data_bits = this->minMax(exponent);

                            auto min_data_bit = std::get<0>(min_max_data_bits);
                            auto max_data_bit = std::get<1>(min_max_data_bits);

                            if(neg) max_data_bit += 1;

                            if(min_data_bit < min_bit) min_bit = min_data_bit;
                            if(max_data_bit > max_bit) max_bit = max_data_bit;

                        }

                        int width;
                        if(LEADING_BIT) width = (min_bit > max_bit) ? 0 : max_bit + 1;
                        else if(MINOR_BIT) width = (min_bit > max_bit) ? 0 : 8 - min_bit;
                        else width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                        data_bits_datawidth = data_bits_datawidth + (width * non_zeroes);
                        data_width.push_back(width);
                    }
                }
            }

        }

        auto num_data = first_dim * second_dim * third_dim * fourth_dim;
        auto overhead = (uint64_t)((16 + 4) * ceil(num_data / 16.));

        avg_width = accumulate(data_width.begin(), data_width.end(), 0.0) / data_width.size();
        bits_baseline = (uint64_t)num_data * 8;
        bits_datawidth = overhead + data_bits_datawidth;

    }

    template <typename T>
    void DynamicStripesFP<T>::computeAvgWidthDataSeq2Seq(const base::Array<T> &data, double &avg_width,
            uint64_t &bits_baseline, uint64_t &bits_datawidth) {

        const uint16_t mask = 0x80;

        const std::vector<size_t> &data_shape = data.getShape();

        auto first_dim = data_shape[0];
        auto second_dim = data_shape[1];
        auto third_dim = data_shape[2];

        std::vector<double> data_width;
        uint64_t data_bits_datawidth = 0;
        for(int i = 0; i < first_dim; i++) {
            for (int j = 0; j < second_dim; j++) {
                for (int k = 0; k < third_dim; k += 16) {

                    uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                    for(int dstr = k; dstr < std::min(k + 16,(int)third_dim); dstr++) {

                        auto data_float = data.get(i, j, dstr);

                        if(data_float == 0) continue;

                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto biased_exponent = std::get<1>(data_bfloat);

                        int unbiased_exponent = biased_exponent - 127;
                        uint16_t exponent = this->sign_magnitude((short)unbiased_exponent, mask);

                        bool neg = false;
                        if((exponent & mask) != 0) {
                            exponent = exponent & ~mask;
                            neg = true;
                        }

                        if(exponent != 0) non_zeroes++;

                        const auto &min_max_data_bits = this->minMax(exponent);

                        auto min_data_bit = std::get<0>(min_max_data_bits);
                        auto max_data_bit = std::get<1>(min_max_data_bits);

                        if(neg) max_data_bit += 1;

                        if(min_data_bit < min_bit) min_bit = min_data_bit;
                        if(max_data_bit > max_bit) max_bit = max_data_bit;

                    }

                    int width;
                    if(LEADING_BIT) width = (min_bit > max_bit) ? 0 : max_bit + 1;
                    else if(MINOR_BIT) width = (min_bit > max_bit) ? 0 : 8 - min_bit;
                    else width = (min_bit > max_bit) ? 0 : max_bit - min_bit + 1;
                    data_bits_datawidth = data_bits_datawidth + (width * non_zeroes);
                    data_width.push_back(width);
                }
            }
        }

        auto num_data = first_dim * second_dim * third_dim;
        auto overhead = (uint64_t)((16 + 4) * ceil(num_data / 16.));

        avg_width = accumulate(data_width.begin(), data_width.end(), 0.0) / data_width.size();
        bits_baseline = (uint64_t)num_data * 8;
        bits_datawidth = overhead + data_bits_datawidth;

    }

    template <typename T>
    void DynamicStripesFP<T>::average_width(const sys::Batch::Simulate &simulate, int epochs) {

        base::Network<T> network_model;
        interface::NetReader<T> reader = interface::NetReader<T>(simulate.network, 0, 0, this->QUIET);
        network_model = reader.read_network_trace_params();

        // Initialize statistics
        std::string arch = "DynamicStripesFP";
        arch += (LEADING_BIT ? "_LB" : "");
        arch += (MINOR_BIT && !LEADING_BIT ? "_MB" : "");
        std::string filename = arch + "_average_width";
        sys::Stats stats = sys::Stats(network_model.getNumLayers(), epochs, filename);

        // Forward stats
        auto fw_act_avg_width = stats.register_double_t("fw_act_avg_width", 0, sys::Average);
        auto fw_act_bits_baseline = stats.register_uint_t("fw_act_bits_baseline", 0, sys::AverageTotal);
        auto fw_act_bits_datawidth = stats.register_uint_t("fw_act_bits_datawidth", 0, sys::AverageTotal);
        auto fw_wgt_avg_width = stats.register_double_t("fw_wgt_avg_width", 0, sys::Average);
        auto fw_wgt_bits_baseline = stats.register_uint_t("fw_wgt_bits_baseline", 0, sys::AverageTotal);
        auto fw_wgt_bits_datawidth = stats.register_uint_t("fw_wgt_bits_datawidth", 0, sys::AverageTotal);

        auto bw_in_grad_avg_width = stats.register_double_t("bw_in_grad_avg_width", 0, sys::Average, true);
        auto bw_in_grad_bits_baseline = stats.register_uint_t("bw_in_grad_bits_baseline", 0, sys::AverageTotal);
        auto bw_in_grad_bits_datawidth = stats.register_uint_t("bw_in_grad_bits_datawidth", 0, sys::AverageTotal);
        auto bw_wgt_grad_avg_width = stats.register_double_t("bw_wgt_grad_avg_width", 0, sys::Average);
        auto bw_wgt_grad_bits_baseline = stats.register_uint_t("bw_wgt_grad_bits_baseline", 0, sys::AverageTotal);
        auto bw_wgt_grad_bits_datawidth = stats.register_uint_t("bw_wgt_grad_bits_datawidth", 0, sys::AverageTotal);
        auto bw_out_grad_avg_width = stats.register_double_t("bw_out_grad_avg_width", 0, sys::Average);
        auto bw_out_grad_bits_baseline = stats.register_uint_t("bw_out_grad_bits_baseline", 0, sys::AverageTotal);
        auto bw_out_grad_bits_datawidth = stats.register_uint_t("bw_out_grad_bits_datawidth", 0, sys::AverageTotal);

        uint32_t traces_mode = 0;
        if(simulate.only_forward) traces_mode = 1;
        else if(simulate.only_backward) traces_mode = 2;
        else traces_mode = 3;

        for (uint32_t epoch = 0; epoch < epochs; epoch++) {

            base::Network<T> network;
            network = this->read_training(simulate.network, simulate.batch, epoch, simulate.decoder_states,traces_mode);

            if(!this->QUIET) std::cout << "Starting simulation training average width for epoch " << epoch << std::endl;

            for (int layer_it = 0; layer_it < network.getLayers().size(); layer_it++) {

                const base::Layer<float> &layer = network.getLayers()[layer_it];

                if (layer.getType() == "LSTM")
                    continue;

                // Forward
                if (network.getForward()) {

                    // Forward traces: Activations
                    // Conv: Batch, Channels, Nx, Ny
                    // InPr: Batch, NumInputs
                    base::Array<T> act = layer.getActivations();
                    if(layer.getType() == "Decoder" || layer.getType() == "Encoder") {
                        computeAvgWidthDataSeq2Seq(act, fw_act_avg_width->value[layer_it][epoch],
                                                   fw_act_bits_baseline->value[layer_it][epoch],
                                                   fw_act_bits_datawidth->value[layer_it][epoch]);
                    } else {
                        if (act.getDimensions() == 2) act.reshape_to_4D();
                        computeAvgWidthDataSecondDim(act, fw_act_avg_width->value[layer_it][epoch],
                                                     fw_act_bits_baseline->value[layer_it][epoch],
                                                     fw_act_bits_datawidth->value[layer_it][epoch]);
                    }

                    // Forward traces: Weights
                    // Conv: Filters, Channels, Kx, Ky
                    // InPr: Filters, NumInputs
                    base::Array<T> wgt = layer.getWeights();
                    if (wgt.getDimensions() == 2) wgt.reshape_to_4D();
                    computeAvgWidthDataSecondDim(wgt, fw_wgt_avg_width->value[layer_it][epoch],
                                                 fw_wgt_bits_baseline->value[layer_it][epoch],
                                                 fw_wgt_bits_datawidth->value[layer_it][epoch]);
                }

                //Backward
                if (network.getBackward()) {

                    // Backward traces: Input Gradients
                    // Conv: Batch, Channels, Nx, Ny
                    // InPr: Batch, NumInputs
                    if (layer_it != 0) {
                        base::Array<T> in_grad = layer.getInputGradients();
                        if (in_grad.getDimensions() == 2) in_grad.reshape_to_4D();
                        computeAvgWidthDataSecondDim(in_grad, bw_in_grad_avg_width->value[layer_it][epoch],
                                                     bw_in_grad_bits_baseline->value[layer_it][epoch],
                                                     bw_in_grad_bits_datawidth->value[layer_it][epoch]);
                    }

                    // Backward traces: Weight Gradients
                    // Conv: Filters, Channels, Kx, Ky
                    // InPr: NumInputs, Filters
                    base::Array<T> wgt_grad = layer.getWeightGradients();
                    if (wgt_grad.getDimensions() == 2) {
                        wgt_grad.reshape_to_4D();
                        computeAvgWidthDataFirstDim(wgt_grad, bw_wgt_grad_avg_width->value[layer_it][epoch],
                                                    bw_wgt_grad_bits_baseline->value[layer_it][epoch],
                                                    bw_wgt_grad_bits_datawidth->value[layer_it][epoch]);
                    } else {
                        computeAvgWidthDataSecondDim(wgt_grad, bw_wgt_grad_avg_width->value[layer_it][epoch],
                                                     bw_wgt_grad_bits_baseline->value[layer_it][epoch],
                                                     bw_wgt_grad_bits_datawidth->value[layer_it][epoch]);
                    }

                    // Backward traces: Output Gradients
                    // Conv: Batch, Channels, Nx, Ny
                    // InPr: Batch, NumInputs
                    base::Array<T> out_grad = layer.getOutputGradients();
                    if (out_grad.getDimensions() == 2) out_grad.reshape_to_4D();
                    computeAvgWidthDataSecondDim(out_grad, bw_out_grad_avg_width->value[layer_it][epoch],
                                                 bw_out_grad_bits_baseline->value[layer_it][epoch],
                                                 bw_out_grad_bits_datawidth->value[layer_it][epoch]);

                }

            }

        }

        //Dump statistics
        std::string header = "Average Width for " + network_model.getName() + "\n";
        stats.dump_csv(network_model.getName(), network_model.getLayersName(), header, this->QUIET);

    }

    template class DynamicStripesFP<float>;

}
