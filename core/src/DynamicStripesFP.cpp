
#include <core/DynamicStripesFP.h>

namespace core {

    /* AVERAGE WIDTH */

    template <typename T>
    void DynamicStripesFP<T>::computeAvgWidthDataFirstDim(const cnpy::Array<T> &data, double &avg_width,
            uint64_t &bits_baseline, uint64_t &bits_datawidth) {

        const int mask = 0x80;

        const std::vector<size_t> &data_shape = data.getShape();
        
        int first_dim = data_shape[0];
        int second_dim = data_shape[1];
        int third_dim = data_shape[2];
        int fourth_dim = data_shape[3];

        std::vector<double> data_width;
        uint64_t data_bits_datawidth = 0;
        for(int i = 0; i < first_dim; i += 16) {
            for (int j = 0; j < second_dim; j++) {
                for (int k = 0; k < third_dim; k++) {
                    for (int l = 0; l < fourth_dim; l++) {

                        uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                        for(int dstr = i; dstr < std::min(i + 16,first_dim); dstr++) {
                            
                            auto data_float = data.get(dstr, j, k, l);

                            if(data_float == 0) continue;

                            auto data_bfloat = this->split_bfloat16(data_float);
                            auto biased_exponent = std::get<1>(data_bfloat);

                            int unbiased_exponent = biased_exponent - 127;
                            auto exponent = this->sign_magnitude((short)unbiased_exponent, mask);

                            bool neg = false;
                            if((exponent & mask) != 0) {
                                exponent = exponent & ~(uint16_t)mask;
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
    void DynamicStripesFP<T>::computeAvgWidthDataSecondDim(const cnpy::Array<T> &data, double &avg_width,
            uint64_t &bits_baseline, uint64_t &bits_datawidth) {

        const int mask = 0x80;

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
                        for(int dstr = j; dstr < std::min(j + 16,second_dim); dstr++) {

                            auto data_float = data.get(i, dstr, k, l);

                            if(data_float == 0) continue;

                            auto data_bfloat = this->split_bfloat16(data_float);
                            auto biased_exponent = std::get<1>(data_bfloat);

                            int unbiased_exponent = biased_exponent - 127;
                            auto exponent = this->sign_magnitude((short)unbiased_exponent, mask);

                            bool neg = false;
                            if((exponent & mask) != 0) {
                                exponent = exponent & ~(uint16_t)mask;
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
    void DynamicStripesFP<T>::computeAvgWidthDataSeq2Seq(const cnpy::Array<T> &data, double &avg_width,
            uint64_t &bits_baseline, uint64_t &bits_datawidth) {

        const int mask = 0x80;

        const std::vector<size_t> &data_shape = data.getShape();

        int first_dim = data_shape[0];
        int second_dim = data_shape[1];
        int third_dim = data_shape[2];

        std::vector<double> data_width;
        uint64_t data_bits_datawidth = 0;
        for(int i = 0; i < first_dim; i++) {
            for (int j = 0; j < second_dim; j++) {
                for (int k = 0; k < third_dim; k += 16) {

                    uint8_t max_bit = 0, min_bit = 16, non_zeroes = 0;
                    for(int dstr = k; dstr < std::min(k + 16,third_dim); dstr++) {

                        auto data_float = data.get(i, j, dstr);

                        if(data_float == 0) continue;

                        auto data_bfloat = this->split_bfloat16(data_float);
                        auto biased_exponent = std::get<1>(data_bfloat);

                        int unbiased_exponent = biased_exponent - 127;
                        auto exponent = this->sign_magnitude((short)unbiased_exponent, mask);

                        bool neg = false;
                        if((exponent & mask) != 0) {
                            exponent = exponent & ~(uint16_t)mask;
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
    void DynamicStripesFP<T>::computeAvgWidthLayer(const Network<T> &network, int layer_it,
            sys::Statistics::Stats &stats, const int epoch, const int epochs) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        const Layer<T> &layer = network.getLayers()[layer_it];

        // Stats
        if(epoch == 0) {
            stats.layers.push_back(layer.getName());
            stats.training_time.emplace_back(std::vector<std::chrono::duration<double>>
                    ((uint64_t)epochs,std::chrono::duration<double>()));

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

        if(network.getForward()) {

            // Forward traces: Activations
            // Conv: Batch, Channels, Nx, Ny
            // InPr: Batch, NumInputs
            cnpy::Array<T> act = layer.getActivations();
            if(layer.getType() == "Decoder" || layer.getType() == "Encoder") {
                computeAvgWidthDataSeq2Seq(act, stats.fw_act_avg_width[layer_it][epoch],
                                           stats.fw_act_bits_baseline[layer_it][epoch],
                                           stats.fw_act_bits_datawidth[layer_it][epoch]);
            } else {
                if (act.getDimensions() == 2) act.reshape_to_4D();
                computeAvgWidthDataSecondDim(act, stats.fw_act_avg_width[layer_it][epoch],
                                             stats.fw_act_bits_baseline[layer_it][epoch],
                                             stats.fw_act_bits_datawidth[layer_it][epoch]);
            }

            // Forward traces: Weights
            // Conv: Filters, Channels, Kx, Ky
            // InPr: Filters, NumInputs
            cnpy::Array<T> wgt = layer.getWeights();
            if (wgt.getDimensions() == 2) wgt.reshape_to_4D();
            computeAvgWidthDataSecondDim(wgt, stats.fw_wgt_avg_width[layer_it][epoch],
                                         stats.fw_wgt_bits_baseline[layer_it][epoch],
                                         stats.fw_wgt_bits_datawidth[layer_it][epoch]);

        }

        if(network.getBackward()) {

            // Backward traces: Input Gradients
            // Conv: Batch, Channels, Nx, Ny
            // InPr: Batch, NumInputs
            if (layer_it != 0) {
                cnpy::Array<T> in_grad = layer.getInputGradients();
                if (in_grad.getDimensions() == 2) in_grad.reshape_to_4D();
                computeAvgWidthDataSecondDim(in_grad, stats.bw_in_grad_avg_width[layer_it][epoch],
                                             stats.bw_in_grad_bits_baseline[layer_it][epoch],
                                             stats.bw_in_grad_bits_datawidth[layer_it][epoch]);
            }

            // Backward traces: Weight Gradients
            // Conv: Filters, Channels, Kx, Ky
            // InPr: NumInputs, Filters
            cnpy::Array<T> wgt_grad = layer.getWeightGradients();
            if (wgt_grad.getDimensions() == 2) {
                wgt_grad.reshape_to_4D();
                computeAvgWidthDataFirstDim(wgt_grad, stats.bw_wgt_grad_avg_width[layer_it][epoch],
                                            stats.bw_wgt_grad_bits_baseline[layer_it][epoch],
                                            stats.bw_wgt_grad_bits_datawidth[layer_it][epoch]);
            } else {
                computeAvgWidthDataSecondDim(wgt_grad, stats.bw_wgt_grad_avg_width[layer_it][epoch],
                                             stats.bw_wgt_grad_bits_baseline[layer_it][epoch],
                                             stats.bw_wgt_grad_bits_datawidth[layer_it][epoch]);
            }

            // Backward traces: Output Gradients
            // Conv: Batch, Channels, Nx, Ny
            // InPr: Batch, NumInputs
            cnpy::Array<T> out_grad = layer.getOutputGradients();
            if (out_grad.getDimensions() == 2) out_grad.reshape_to_4D();
            computeAvgWidthDataSecondDim(out_grad, stats.bw_out_grad_avg_width[layer_it][epoch],
                                         stats.bw_out_grad_bits_baseline[layer_it][epoch],
                                         stats.bw_out_grad_bits_datawidth[layer_it][epoch]);

        }

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
            stats.arch += (LEADING_BIT ? "_LB" : "");
            stats.arch += (MINOR_BIT && !LEADING_BIT ? "_MB" : "");
        }

        for(int layer_it = 0; layer_it < network.getLayers().size(); layer_it++) {
            const Layer<T> &layer = network.getLayers()[layer_it];
            if(layer.getType() == "Convolution" || layer.getType() == "InnerProduct" || layer.getType() == "Encoder" ||
                    layer.getType() == "Decoder")
                computeAvgWidthLayer(network, layer_it, stats, epoch, epochs);
        }

    }

    template class DynamicStripesFP<float>;

}
