
#include <core/InferenceSimulator.h>

namespace core {

    static inline float ReLU(const float &value) {
        return value < 0 ? 0 : value;
    }

    template <typename T>
    void InferenceSimulator<T>::computeConvolution(const core::Layer<T> &layer, cnpy::Array<T> &result, bool has_ReLu) {
        // Simplify names getting their pointers
        const cnpy::Array<T> &wgt = layer.getWeights();
        const std::vector<size_t> &wgt_shape = wgt.getShape();
        const cnpy::Array<T> &bias = layer.getBias();
        const cnpy::Array<T> &act = layer.getActivations();
        const std::vector<size_t> &act_shape = act.getShape();

        int padding = layer.getPadding();
        int stride = layer.getStride();
        int Kx = layer.getKx();
        int Ky = layer.getKy();

        // Initialize variables
        std::vector<size_t> output_shape;
        std::vector<T> output_activations;
        T sum;

        //Adjust padding
        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (act_shape[2] - wgt_shape[2] + 2*padding)/stride + 1;
        long out_y = (act_shape[3] - wgt_shape[3] + 2*padding)/stride + 1;

        // Set filter grouping
        int groups = (int)act.getShape()[1] / (int)wgt_shape[1];
        int it_per_group = (int)wgt_shape[0] / groups;

        // Convolution
        for(int n=0; n<act_shape[0]; n++) {
            int current_group = 0, group_m = 0, start_group = 0;
            for(int m=0; m<wgt_shape[0]; m++) {
                for(int x=0; x<out_x; x++) {
                    for(int y=0; y<out_y; y++) {
                        sum = bias.get(m);
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = start_group; k < wgt_shape[1] + start_group; k++) {
                                    sum += padded_act.get(n, k, stride * x + i, stride * y + j) *
                                            wgt.get(m, k - start_group, i, j);

                                }
                            }
                        }
                        if (has_ReLu) sum = ReLU(sum);
                        output_activations.push_back(sum);
                    }
                }
                group_m++;
                if(group_m >= it_per_group) {
                    group_m = 0;
                    current_group++;
                    start_group = (int)wgt_shape[1]*current_group;
                }
            }
        }
        output_shape.push_back(act.getShape()[0]);
        output_shape.push_back(wgt.getShape()[0]);
        output_shape.push_back((unsigned)out_x);
        output_shape.push_back((unsigned)out_y);
        result.set_values(output_activations,output_shape);
    }

    template <typename T>
    void InferenceSimulator<T>::computeInnerProduct(const Layer<T> &layer, cnpy::Array<T> &result, bool has_ReLu) {
        // Simplify names getting their pointers
        const cnpy::Array<T> &wgt = layer.getWeights();
        const std::vector<size_t> &wgt_shape = wgt.getShape();
        const cnpy::Array<T> &bias = layer.getBias();
        const cnpy::Array<T> &act = layer.getActivations();
        const std::vector<size_t> &act_shape = act.getShape();


        std::vector<size_t> output_shape;
        std::vector<T> output_activations;

        if(layer.getActivations().getDimensions() == 2) {
            for (uint16_t n = 0; n<act_shape[0]; n++) {
                for (uint16_t m = 0; m<wgt_shape[0]; m++) {
                    T sum = bias.get(m);
                    for (uint16_t k = 0; k<wgt_shape[1]; k++) {
                        sum += act.get(n, k) * wgt.get(m, k);
                    }
                    if (has_ReLu) sum = ReLU(sum);
                    output_activations.push_back(sum);
                }
            }
        } else if (layer.getActivations().getDimensions() == 4) {
            for (uint16_t n = 0; n<act_shape[0]; n++) {
                for (uint16_t m = 0; m<wgt_shape[0]; m++) {
                    T sum = bias.get(m);
                    for (uint16_t k = 0; k<wgt_shape[1]; k++) {
                        int f_dim = (int)(k / (act_shape[2]*act_shape[3]));
                        auto rem = k % (act_shape[2]*act_shape[3]);
                        int s_dim = (int)(rem / act_shape[3]);
                        int t_dim = (int)(rem % act_shape[3]);
                        sum += act.get(n, f_dim, s_dim, t_dim) * wgt.get(m, k);
                    }
                    if (has_ReLu) sum = ReLU(sum);
                    output_activations.push_back(sum);
                }
            }
        }

        output_shape.push_back(act_shape[0]);
        output_shape.push_back(output_activations.size());
        result.set_values(output_activations,output_shape);

    }

    template <typename T>
    void check_values(const Layer<T> &layer, const cnpy::Array<T> &test, const cnpy::Array<T> &result,
            float min_error = .01) {

        std::cout << "Checking values for layer: " << layer.getName() << " of type: "<< layer.getType() << "... ";
        if(test.getMax_index() != result.getMax_index()) {
            std::cout << "SIZE ERROR" << std::endl;
            return;
        }
        int count = 0;
        for(uint32_t i = 0; i < test.getMax_index(); i++) {
            if(fabsf(test.get(i) - result.get(i)) > min_error) count++;
        }
        std::cout << "ERRORS: " << count << " out of " << test.getMax_index() << " absolute error tolerance of "
        << min_error << std::endl;
    }

    template <typename T>
    void InferenceSimulator<T>::run(const Network<T> &network) {
        int index = 0;
        unsigned long num_layers = network.getLayers().size();
        while(index < num_layers) {
            Layer<T> layer = network.getLayers()[index];
            cnpy::Array<T> result;
            index++;

            bool has_ReLU = index < num_layers && network.getLayers()[index].getType() == "ReLU";
            if(has_ReLU) index++;

            if(layer.getType() == "Convolution") {
                computeConvolution(layer, result, has_ReLU);
                check_values(layer, layer.getOutput_activations(), result);
            } else if(layer.getType() == "InnerProduct") {
                computeInnerProduct(layer, result, has_ReLU);
                check_values(layer,layer.getOutput_activations(),result);
            }

        }

    }

    template class InferenceSimulator<float>;

}