
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
        long out_x = (act_shape[2] - wgt_shape[2] + 2*padding)/stride + 1;
        long out_y = (act_shape[3] - wgt_shape[3] + 2*padding)/stride + 1;
        long unpadded_x = out_x - 2*padding;
        long unpadded_y = out_y - 2*padding;

        // Set filter batching
        int batches = (int)act.getShape()[1] / (int)wgt_shape[1];
        int it_per_batch = (int)wgt_shape[0] / batches;
        int current_batch = 0, batch_m =0, start_batch = 0;

        // Convolution
        for(int n=0; n<act_shape[0]; n++) {
            for(int m=0; m<wgt_shape[0]; m++) {
                for(int x=0-padding; x<unpadded_x + padding; x++) {
                    for(int y=0-padding; y<unpadded_y + padding; y++) {
                        sum = 0;
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = start_batch; k < wgt_shape[1] + start_batch; k++) {
                                    if(padding) {
                                        int act_x = stride * x + i;
                                        int act_y = stride * y + j;
                                        if ((act_x >= 0 && act_x < out_x) && act_y >= 0 && act_y < out_y)
                                            sum += act.get(n, k, stride * x + i, stride * y + j) *
                                                   wgt.get(m, k - start_batch, i, j);
                                    } else
                                        sum += act.get(n, k, stride * x + i, stride * y + j) *
                                               wgt.get(m, k - start_batch, i, j);
                                }
                            }
                        }
                        if (has_ReLu) sum = ReLU(sum);
                        output_activations.push_back(sum);
                    }
                }
                batch_m++;
                if(batch_m >= it_per_batch) {
                    batch_m = 0;
                    current_batch++;
                    start_batch = (int)wgt_shape[1]*current_batch;
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
        std::vector<size_t> output_shape;
        std::vector<T> output_activations;
        for(unsigned long long units=0; units<layer.getWeights().getShape()[0]; units++) {
            T sum = 0.0;
            for (unsigned long long input_act_num=0; input_act_num<layer.getWeights().getShape()[1]; input_act_num++){
                sum += layer.getActivations().get(input_act_num) * layer.getWeights().get(units,input_act_num);
            }
            if(has_ReLu) sum = ReLU(sum);
            output_activations.push_back(sum);
        }
        // send the results
        output_shape.push_back(1);
        output_shape.push_back(output_activations.size());
        result.set_values(output_activations,output_shape);

    }

    template <typename T>
    void check_values(const Layer<T> &layer, const cnpy::Array<T> &test, const cnpy::Array<T> &result,
            const float min_error = 1.35) {

        std::cout << "Checking values for layer: " << layer.getName() << " of type: "<< layer.getType() << "... ";
        if(test.getMax_index() != result.getMax_index()) {
            std::cout << "SIZE ERROR" << std::endl;
            return;
        }
        int count = 0;
        for(unsigned long long i = 0; i < test.getMax_index(); i++) {
            if(fabsf(test.get(i) - result.get(i)) > min_error)  count++;
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