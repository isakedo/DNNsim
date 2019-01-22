
#include <core/InferenceSimulator.h>

namespace core {

    static inline float ReLU(const float &value) {
        return value < 0 ? 0 : value;
    }

    template <typename T>
    void InferenceSimulator<T>::computeConvolution(const core::Layer<T> &layer, cnpy::Array<T> &result, bool has_ReLu) {

        const cnpy::Array<T> &act = layer.getActivations();
        cnpy::Array<T> wgt = layer.getWeights();
        if(wgt.getDimensions() == 2) wgt.reshape_to_4D();
        const cnpy::Array<T> &bias = layer.getBias();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int act_channels = act_shape[1];
        int Nx = act_shape[2];
        int Ny = act_shape[3];
        if(this->FAST_MODE) batch_size = 1;

        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        int Kx = wgt_shape[2];
        int Ky = wgt_shape[3];

        int padding = layer.getPadding();
        int stride = layer.getStride();

        cnpy::Array<T> padded_act = this->adjustPadding(act,padding);
        long out_x = (Nx - Kx + 2*padding)/stride + 1;
        long out_y = (Ny - Ky + 2*padding)/stride + 1;

        int groups = act_channels / wgt_channels;
        int it_per_group = num_filters / groups;

        // Initialize variables
        std::vector<size_t> output_shape;
        long offset = out_x*out_y*num_filters;
        std::vector<T> output_activations ((unsigned)(batch_size*offset), 0);
        int current_group = 0, group_m =0, start_group = 0;
        T sum;
        int n;

        // Convolution
        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n,current_group,group_m,start_group,sum)
        #endif
        for(n=0; n<batch_size; n++) {
            current_group = 0; group_m = 0; start_group = 0;
            for(int m=0; m<num_filters; m++) {
                for(int x=0; x<out_x; x++) {
                    for(int y=0; y<out_y; y++) {
                        sum = bias.get((unsigned)m);
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = start_group; k < wgt_channels + start_group; k++) {
                                    sum += padded_act.get(n, k, stride * x + i, stride * y + j) *
                                            wgt.get(m, k - start_group, i, j);

                                }
                            }
                        }
                        if (has_ReLu) sum = ReLU(sum);
                        auto pos = m*out_x*out_y + x*out_y + y;
                        output_activations[offset*n+pos] = sum;
                    }
                }
                group_m++;
                if(group_m >= it_per_group) {
                    group_m = 0;
                    current_group++;
                    start_group = wgt_channels*current_group;
                }
            }
        }
        output_shape.push_back((unsigned)batch_size);
        output_shape.push_back((unsigned)num_filters);
        output_shape.push_back((unsigned)out_x);
        output_shape.push_back((unsigned)out_y);
        result.set_values(output_activations,output_shape);
    }

    template <typename T>
    void InferenceSimulator<T>::computeInnerProduct(const Layer<T> &layer, cnpy::Array<T> &result, bool has_ReLu) {

        cnpy::Array<T> act = layer.getActivations();
        if(act.getDimensions() == 4) act.reshape_to_2D();
        const cnpy::Array<T> &wgt = layer.getWeights();
        const cnpy::Array<T> &bias = layer.getBias();

        const std::vector<size_t> &act_shape = act.getShape();
        const std::vector<size_t> &wgt_shape = wgt.getShape();

        int batch_size = act_shape[0];
        int num_filters = wgt_shape[0];
        int wgt_channels = wgt_shape[1];
        if(this->FAST_MODE) batch_size = 1;

        std::vector<size_t> output_shape;
        std::vector<T> output_activations ((unsigned)(batch_size*num_filters), 0);

        T sum;
        int n;

        #ifdef OPENMP
        auto max_threads = omp_get_max_threads();
        omp_set_num_threads(std::min(max_threads,this->N_THREADS));
        #pragma omp parallel for private(n,sum)
        #endif
        for (n = 0; n<batch_size; n++) {
            for (int m = 0; m<num_filters; m++) {
                sum = bias.get((unsigned)m);
                for (int k = 0; k<wgt_channels; k++) {
                    sum += act.get(n, k) * wgt.get(m, k);
                }
                if (has_ReLu) sum = ReLU(sum);
                output_activations[n*num_filters + m] = sum;
            }
        }

        output_shape.push_back((unsigned)batch_size);
        output_shape.push_back((unsigned)num_filters);
        result.set_values(output_activations,output_shape);

    }

    template <typename T>
    void InferenceSimulator<T>::check_values(const Layer<T> &layer, const cnpy::Array<T> &test,
            const cnpy::Array<T> &result, float min_error) {

        std::cout << "Checking values for layer: " << layer.getName() << " of type: "<< layer.getType() << "... ";
        uint32_t test_size = this->FAST_MODE ?
                test.getMax_index() / layer.getActivations().getShape()[0] : test.getMax_index();
        if(test_size != result.getMax_index()) {
            std::cout << "SIZE ERROR" << std::endl;
            return;
        }
        int count = 0;
        for(uint32_t i = 0; i < result.getMax_index(); i++) {
            if(fabsf(test.get(i) - result.get(i)) > min_error) count++;
        }
        std::cout << "ERRORS: " << count << " out of " << result.getMax_index() << " absolute error tolerance of "
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