
#include <core/Simulator.h>

namespace core {

    static inline float ReLU(const float &value) {
        return value < 0 ? 0 : value;
    }

    void Simulator::computeConvolution(const core::Layer &layer, cnpy::Array &result, bool has_ReLu) {

    }

    void Simulator::computeInnerProduct(const Layer &layer, cnpy::Array &result, bool has_ReLu) {
        layer.getActivations().get(0,0,0,0);
        layer.getWeights().get(0,0,0,0);
        layer.getActivations().getDimensions();
        layer.getActivations().getShape()[0];
        std::vector<size_t> output_shape;
        std::vector<float> output_activations;
        //std::cout<< "\n" <<layer.getActivations().getShape()[0]<< "\n" <<layer.getActivations().getShape()[1]<< "\n" <<layer.getActivations().getShape()[2]<< "\n" <<layer.getActivations().getShape()[3] ;
        for(unsigned long long units=0; units<layer.getWeights().getShape()[0]; units++) {

            float sum = 0.0;
            for (unsigned long long input_act_num=0; input_act_num<layer.getWeights().getShape()[1]; input_act_num++){
                    // No relu t=yet
                sum += layer.getActivations().get(input_act_num) * layer.getWeights().get(units,input_act_num);
             //   std::cout<< "\n"<< "activation counter"<<input_act_num;
            }
            output_activations.push_back(sum);
          //  std::cout<< "\n"<< "Your final result"<<output_activations[units];


        }
        // send the results
        output_shape.push_back(1);
        output_shape.push_back(output_activations.size());
        result.set_values(output_activations,output_shape);

    }

    void check_values(const Layer &layer, const cnpy::Array &test, const cnpy::Array &result,
            const float min_error = 0.001) {

        std::cout << "Checking values for layer: " << layer.getName() << " of type: "<< layer.getType() << "... ";
        if(test.getMax_index() != result.getMax_index()) {
            std::cout << "ERROR" << std::endl;
            return;
        }
        int count = 0;
        for(unsigned long long i = 0; i < test.getMax_index(); i++) {
            if(fabsf(test.get(i) - result.get(i)) > min_error) {
              //  std::cout << "ERROR" << std::endl;
                std::cout << "target: " << test.get(i) << "result: " << result.get(i) << std::endl;
                count++;
             //   return;
            }
        }
        //std::cout << "GOOD" << std::endl;
        std::cout << "ERRORS: " << count << " out of " << test.getMax_index() << " error tolerance: " << min_error << std::endl;
    }

    void Simulator::run(const Network &network) {
        int index = 0;
        unsigned long num_layers = network.getLayers().size();
        while(index < num_layers) {
            Layer layer = network.getLayers()[index];
            cnpy::Array result;
            index++;

            bool has_ReLU = index < num_layers && network.getLayers()[index].getType() == "ReLU";
            if(has_ReLU) index++;

            if(layer.getType() == "Convolution") {
                computeConvolution(layer, result, has_ReLU);
                check_values(layer,layer.getOutput_activations(),result);
            } else if(layer.getType() == "InnerProduct") {
                computeInnerProduct(layer, result, has_ReLU);
                check_values(layer,layer.getOutput_activations(),result);
            }

        }

    }

}