
#include <core/Simulator.h>

namespace core {

    static inline float ReLU(const float &value) {
        return value < 0 ? 0 : value;
    }

    void Simulator::computeConvolution(const core::Layer &layer, cnpy::Array &result, bool has_ReLu) {
        cnpy::Array wgt = layer.getWeights();
        std::vector<size_t> wgt_shape = wgt.getShape();
        cnpy::Array act = layer.getActivations();
       /* cnpy::Array act;
        std::vector<float> a;
        for(int i=0;i <16;i++)
            a.push_back(i);
        std::vector<size_t> b;
        b[0] = 1;
        b[1] = 1;
        b[2] = 4;
        b[3] = 4;
        act.set_values(a,b);*/
        std::cout << act.get(0, 0, 0, 0) << " ";
      //  std::vector<size_t> act_shape = act.getShape();
        int padding = layer.getPadding();
        int stride = layer.getStride();
        int Kx = layer.getKx();
        int Ky = layer.getKy();

        std::vector<size_t> output_shape;
        int output_0, output_1, output_2, output_3 = 0;
        std::vector<float> output_activations;
        std::cout << act.get(0, 0, 0, 0) << " ";
        //padding
        for(int n_act=0; n_act<act.getShape()[0];n_act++){
            for(int act_channel=0;act_channel<act.getShape()[1];act_channel++){
                for(int j=0;j<padding;j++) {
                    for (int i = 0; i < act.getShape()[3] + 2 * padding; i++) {
                        act.updateActivations(n_act, act_channel, j, i, 0);
                        std::cout << act.get(n_act, act_channel, j, i) << " ";
                        act.updateActivations(n_act, act_channel, j + act.getShape()[2], i, 0);
                    }
                    for (int k = padding; k < act.getShape()[2] + padding; k++) {
                        act.updateActivations(n_act, act_channel, k, 0, 0);
                        act.updateActivations(n_act, act_channel, k, act.getShape()[3] + j, 0);
                    }
                }
            }
            std::cout << act.get(0, 0, 0, 0) << " ";
        }
        act.updateShape(2*padding,2*padding);
      /*  for(int i=0;i<act.getShape()[2];i++) {
            for (int j = 0; j < act.getShape()[3]; j++)
                std::cout << act.get(0, 0, 0, 0) << " ";
            std::cout<< std::endl;
        }*/
/*
        //conv
        for(int filter=0; filter<wgt_shape[0];filter++){
            for(int n_act=0;n_act<act.getShape()[0];n_act++) {
                for (int wgt_channel = 0; wgt_channel < wgt_shape[1]; wgt_channel++) {
                    for (int j = 0; j + Ky < act.getShape()[2]; j += stride) {
                        for (int i = 0; i + Kx < act.getShape()[3]; i += stride)  {
                            float sum = 0;
                            for (int y = j; y < Ky+j; y++) {
                                for (int x = i; x < Kx+i; x++) {
                                    sum += act.get(n_act, wgt_channel, y, x) * wgt.get(filter, wgt_channel, Ky+j-y-1, Kx+i-x-1); //rotate channel horizontally and vertically
                                }
                            }
                            if (has_ReLu)
                                sum = ReLU(sum);
                            //std::cout << sum << std::endl;
                            //result.updateActivations(n_act,wgt_channel,j,i,sum);
                            //result.updateShape(0,1);
                            output_3++;
                            output_activations.push_back(sum);
                        }
                        //result.updateShape(1,0);
                        output_2++;
                    }
                }
            }
        }
        output_shape.push_back(output_0);
        output_shape.push_back(output_1);
        output_shape.push_back(output_2);
        output_shape.push_back(output_3);
        result.set_values(output_activations,output_shape);*/
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
               // check_values(layer,layer.getOutput_activations(),result);
            } else if(layer.getType() == "InnerProduct") {
                //computeInnerProduct(layer, result, has_ReLU);
                //check_values(layer,layer.getOutput_activations(),result);
            }

        }

    }

}