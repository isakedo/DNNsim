
#include <core/Simulator.h>

namespace core {

    static inline float ReLU(const float &value) {
        return value < 0 ? 0 : value;
    }

    void Simulator::computeConvolution(const core::Layer &layer, cnpy::Array &result, bool has_ReLu) {
        const cnpy::Array &wgt = layer.getWeights();  //Avoid copying the vector
        const std::vector<size_t> &wgt_shape = wgt.getShape();
        const cnpy::Array &act = layer.getActivations();
        //  std::vector<size_t> act_shape = act.getShape();
        int padding = layer.getPadding();
        int stride = layer.getStride();
        int Kx = layer.getKx();
        int Ky = layer.getKy();

        std::vector<size_t> output_shape;

       /* cnpy::Array b;
        std::vector<float> a;
        for(int i=0;i<16;i++)
            a.push_back(i);
        std::vector<size_t>
        b.set_values(a,)
         */
        int output = 0;
        // int output_2 = 0;
        //  int output_3 = 0;
        // int output_0 = 0;
        std::vector<float> output_activations;
        // std::cout << act.get(0, 0, 0, 0) << " ";
        //conv
        double sum;
        //padding =0 ;
        std::cout << wgt_shape[0] << " "  << wgt_shape[1] << " "<< wgt_shape[2] << " "<< wgt_shape[3] << " " << Ky <<" " <<  Kx << " \n";
        std::cout << act.getShape()[0] << " "  << act.getShape()[1] << " "<< act.getShape()[2] << " "<< act.getShape()[3] << " \n";
        std::cout << layer.getOutput_activations().getShape()[0] << " "  << layer.getOutput_activations().getShape()[1] << " "<< layer.getOutput_activations().getShape()[2] << " "<< layer.getOutput_activations().getShape()[3] << " \n";

        for(int filter=0; filter<wgt_shape[0];filter++){
            for(int n_act=0;n_act<act.getShape()[0];n_act++) {
                    for (int i = 0-padding; i + Ky  <= act.getShape()[2]+padding; i += stride)  {
                        for (int j = 0-padding; j + Kx  <= act.getShape()[3]+padding; j += stride) {
                        sum = 0;
                        for (int wgt_channel = 0; wgt_channel < wgt_shape[1]; wgt_channel++) {
                            int counter =0;
                            for (int x = i; x < Ky + i; x++) {
                                for (int y = j; y < Kx + j; y++) {
                                    sum += ((y >= 0 && y < act.getShape()[3]) && (x >= 0 && x < act.getShape()[2])) ? (act.get(n_act, wgt_channel, x,y) *wgt.get(filter, wgt_channel,x-i,y-j)): 0; //padding //rotate channel horizontally and vertically
                                    //std::cout << counter;
                                   // std::cout << ": " << sum << std::endl;
                                    counter++;
                                }
                            }
                            //Sstd::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
                        }
                        if (has_ReLu)
                            sum = ReLU(sum);
                        //std::cout << sum << std::endl;
                        output++;
                        output_activations.push_back(sum);
                        if(fabsf(layer.getOutput_activations().get(output) - sum) > 0.5) {
                            std::cout << "ERROR" << std::endl;
                                std::cout << "target: " << layer.getOutput_activations().get(output) << "result: " << sum << std::endl;
                                printf("%d %d %d %d\n",filter,n_act,i,j);
                                //return;
                                //exit(5);
                            }
                        //result.updateActivations(n_act,wgt_channel,j,i,sum);
                        //result.updateShape(0,1);

                        //output_activations.push_back(sum);
                        //result.updateShape(1,0);
                        //output_2++;
                    }
                    //  output_1++;
                }
                // output_0++;
            }
        }
    /*    int unpadded_x = layer.getOutput_activations().getShape()[2] - 2*padding;
        int unpadded_y = layer.getOutput_activations().getShape()[3] - 2*padding;
        for(int n=0;n<act.getShape()[0];n++) {
            for(int m=0; m<wgt_shape[0];m++) {
                for(int x=0-padding; x<unpadded_x + padding; x++) {
                    for(int y=0-padding; y<unpadded_y + padding; y++) {
                        sum = 0;
                        for (int i = 0; i < Kx; i++) {
                            for (int j = 0; j < Ky; j++) {
                                for (int k = 0; k < wgt_shape[1]; k++) {
                                    sum += ((stride *x >= 0 && stride *x < unpadded_x) && (stride *y >= 0 && stride *y < unpadded_y)) ? act.get(n,k,stride * x + i, stride * y + j) * wgt.get(m,k,i,j) : 0;
                                }
                            }
                        }
                        if (has_ReLu)
                            sum = ReLU(sum);
                        output++;
                        output_activations.push_back(sum);
                    }
                }
            }
        }*/
        output_shape.push_back(0);
        output_shape.push_back(0);
        output_shape.push_back(0);
        output_shape.push_back(0);
       // std::cout << output << std::endl;
       // std::cout << output_activations.size() << std::endl;
        result.set_values(output_activations,output_shape);
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
            const float min_error = 0.5) {

        std::cout << "Checking values for layer: " << layer.getName() << " of type: "<< layer.getType() << "... ";
        if(test.getMax_index() != result.getMax_index()) {
            std::cout << test.getMax_index() << " " << result.getMax_index() << "\n";
            std::cout << "ERROR" << std::endl;
            return;
        }
        int count = 0;
        for(unsigned long long i = 0; i < test.getMax_index(); i++) {
            if(fabsf(test.get(i) - result.get(i)) > min_error) {
               // std::cout << "ERROR" << std::endl;
               // std::cout << "target: " << test.get(i) << "result: " << result.get(i) << std::endl;
                count++;
                //return;
            }
        }
        //std::cout << "GOOD" << std::endl;
      //  std::cout << "ERRORS: " << count << " out of " << test.getMax_index() << " error tolerance: " << min_error << std::endl;
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
                if(layer.getName() == "conv2") {
                    computeConvolution(layer, result, has_ReLU);
                    check_values(layer, layer.getOutput_activations(), result);
                }
            } else if(layer.getType() == "InnerProduct") {
                //computeInnerProduct(layer, result, has_ReLU);
                //check_values(layer,layer.getOutput_activations(),result);
            }

        }

    }

}