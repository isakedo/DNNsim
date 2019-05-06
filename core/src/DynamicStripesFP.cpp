
#include <core/DynamicStripesFP.h>

namespace core {

    /* AVERAGE WIDTH */

    template <typename T>
    void DynamicStripesFP<T>::computeAvgWidthLayer(const Network<T> &network, int layer_it,
            sys::Statistics::Stats &stats, const int network_bits, const int epoch, const int epochs) {

        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        const Layer<T> &layer = network.getLayers()[layer_it];

        if(epoch == 0) {
            stats.layers.push_back(layer.getName());
            stats.training_time.emplace_back(std::vector<std::chrono::duration<double>>
                    (epochs,std::chrono::duration<double>()));
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
        }

        for(int layer_it = 0; layer_it < network.getLayers().size(); layer_it++) {
            const Layer<T> &layer = network.getLayers()[layer_it];
            if(layer.getType() == "Convolution" || layer.getType() == "InnerProduct")
                computeAvgWidthLayer(network, layer_it, stats, network.getNetwork_bits(), epoch, epochs);
        }

    }

    template class DynamicStripesFP<float>;

}
