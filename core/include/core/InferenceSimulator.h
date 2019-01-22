#ifndef DNNSIM_INFERENCESIMULATOR_H
#define DNNSIM_INFERENCESIMULATOR_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class InferenceSimulator : public Simulator<T> {

    private:

        /* Check if the values simulated are correct with an absolute error tolerance
         * @param layer     Layer we want to check
         * @param test      Original output activations
         * @param result    Calculated output activations
         * @param min_error Minimum error tolerance
         */
        void check_values(const Layer<T> &layer, const cnpy::Array<T> &test, const cnpy::Array<T> &result,
                float min_error = .01);

        /* Compute the output activations for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param result    Output array where output activations are stored
         * @param has_ReLU  Bool to know if the layer has ReLU
         */
        void computeConvolution(const Layer<T> &layer, cnpy::Array<T> &result, bool has_ReLU);

        /* Compute the output activations for a fully connected layers
         * @param layer     Layer for which we want to calculate the outputs
         * @param result    Output array where output activations are stored
         * @param has_ReLU  Bool to know if the layer has ReLU
         */
        void computeInnerProduct(const Layer<T> &layer, cnpy::Array<T> &result, bool has_ReLU);

    public:

        /* Constructor
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        InferenceSimulator(uint8_t _N_THREADS, bool _FAST_MODE) : Simulator<T>(_N_THREADS,_FAST_MODE) {}

        /* Compute the output activations fot all the layers in the network, and check their correctness
         * @param network   Initialised network for the computation
         */
        void run(const Network<T> &network);
        
    };
}

#endif //DNNSIM_INFERENCESIMULATOR_H
