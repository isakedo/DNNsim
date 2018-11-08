#ifndef DNNSIM_TIMINGSIMULATOR_H
#define DNNSIM_TIMINGSIMULATOR_H


#include <sys/common.h>
#include <sys/Statistics.h>
#include <cnpy/Array.h>
#include "Layer.h"
#include "Network.h"

namespace core {

    template <typename T>
    class Simulator {

    protected:

        /* Return a vector zero padded
         * @param array     Array we want to apply padding
         * @param padding   Padding of the layer
         * @return          Array zero padded
         */
        cnpy::Array<T> adjustPadding(const cnpy::Array<T> &array, int padding);

        /* Return the optimal encoding for the given value
         * @param value     Value we want to encode
         * @param mag       Magnitude of the value
         * @param prec      Precision of the value
         * @return          Value with the optimal encoding
         */
        uint16_t booth_encoding(uint16_t value, int mag, int prec);

    private:

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        virtual void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats) = 0;

    public:

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        virtual void run(const Network<T> &network) = 0;

    };

}

#endif //DNNSIM_TIMINGSIMULATOR_H
