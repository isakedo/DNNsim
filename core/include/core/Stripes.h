#ifndef DNNSIM_STRIPES_H
#define DNNSIM_STRIPES_H

#include "Simulator.h"

#define FC_MULTIPLEX_COLUMNS // Execute each mult-add in a different column
#define WEIGHT_LANES 16 // Number of weight lanes

namespace core {

    template <typename T>
    class Stripes : public Simulator<T> {

    private:

        /* Number of columns */
        const int N_COLUMNS;

        /* Number of rows */
        const int N_ROWS;

        /* Bits per PE */
        const int BITS_PE;

        /* Compute number of one bit multiplications
         * @param layer_prec    Layer precision
         * @return              Number of one bit multiplications
         */
        static inline uint16_t computeStripesBitsPE(uint8_t layer_prec);

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the timing for a 2D convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution2D(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the timing for a fully-connected layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the potentials for a convolutional layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the potentials for a inner product layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

    public:

        /* Constructor
         * @param _N_COLUMNS    Number of columns
         * @param _N_ROWS       Number of rows
         * @param _BITS_PE      Number of bits per PE
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        Stripes(int _N_COLUMNS, int _N_ROWS, int _BITS_PE, uint8_t _N_THREADS, bool _FAST_MODE) :
                Simulator<T>(_N_THREADS,_FAST_MODE), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), BITS_PE(_BITS_PE) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network);

        /* Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const Network<T> &network);

    };

}

#endif //DNNSIM_STRIPES_H
