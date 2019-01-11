#ifndef DNNSIM_BITTACTICAL_P_H
#define DNNSIM_BITTACTICAL_P_H

#include "BitTactical.h"

namespace core {

    template <typename T>
    class BitTacticalP : public BitTactical<T> {

    private:

        /* Compute number of one bit multiplications given a weights and an activation
         * @param wgt               Weight
         * @param act_layer_rec     Layer precision
         * @return                  Number of one bit multiplications
         */
        uint8_t computeTacticalPBitsPE(uint16_t wgt, uint8_t act_layer_prec);

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats) override;

        /* Compute the timing for a fully-connected layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeInnerProduct(const Layer<T> &layer, sys::Statistics::Stats &stats) override;

        /* Compute the potentials for a convolutional layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats) override;

        /* Compute the potentials for a inner product layer
         * @param layer     Layer for which we want to calculate potentials
         * @param stats     Statistics to fill
         */
        void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats) override;

    public:

        /* Constructor
         * @param _N_COLUMNS            Number of columns
         * @param _N_ROWS               Number of rows
         * @param _LOOKAHEAD_D          Value for scheduler lookahead
         * @param _LOOKASIDE_H          Value for scheduler lookaside
         * @param _SEARCH_SHAPE         Type of search
         */
        BitTacticalP(int _N_COLUMNS, int _N_ROWS, int _LOOKAHEAD_H, int _LOOKASIDE_D, const char _SEARCH_SHAPE) :
                BitTactical<T>(_N_COLUMNS,_N_ROWS,_LOOKAHEAD_H,_LOOKASIDE_D,_SEARCH_SHAPE) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const Network<T> &network) override;

        /* Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const Network<T> &network) override;

    };

}

#endif //DNNSIM_BITTACTICAL_P_H
