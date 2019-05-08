#ifndef DNNSIM_DYNAMICSTRIPESFP_H
#define DNNSIM_DYNAMICSTRIPESFP_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class DynamicStripesFP : public Simulator<T> {

    private:

        /* Calculate only the leading bit for dynamic precisions */
        const bool LEADING_BIT;

        /* Calculate only the minor bit for dynamic precisions */
        const bool MINOR_BIT;

        /* Compute the average width along the second dimensions
         * @param data              Input data
         * @param dstr_first_dim    Apply dynamic precision through first dimension instead of second
         * @param avg_width         Average width per group stat
         * @param bits_baseline     Bits of the baseline stat
         * @param bits_datawidth    Bits for the datawidth per group stat
         */
        void computeAvgWidthData(const cnpy::Array<T> &data, bool dstr_first_dim, double &avg_width,
                uint64_t &bits_baseline, uint64_t &bits_datawidth);

        /* Compute the average width for a layer
         * @param network       Network we want to check
         * @param layer_it      Index pointing the current layer in the network
         * @param stats         Statistics to fill
         * @param epoch		    Current epoch
		 * @param epochs        Number of epochs
         */
        void computeAvgWidthLayer(const Network<T> &network, int layer_it, sys::Statistics::Stats &stats,
                int epoch, int epochs);

    public:

        /* Constructor
         * @param _LEADING_BIT     Calculate only the leading bit for dynamic precisions
         * @param _MINOR_BIT       Calculate only the minor bit for dynamic precisions
         * @param _N_THREADS       Number of parallel threads for multi-threading execution
         * @param _FAST_MODE       Enable fast mode to simulate only one image
         */
        DynamicStripesFP(bool _LEADING_BIT, bool _MINOR_BIT, uint8_t _N_THREADS, bool _FAST_MODE) :
                Simulator<T>(_N_THREADS,_FAST_MODE), LEADING_BIT(_LEADING_BIT), MINOR_BIT(_MINOR_BIT) {}

        /* Calculate the average width in the network transformed to sign-magnitude
         * @param network   Network we want to check
		 * @param stats		Shared stats for the epochs
         * @param epoch		Current epoch
		 * @param epochs    Number of epochs
         */
        void average_width(const Network<T> &network, sys::Statistics::Stats &stats, int epoch, int epochs);

    };

}

#endif //DNNSIM_DYNAMICSTRIPESFP_H
