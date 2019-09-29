#ifndef DNNSIM_DYNAMICSTRIPESFP_H
#define DNNSIM_DYNAMICSTRIPESFP_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class DynamicStripesFP : public Simulator<T> {

    private:

        /** Calculate only the leading bit for dynamic precisions */
        const bool LEADING_BIT;

        /** Calculate only the minor bit for dynamic precisions */
        const bool MINOR_BIT;

        /** Compute the average width along the first dimension
         * @param data              Input data
         * @param avg_width         Average width per group stat
         * @param bits_baseline     Bits of the baseline stat
         * @param bits_datawidth    Bits for the datawidth per group stat
         */
        void computeAvgWidthDataFirstDim(const base::Array<T> &data, double &avg_width, uint64_t &bits_baseline,
                uint64_t &bits_datawidth);

        /** Compute the average width along the second dimension
         * @param data              Input data
         * @param avg_width         Average width per group stat
         * @param bits_baseline     Bits of the baseline stat
         * @param bits_datawidth    Bits for the datawidth per group stat
         */
        void computeAvgWidthDataSecondDim(const base::Array<T> &data, double &avg_width, uint64_t &bits_baseline,
                uint64_t &bits_datawidth);

        /** Compute the average width along the third dimension for Seq2Seq models
         * @param data              Input data
         * @param avg_width         Average width per group stat
         * @param bits_baseline     Bits of the baseline stat
         * @param bits_datawidth    Bits for the datawidth per group stat
         */
        void computeAvgWidthDataSeq2Seq(const base::Array<T> &data, double &avg_width, uint64_t &bits_baseline,
                uint64_t &bits_datawidth);

    public:

        /** Constructor
         * @param _LEADING_BIT     Calculate only the leading bit for dynamic precisions
         * @param _MINOR_BIT       Calculate only the minor bit for dynamic precisions
         * @param _N_THREADS       Number of parallel threads for multi-threading execution
         * @param _FAST_MODE       Enable fast mode to simulate only one image
         * @param _QUIET           Avoid std::out messages
         * @param _CHECK            Check the correctness of the simulations
         */
        DynamicStripesFP(bool _LEADING_BIT, bool _MINOR_BIT, uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET,
                bool _CHECK) : Simulator<T>(_N_THREADS,_FAST_MODE,_QUIET,_CHECK), LEADING_BIT(_LEADING_BIT),
                MINOR_BIT(_MINOR_BIT) {}

        /** Calculate the average width in the network transformed to sign-magnitude
         * @param simulate  Simulate configuration
         * @param epochs    Number of epochs
         */
        void average_width(const sys::Batch::Simulate &simulate, int epochs);

    };

}

#endif //DNNSIM_DYNAMICSTRIPESFP_H
