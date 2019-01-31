#ifndef DNNSIM_SCNN_H
#define DNNSIM_SCNN_H

#include "Simulator.h"

#define ZERO_COUNT // Count zeroes as 1 cycle

typedef std::vector<std::tuple<int,int,int>> idxMap;

namespace core {

    template <typename T>
    class SCNN : public Simulator<T> {

    private:

        /* Number of PE columns */
        const int Wt;

        /* Number of PE rows */
        const int Ht;

        /* Number of max filters per PE */
        const int Kt;

        /* Column multipliers per PE */
        const int I;

        /* Row multipliers per PE */
        const int F;

        /* Output accumulator size */
        const int out_acc_size;

        struct PE_stats {
            uint32_t cycles = 0;
            uint32_t mults = 0;
            uint32_t idle_conflicts = 0;
            uint32_t accumulator_updates = 0;
            uint32_t i_loop = 0;
            uint32_t f_loop = 0;
        };

        int map_accumulator(int k, int x, int y);

        /* Compute number of one bit multiplications given a weight and an activation
         * @param act       Activation
         * @param wgt       Weight
         * @return          Number of one bit multiplications
         */
        uint16_t computeSCNNBitsPE(T act, T wgt);

        PE_stats computeSCNNPE(int W, int H, int stride, int padding, const idxMap &act, const idxMap &wgt);

        void computeSCNNTile(int n, int ct, int ck, int kc, int tw, int th, int X, int Y, int Kc, int K, int W, int H,
                int R, int S, int stride, int padding, const cnpy::Array<T> &act, const cnpy::Array<T> &wgt,
                sys::Statistics::Stats &stats);

        /* Compute the timing for a convolutional layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeConvolution(const Layer<T> &layer, sys::Statistics::Stats &stats);

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
        virtual void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats);

    public:

        /* Constructor
         * @param _Wt           Number of PE columns
         * @param _Ht           Number of PE rows
         * @param _Kt           Number of max filters per PE
         * @param _I            Column multipliers per PE
         * @param _F            Row multipliers per PE
         * @param _out_acc_size Output accumulator size
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        SCNN(int _Wt, int _Ht, int _Kt, int _I, int _F, int _out_acc_size, uint8_t _N_THREADS, bool _FAST_MODE) :
            Simulator<T>(_N_THREADS, _FAST_MODE), Wt(_Wt), Ht(_Ht), Kt(_Kt), I(_I), F(_F), out_acc_size(_out_acc_size){}

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

#endif //DNNSIM_SCNN_H
