#ifndef DNNSIM_SCNN_H
#define DNNSIM_SCNN_H

#include "Simulator.h"

#define ZERO_COUNT // Count zeroes as 1 cycle

typedef std::vector<std::tuple<int,int,int>> wgt_idxMap;

namespace core {

    template <typename T>
    class SCNN : public Simulator<T> {

    protected:

        /* Number of PE columns */
        const uint32_t Wt;

        /* Number of PE rows */
        const uint32_t Ht;

        /* Column multipliers per PE */
        const uint32_t I;

        /* Row multipliers per PE */
        const uint32_t F;

        /* Output accumulator size */
        const int out_acc_size;

        /* Number of banks */
        const int BANKS;

        /* Calculate in which bank the output activation is mapped
         * @param k         Filter
         * @param x         X position
         * @param y         W position
         * @return          Accumulator bank index
         */
        int map_accumulator(uint32_t k, uint32_t x, uint32_t y);

    private:

        typedef std::vector<std::tuple<int,int>> act_idxMap;

        struct PE_stats {
            uint32_t cycles = 0;
            uint32_t mults = 0;
            uint32_t idle_conflicts = 0;
            uint32_t accumulator_updates = 0;
            uint32_t i_loop = 0;
            uint32_t f_loop = 0;
        };

        /* Compute number of one bit multiplications given a weight and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @param network_bits  Max bits network
         * @return              Number of one bit multiplications
         */
        uint16_t computeSCNNBitsPE(T act, T wgt, int network_bits);

        /* Compute SCNN processing engine
         * @param W         Width of the output activations
         * @param H         Height of the output activations
         * @param stride    Stride for the layer
         * @param act       1D activations queue with linearized activations indexes to be processed
         * @param wgt       1D weights queue with linearized activations indexes to be processed
         * @return          Return stats for the given PE
         */
        PE_stats computeSCNNPE(uint64_t W, uint64_t H, int stride, const act_idxMap &act, const wgt_idxMap &wgt);

        /* Compute SCNN tile
         * @param n         Number of batch
         * @param ct        Channel to be processed within a filter
         * @param ck        Channel offset for per group filters
         * @param kc        First filter to be processed
         * @param tw        Width range of the output activations to be processed
         * @param th        Height range of the output activations to be processed
         * @param X         Width of the activations
         * @param Y         Height of the activations
         * @param Kc        Max number of channels per group
         * @param K         Max number of activations channels
         * @param W         Width of the output activations
         * @param H         Height of the output activations
         * @param R         Kernel width for the weights
         * @param S         Kernel height for the weights
         * @param stride    Stride for the layer
         * @param padding   Padding for the layer
         * @param act       Activations for the layer
         * @param wgt       Weights for the layer
         * @param stats     Statistics to fill
         */
        void computeSCNNTile(int n, int ct, int ck, int kc, int tw, int th, uint64_t X, uint64_t Y, int Kc, uint64_t K,
                uint64_t W, uint64_t H, uint64_t R, uint64_t S, int stride, int padding, const cnpy::Array<T> &act,
                const cnpy::Array<T> &wgt, sys::Statistics::Stats &stats);

        /* Compute the timing for a layer
         * @param layer     Layer for which we want to calculate the outputs
         * @param stats     Statistics to fill
         */
        void computeSCNNLayer(const Layer<T> &layer, sys::Statistics::Stats &stats);

        /* Compute the potentials for a convolutional layer
         * @param layer         Layer for which we want to calculate potentials
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        virtual void computePotentialsConvolution(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
                int network_bits);

        /* Compute the potentials for a inner product layer
         * @param layer         Layer for which we want to calculate potentials
         * @param stats         Statistics to fill
         * @param network_bits  Max bits network
         */
        virtual void computePotentialsInnerProduct(const core::Layer<T> &layer, sys::Statistics::Stats &stats,
                int network_bits);

    public:

        /* Constructor
         * @param _Wt           Number of PE columns
         * @param _Ht           Number of PE rows
         * @param _I            Column multipliers per PE
         * @param _F            Row multipliers per PE
         * @param _out_acc_size Output accumulator size
         * @param _BANKS        Number of banks
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         */
        SCNN(uint32_t _Wt, uint32_t _Ht, uint32_t _I, uint32_t _F, uint32_t _out_acc_size, uint32_t _BANKS,
            uint8_t _N_THREADS, bool _FAST_MODE) : Simulator<T>(_N_THREADS, _FAST_MODE), Wt(_Wt), Ht(_Ht), I(_I), F(_F),
            out_acc_size(_out_acc_size), BANKS(_BANKS) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        virtual void run(const Network<T> &network);

        /* Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        virtual void potentials(const Network<T> &network);

    };

}

#endif //DNNSIM_SCNN_H
