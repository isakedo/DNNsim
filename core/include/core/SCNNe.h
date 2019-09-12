#ifndef DNNSIM_SCNNE_H
#define DNNSIM_SCNNE_H

#include "SCNN.h"

#define BOOTH_ENCODING

namespace core {

    template <typename T>
    class SCNNe : public SCNN<T> {

    private:

        typedef std::vector<std::tuple<int,int,uint8_t>> act_idxMap;

        /* Number of bits in series that the PE process */
        const uint32_t PE_SERIAL_BITS;

        struct PE_stats {
            uint32_t cycles = 0;
            uint32_t mults = 0;
            uint32_t idle_conflicts = 0;
            uint32_t idle_column_cycles = 0;
            uint32_t column_stalls = 0;
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
        uint16_t computeSCNNeBitsPE(T act, T wgt, int network_bits);

        /* Compute SCNNe processing engine
         * @param W         Width of the output activations
         * @param H         Height of the output activations
         * @param stride    Stride for the layer
         * @param act       1D activations queue with linearized activations indexes to be processed
         * @param wgt       1D weights queue with linearized activations indexes to be processed
         * @return          Return stats for the given PE
         */
        PE_stats computeSCNNePE(uint64_t W, uint64_t H, int stride, const act_idxMap &act, const wgt_idxMap &wgt);

        /* Compute SCNNp tile
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
        void computeSCNNeTile(int n, int ct, int ck, int kc, int tw, int th, uint64_t X, uint64_t Y, int Kc, uint64_t K,
                uint64_t W, uint64_t H, uint64_t R, uint64_t S, int stride, int padding, const cnpy::Array<T> &act,
                const cnpy::Array<T> &wgt, sys::Statistics::Stats &stats);

    public:

        /* Constructor
         * @param _Wt               Number of PE columns
         * @param _Ht               Number of PE rows
         * @param _I                Column multipliers per PE
         * @param _F                Row multipliers per PE
         * @param _out_acc_size     Output accumulator size
         * @param _BANKS            Number of banks
         * @param _PE_SERIAL_BITS   Number of bits in series that the PE process
         * @param _N_THREADS        Number of parallel threads for multi-threading execution
         * @param _FAST_MODE        Enable fast mode to simulate only one image
         * @param _QUIET            Avoid std::out messages
         */
        SCNNe(uint32_t _Wt, uint32_t _Ht, uint32_t _I, uint32_t _F, uint32_t _out_acc_size, uint32_t _BANKS,
                uint32_t _PE_SERIAL_BITS, uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET) :
                SCNN<T>(_Wt,_Ht,_I,_F,_out_acc_size,_BANKS,_N_THREADS,_FAST_MODE,_QUIET),
                PE_SERIAL_BITS(_PE_SERIAL_BITS) {}

        /* Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const base::Network<T> &network) override;

        /* Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        void potentials(const base::Network<T> &network) override;

    };

}

#endif //DNNSIM_SCNN_H
