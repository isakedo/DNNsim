#ifndef DNNSIM_SCNN_H
#define DNNSIM_SCNN_H

#include "Simulator.h"

#define ZERO_COUNT // Count zeroes as 1 cycle

typedef std::vector<std::tuple<int,int,int>> wgt_idxMap;

typedef std::vector<std::tuple<int,int,int,uint64_t>> wgt_idxAddrMap;
typedef std::vector<std::vector<std::vector<std::vector<wgt_idxAddrMap>>>> wgt_addr_queue;

typedef std::vector<std::tuple<int,int,uint16_t,uint64_t>> act_idxAddrMap;
typedef std::vector<std::vector<std::vector<std::vector<std::vector<act_idxAddrMap>>>>> act_addr_queue;

namespace core {

    /**
     * SCNN simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class SCNN : public Simulator<T> {

    protected:

        /** Number of PE columns */
        const uint32_t Wt;

        /** Number of PE rows */
        const uint32_t Ht;

        /** Column multipliers per PE */
        const uint32_t I;

        /** Row multipliers per PE */
        const uint32_t F;

        /** Output accumulator size */
        const int out_acc_size;

        /** Number of banks */
        const int BANKS;

        /** Simulate baseline only */
        const bool BASELINE;

        /** Calculate in which bank the output activation is mapped
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

        struct Tile_stats {
            uint32_t cycles = 0;
            uint32_t dense_cycles = 0;
            uint32_t mults = 0;
            uint32_t idle_bricks = 0;
            uint32_t idle_conflicts = 0;
            uint32_t idle_pe = 0;
            uint32_t weight_buff_reads = 0;
            uint32_t act_buff_reads = 0;
            uint32_t accumulator_updates = 0;
            uint32_t i_loop = 0;
            uint32_t f_loop = 0;
            uint32_t offchip_weight_reads = 0;

        };

        /** Compute number of one bit multiplications given a weight and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @param network_bits  Max bits network
         * @return              Number of one bit multiplications
         */
        uint16_t computeSCNNBitsPE(T act, T wgt, int network_bits);

        /** Compute SCNN processing engine
         * @param W         Width of the output activations
         * @param H         Height of the output activations
         * @param stride    Stride for the layer
         * @param act       1D activations queue with linearized activations indexes to be processed
         * @param wgt       1D weights queue with linearized activations indexes to be processed
         * @return          Return stats for the given PE
         */
        PE_stats computeSCNNPE(uint64_t W, uint64_t H, int stride, const act_idxMap &act, const wgt_idxMap &wgt);

        /** Compute SCNN tile
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
         * @return          Return stats for the current tile
         */
        Tile_stats computeSCNNTile(int n, int ct, int ck, int kc, int tw, int th, uint64_t X, uint64_t Y, int Kc,
                uint64_t K, uint64_t W, uint64_t H, uint64_t R, uint64_t S, int stride, int padding,
                const base::Array<T> &act, const base::Array<T> &wgt);

    public:

        /** Constructor
         * @param _Wt           Number of PE columns
         * @param _Ht           Number of PE rows
         * @param _I            Column multipliers per PE
         * @param _F            Row multipliers per PE
         * @param _out_acc_size Output accumulator size
         * @param _BANKS        Number of banks
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         * @param _QUIET        Avoid std::out messages
         * @param _CHECK        Check the correctness of the simulations
         */
        SCNN(uint32_t _Wt, uint32_t _Ht, uint32_t _I, uint32_t _F, uint32_t _out_acc_size, uint32_t _BANKS,
             uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) : Simulator<T>(_N_THREADS,_FAST_MODE,_QUIET,
              _CHECK), Wt(_Wt), Ht(_Ht), I(_I), F(_F), out_acc_size(_out_acc_size), BANKS(_BANKS), BASELINE(false) {}

        /** Constructor
         * @param _Wt           Number of PE columns
         * @param _Ht           Number of PE rows
         * @param _I            Column multipliers per PE
         * @param _F            Row multipliers per PE
         * @param _out_acc_size Output accumulator size
         * @param _BANKS        Number of banks
         * @param _BASELINE     Simulate only baseline
         * @param _MEMORY       Memory model
         * @param _N_THREADS    Number of parallel threads for multi-threading execution
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         * @param _QUIET        Avoid std::out messages
         * @param _CHECK        Check the correctness of the simulations
         */
        SCNN(uint32_t _Wt, uint32_t _Ht, uint32_t _I, uint32_t _F, uint32_t _out_acc_size, uint32_t _BANKS,
                bool _BASELINE, const Memory &_MEMORY, uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) :
                Simulator<T>(_MEMORY,_N_THREADS,_FAST_MODE,_QUIET,_CHECK), Wt(_Wt), Ht(_Ht), I(_I), F(_F),
                out_acc_size(_out_acc_size), BANKS(_BANKS), BASELINE(_BASELINE) {}

        /** Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        virtual void run(const base::Network<T> &network);

        /** Calculate potentials for the given network
         * @param network   Network we want to calculate work reduction
         */
        virtual void potentials(const base::Network<T> &network);

        /** Simulate scnn memory cycles for on-chip memory dynamic width storage
         * @param network   Network we want to check
         */
        void on_chip_cycles(const base::Network<T> &network);

    };

}

#endif //DNNSIM_SCNN_H
