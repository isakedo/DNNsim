#ifndef DNNSIM_SCNN_H
#define DNNSIM_SCNN_H

#include "Architecture.h"

typedef std::vector<std::tuple<int,int>> act_idxMap;
typedef std::vector<std::tuple<int,int,int>> wgt_idxMap;

namespace core {

    /**
     * SCNN simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class SCNN : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** Number of PE columns */
        const uint32_t Wt;

        /** Number of PE rows */
        const uint32_t Ht;

        /** Column multipliers per PE */
        const uint32_t I;

        /** Row multipliers per PE */
        const uint32_t F;

        /** Output accumulator size */
        const int OUT_ACC_SIZE;

        /** Number of banks */
        const int BANKS;

        /** Enable fast mode: only one image */
        const bool FAST_MODE = false;

        /** Avoid std::out messages */
        const bool QUIET = false;

        /* CYCLES */

        /** Calculate in which bank the output activation is mapped
         * @param k         Filter
         * @param x         X position
         * @param y         W position
         * @return          Accumulator bank index
         */
        int map_accumulator(uint32_t k, uint32_t x, uint32_t y);

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

        /* AUXILIARY FUNCTIONS */

        /**
         * Return name of the class
         * @return Name
         */
        std::string name() override;

        /**
         * Convert the data representation to the one need it.
         * @param data          Array of values
         * @param data_prec     Activation layer precision
         */
        void dataConversion(base::Array<T> &data, uint8_t data_prec) override {}

        /* CYCLES (NOT USED)*/

        /**
         * Return stats filename for the architecture in the cycles function
         * @return Filename
         */
        std::string filename() override;

        /**
         * Return stats header for the architecture in the cycles function
         * @return Header
         */
        std::string header() override;

        /**
         * Return if calculate deltas for the window buffer
         * @return True if diffy, False if not
         */
        bool diffy() override;

        /**
         * Return if schedule the weight buffer
         * @return True if weight buffer to schedule, False if not
         */
        bool schedule() override;

        /**
         * Calculate cycles for all the tiles
         * @param tiles_data Processing information for all the tiles
         */
        void process_tiles(const std::vector<TileData<T>> &tiles_data) override;

        /* POTENTIALS */

        /**
         * Return stats filename for the architecture in the potentials function
         * @return Filename
         */
        std::string filename_pot() override;

        /**
         * Return stats header for the architecture in the potentials function
         * @return Header
         */
        std::string header_pot() override;

        /** Compute number of one bit multiplications given a weights and an activation
         * @param act           Activation
         * @param wgt           Weight
         * @return              Number of one bit multiplications
         */
        uint16_t computeBits(T act, T wgt) override;

    public:

        /** Constructor
         * @param _Wt           Number of PE columns
         * @param _Ht           Number of PE rows
         * @param _I            Column multipliers per PE
         * @param _F            Row multipliers per PE
         * @param _OUT_ACC_SIZE Output accumulator size
         * @param _BANKS        Number of banks
         * @param _FAST_MODE    Enable fast mode to simulate only one image
         * @param _QUIET        Avoid std::out messages
         */
        SCNN(uint32_t _Wt, uint32_t _Ht, uint32_t _I, uint32_t _F, uint32_t _OUT_ACC_SIZE, uint32_t _BANKS,
             bool _FAST_MODE, bool _QUIET) : Wt(_Wt), Ht(_Ht), I(_I), F(_F), OUT_ACC_SIZE(_OUT_ACC_SIZE), BANKS(_BANKS),
             QUIET(_QUIET) {}

        /** Run the timing simulator of the architecture
         * @param network   Network we want to simulate
         */
        void run(const base::Network<T> &network);

    };

}

#endif //DNNSIM_SCNN_H
