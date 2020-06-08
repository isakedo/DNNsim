#ifndef DNNSIM_SHAPESHIFTER_H
#define DNNSIM_SHAPESHIFTER_H

#include "Architecture.h"

namespace core {

    /**
     * ShapeShifter simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class ShapeShifter : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** Number of columns per group */
        const uint32_t GROUP_SIZE;

        /** Number of registers per SIP */
        const uint32_t COLUMN_REGISTERS;

        /** Calculate also the minor bit for dynamic precisions */
        const bool MINOR_BIT;

        /** Diffy simulation */
        const bool DIFFY;

        /** BitTactical simulation */
        const bool TCL;

        /** Ready compute cycle */
        uint64_t ready_compute_cycle = 0;

        /** Previous index */
        uint64_t previous_index = 0;

        /** Previous ending cycles */
        std::vector<uint64_t> previous_cycles;

        /** Previous compute cycles */
        std::vector<uint64_t> previous_compute_cycles;

        /** Activations mask to remove negative numbers */
        uint16_t act_mask = 0;

        /* AUXILIARY FUNCTIONS */

        /**
         * Initialise layer
         * @param _act_prec      Activations precision
         * @param _wgt_prec      Weights precision
         * @param _act_blks      Activation steps
         * @param _wgt_blks      Weight steps
         * @param _network_width Network width
         * @param _signed_act    Signed activations
         * @param _signed_wgt    Signed weights
         * @param _linear        Linear layer
         * @param EF_COLUMNS     Number of effective columns
         */
        void configure_layer(int _act_prec, int _wgt_prec, int _act_blks, int _wgt_blks, int _network_width,
                bool _signed_act, bool _signed_wgt, bool _linear, uint64_t EF_COLUMNS) override;

        /**
         * Get number of cycles
         * @return Cycles
         */
        uint64_t getCycles() const override;

        /**
         * Return name of the class
         * @return Name
         */
        std::string name() override;

        /* CYCLES */

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
         * Calculate cycles for the current pe
         * @param act_row       Act rows
         * @param wgt_row       Wgt row
         * @param window_idx    Window index
         * @param filter_idx    Filter index
         * @param lanes         Number of lanes
         * @param time          Current time
         * @param min_group_bit Minor bit for the group (Overwritten)
         * @param max_group_bit Leading bit for the group (Overwritten)
         */
        void process_pe(const BufferSet<T> &act_row, const BufferRow<T> &wgt_row, int window_idx, int filter_idx,
                int lanes, int time, int &min_group_bit, int &max_group_bit, int act_blk);

        /**
         * Calculate cycles for linear layers
         * @param tile_data Processing information for all the tiles
         */
        void process_linear(const std::vector<TileData<T>> &tiles_data);

        /**
         * Calculate cycles for matrix multiply
         * @param tile_data Processing information for all the tiles
         */
        void process_mmul(const std::vector<TileData<T>> &tiles_data);

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
         * @param _LANES                  Number of concurrent multiplications per PE
         * @param _COLUMNS                Number of columns
         * @param _ROWS                   Number of rows
         * @param _TILES                  Number of tiles
         * @param _PE_WIDTH                  Bits per PE
         * @param _GROUP_SIZE               Granularity for dynamic precisions
         * @param _COLUMN_REGISTERS         Number of registers per SIP
         * @param _MINOR_BIT                Calculate also the minor bit for dynamic precisions
         * @param _DIFFY                    Enable Diffy
         * @param _TCL                      Enable BitTactical simulation
         */
        ShapeShifter(uint32_t _LANES, uint32_t _COLUMNS, uint32_t _ROWS, uint32_t _TILES, uint32_t _PE_WIDTH,
                uint32_t _GROUP_SIZE, uint32_t _COLUMN_REGISTERS, bool _MINOR_BIT, bool _DIFFY, bool _TCL) :
                Architecture<T>(_LANES, _COLUMNS, _ROWS, _TILES, _PE_WIDTH), GROUP_SIZE(_GROUP_SIZE),
                COLUMN_REGISTERS(_COLUMN_REGISTERS), MINOR_BIT(_MINOR_BIT), DIFFY(_DIFFY), TCL(_TCL) {}

    };

}

#endif //DNNSIM_SHAPESHIFTER_H
