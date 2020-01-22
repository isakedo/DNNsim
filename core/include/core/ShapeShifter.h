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
        const bool TCT;

        /** Ready compute cycle */
        uint64_t ready_compute_cycle;

        /** Previous index */
        uint64_t previous_index;

        /** Previous ending cycles */
        std::vector<uint64_t> previous_cycles;

        /** Previous compute cycles */
        std::vector<uint64_t> previous_compute_cycles;

        /** Activations mask to remove negative numbers */
        uint16_t act_mask = 0;

        /* AUXILIARY FUNCTIONS */

        /**
         * Initialise layer
         * @param _act_prec     Activations precision
         * @param _wgt_prec     Weights precision
         * @param _network_bits Network bits
         * @param _linear       Linear layer
         * @param COLUMNS       Number of columns
         */
        void initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear, uint64_t COLUMNS) override;

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

        /**
         * Convert the data representation to the one need it.
         * @param data          Array of values
         * @param data_prec     Activation layer precision
         */
        void dataConversion(base::Array<T> &data, uint8_t data_prec) override;

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
                int lanes, int time, int &min_group_bit, int &max_group_bit);

        /**
         * Calculate cycles for linear layers
         * @param tile_data Processing information for all the tiles
         */
        void process_linear(const std::vector<TileData<T>> &tiles_data);

        /**
         * Calculate cycles for convolutional layers
         * @param tile_data Processing information for all the tiles
         */
        void process_convolution(const std::vector<TileData<T>> &tiles_data);

        /**
         * Calculate cycles for all the tiles
         * @param tiles_data Processing information for all the tiles
         */
        void process_tiles(const std::vector<TileData<T>> &tiles_data) override;

        /**
         * Return true if ready to feed need data
         * @return True if ready to process data
         */
        bool ready() override;

        /**
         * Return true if processing has finished
         * @return True if done
         */
        bool flush() override;

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
         * @param _GROUP_SIZE               Granularity for dynamic precisions
         * @param _COLUMN_REGISTERS         Number of registers per SIP
         * @param _MINOR_BIT                Calculate also the minor bit for dynamic precisions
         * @param _DIFFY                    Enable Diffy
         * @param _TCT                      Enable BitTactical simulation
         */
        ShapeShifter(uint32_t _GROUP_SIZE, uint32_t _COLUMN_REGISTERS, bool _MINOR_BIT, bool _DIFFY, bool _TCT) :
                GROUP_SIZE(_GROUP_SIZE), COLUMN_REGISTERS(_COLUMN_REGISTERS), MINOR_BIT(_MINOR_BIT), DIFFY(_DIFFY),
                TCT(_TCT) {}

    };

}

#endif //DNNSIM_SHAPESHIFTER_H
