#ifndef DNNSIM_BITPRAGMATIC_H
#define DNNSIM_BITPRAGMATIC_H

#include "Architecture.h"

namespace core {

    /**
     * BitPragmatic simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class BitPragmatic : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** Bits of the first stage in the two stages shifting */
        const uint32_t BITS_FIRST_STAGE;

        /** Number of registers per SIP */
        const uint32_t COLUMN_REGISTERS;

        /** Activate booth-like encoding */
        const bool BOOTH_ENCODING;

        /** Diffy simulation */
        const bool DIFFY;

        /** BitTactical simulation */
        const bool TCT;

        /** Previous ending cycles */
        std::vector<std::vector<uint64_t>> previous_cycles;

        /** Activations mask to remove negative numbers */
        uint16_t act_mask = 0;

        /* AUXILIARY FUNCTIONS */

        /**
         * Initialise layer
         * @param _act_prec     Activations precision
         * @param _wgt_prec     Weights precision
         * @param _network_bits Network bits
         * @param _linear       Linear layer
         */
        void initialise_layer(int _act_prec, int _wgt_prec, int _network_bits, bool _linear);

        /**
         * Initialise stats to zero
         * @param COLUMNS   Number of columns
         * @param TILES     Number of tiles
         */
        void initialise_batch(uint64_t COLUMNS, uint64_t TILES);

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
         * @return              Cycles for the PE
         */
        uint16_t process_pe(const BufferSet<T> &act_row, const BufferRow<T> &wgt_row, int window_idx, int filter_idx,
                int lanes, int time);

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
         * @param _BITS_FIRST_STAGE     Bits of the first stage in the two stages shifting
         * @param _COLUMN_REGISTERS     Number of registers per SIP
         * @param _BOOTH_ENCODING       Activate booth-like encoding
         * @param _DIFFY                Enable Diffy
         * @param _TCT                  Enable BitTactical simulation
         */
        BitPragmatic(uint32_t _BITS_FIRST_STAGE, uint32_t _COLUMN_REGISTERS, bool _BOOTH_ENCODING, bool _DIFFY,
                bool _TCT) : BITS_FIRST_STAGE(_BITS_FIRST_STAGE), COLUMN_REGISTERS(_COLUMN_REGISTERS),
                BOOTH_ENCODING(_BOOTH_ENCODING), DIFFY(_DIFFY), TCT(_TCT) {}

    };

}

#endif //DNNSIM_BITPRAGMATIC_H
