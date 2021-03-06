#ifndef DNNSIM_LACONIC_H
#define DNNSIM_LACONIC_H

#include "Architecture.h"

namespace core {

    /**
     * Laconic simulator
     * @tparam T 16 bits fixed point
     */
    template <typename T>
    class Laconic : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** Activate booth-like encoding */
        const bool BOOTH_ENCODING;

        /* AUXILIARY FUNCTIONS */

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
         */
        void dataConversion(base::Array<T> &data) override;

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
         * Calculate cycles for linear layers
         * @param tiles_data Processing information for all the tiles
         */
        void process_linear(const std::shared_ptr<TilesData<T>> &tiles_data);

        /**
         * Calculate cycles for matrix multiply
         * @param tile_data Processing information for all the tiles
         */
        void process_mmul(const std::shared_ptr<TilesData<T>> &tiles_data);

        /**
         * Calculate cycles for all the tiles
         * @param tiles_data Processing information for all the tiles
         */
        void process_tiles(const std::shared_ptr<TilesData<T>> &tiles_data) override;

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
         * @param _LANES          Number of concurrent multiplications per PE
         * @param _COLUMNS        Number of columns
         * @param _ROWS           Number of rows
         * @param _TILES          Number of tiles
         * @param _PE_WIDTH          Bits per PE
         * @param _BOOTH_ENCODING   Activate booth-like encoding
         */
        Laconic(uint32_t _LANES, uint32_t _COLUMNS, uint32_t _ROWS, uint32_t _TILES, uint32_t _PE_WIDTH,
                bool _BOOTH_ENCODING) : Architecture<T>(_LANES, _COLUMNS, _ROWS, _TILES, _PE_WIDTH),
                BOOTH_ENCODING(_BOOTH_ENCODING) {}

    };

}

#endif //DNNSIM_LACONIC_H
