#ifndef DNNSIM_PARALLEL_H
#define DNNSIM_PARALLEL_H

#include "Architecture.h"

namespace core {

    /**
     * DaDianNao simulator
     * @tparam T 16 bits fixed point or 32 bits floating-point
     */
    template <typename T>
    class DaDianNao : public Architecture<T> {

    private:

        /* PARAMETERS */

        /** BitTactical simulation */
        const bool TCL;

        /* AUXILIARY FUNCTIONS */

        /**
         * Return name of the class
         * @return Name
         */
        std::string name() override;

        /* CYCLES */

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
         * @param _LANES    Number of concurrent multiplications per PE
         * @param _COLUMNS  Number of columns
         * @param _ROWS     Number of rows
         * @param _TILES    Number of tiles
         * @param _PE_WIDTH    Bits per PE
         * @param _TCL        Enable BitTactical simulation
         */
        DaDianNao(uint32_t _LANES, uint32_t _COLUMNS, uint32_t _ROWS, uint32_t _TILES, uint32_t _PE_WIDTH,
                bool _TCL) : Architecture<T>(_LANES, _COLUMNS, _ROWS, _TILES, _PE_WIDTH), TCL(_TCL) {}

    };

}

#endif //DNNSIM_PARALLEL_H
