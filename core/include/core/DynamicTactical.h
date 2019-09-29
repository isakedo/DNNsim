#ifndef DNNSIM_DYNAMICTACTICAL_H
#define DNNSIM_DYNAMICTACTICAL_H

#include "Simulator.h"

namespace core {

    template <typename T>
    class DynamicTactical : public Simulator<T> {

    private:

        /** Number of concurrent multiplications per PE */
        const uint32_t N_LANES;

        /** Number of columns */
        const uint32_t N_COLUMNS;

        /** Number of rows */
        const uint32_t N_ROWS;

        /** Number of rows */
        const uint32_t N_TILES;

    public:

        /** Constructor
         * @param _N_LANES          Number of concurrent multiplications per PE
         * @param _N_COLUMNS        Number of columns
         * @param _N_ROWS           Number of rows
         * @param _N_TILES          Number of tiles
         * @param _N_THREADS        Number of parallel threads for multi-threading execution
         * @param _FAST_MODE        Enable fast mode to simulate only one image
         * @param _QUIET            Avoid std::out messages
         * @param _CHECK            Check the correctness of the simulations
         */
        DynamicTactical(uint32_t _N_LANES, uint32_t _N_COLUMNS, uint32_t _N_ROWS, uint32_t _N_TILES,
                uint8_t _N_THREADS, bool _FAST_MODE, bool _QUIET, bool _CHECK) : Simulator<T>(_N_THREADS,_FAST_MODE,
                _QUIET,_CHECK), N_LANES(_N_LANES), N_COLUMNS(_N_COLUMNS), N_ROWS(_N_ROWS), N_TILES(_N_TILES) {}

        /** Run the timing simulator of the architecture
         * @param simulate  Simulate configuration
         * @param epochs    Number of epochs
         */
        void run(const sys::Batch::Simulate &simulate, int epochs);

        /** Calculate work reduction for the given network
         * @param simulate  Simulate configuration
         * @param epochs    Number of epochs
         */
        void potentials(const sys::Batch::Simulate &simulate, int epochs);

    };

}

#endif //DNNSIM_BITTACTICAL_H
