#ifndef DNNSIM_COMPOSER_H
#define DNNSIM_COMPOSER_H

#include <core/Utils.h>

namespace core {

    /**
     * Composer column
     * @tparam T Data type values
     */
    template <typename T>
    class Composer {

    private:

        /** Parallel inputs per tile */
        const uint32_t INPUTS;

        /** Delay */
        const uint32_t DELAY;

    public:

        /**
         * Constructor
         * @param _INPUTS Parallel inputs per tile
         * @param _DELAY  Delay
         */
        Composer(uint32_t _INPUTS, uint32_t _DELAY) : INPUTS(_INPUTS), DELAY(_DELAY) {}

        /**
         * Return stats header for the Composer
         * @return Header
         */
        std::string header();

        /**
         * Calculate the delay of the composer column
         * @param tiles_data Outputs per tile
         * @return Delay in cycles
         */
        uint32_t calculate_delay(const std::vector<TileData<T>> &tiles_data);

    };

}

#endif //DNNSIM_COMPOSER_H
