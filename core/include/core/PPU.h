#ifndef DNNSIM_PPU_H
#define DNNSIM_PPU_H

#include <core/Utils.h>

namespace core {

    /**
     *
     */
    template <typename T>
    class PPU {

    private:

        const uint32_t INPUTS;

        const uint32_t DELAY;

    public:

        PPU(uint32_t _INPUTS, uint32_t _DELAY) : INPUTS(_INPUTS), DELAY(_DELAY) {}

        uint32_t calculate_delay(const std::vector<TileData<T>> &tiles_data);

    };

}

#endif //DNNSIM_PPU_H
