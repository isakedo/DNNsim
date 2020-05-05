
#include <core/PPU.h>

namespace core {

    template <typename T>
    uint32_t PPU<T>::calculate_delay(const std::vector<core::TileData<T>> &tiles_data) {
        uint64_t inputs = 0;
        for (const auto &tile_data : tiles_data) {

            if (!tile_data.write)
                continue;

            inputs += tile_data.out_banks.size();

        }
        auto input_steps = ceil(inputs / (double)INPUTS);
        auto delay = input_steps * DELAY;
        return delay;
    }

    INITIALISE_DATA_TYPES(PPU);

}
