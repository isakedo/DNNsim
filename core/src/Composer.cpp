
#include <core/Composer.h>

namespace core {

    template <typename T>
    std::string Composer<T>::header() {
        std::string header = "Number of inputs in parallel per tile: " + std::to_string(INPUTS) + "\n";
        header += "Delay per tile: " + std::to_string(DELAY) + "\n";
        return header;
    }

    template <typename T>
    void Composer<T>::calculate_delay(const std::shared_ptr<TilesData<T>> &tiles_data) {
        uint64_t max_delay = 0;
        for (const auto &tile_data : tiles_data->data) {

            if (!tile_data.write)
                continue;

            auto input_step = ceil(tile_data.out_banks.size() / (double)INPUTS);
            auto delay = input_step * DELAY;
            if (delay > max_delay)
                max_delay = delay;

        }
        *global_cycle += max_delay;
    }

    INITIALISE_DATA_TYPES(Composer);

}
