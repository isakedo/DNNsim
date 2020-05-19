
#include <core/PPU.h>

namespace core {

    template <typename T>
    std::string PPU<T>::header() {
        std::string header = "Number of inputs in parallel: " + std::to_string(INPUTS) + "\n";
        header += "Delay: " + std::to_string(DELAY) + "\n";
        return header;
    }

    template <typename T>
    void PPU<T>::calculate_delay(uint64_t outputs) {
        auto input_steps = ceil(outputs / (double)INPUTS);
        auto delay = input_steps * DELAY;
        *global_cycle += delay;
    }

    INITIALISE_DATA_TYPES(PPU);

}
