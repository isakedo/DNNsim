#ifndef DNNSIM_PPU_H
#define DNNSIM_PPU_H

#include <core/Utils.h>

namespace core {

    /**
     * Post-Processing Unit
     * @tparam T Data type values
     */
    template <typename T>
    class PPU {

    private:

        /** Concurrent inputs */
        const uint32_t INPUTS;

        /** Processing delay per step */
        const uint32_t DELAY;

        /** Global cycle */
        std::shared_ptr<uint64_t> global_cycle;

    public:

        /**
         * Constructor
         * @param _INPUTS Concurrent inputs
         * @param _DELAY Processing delay per step
         */
        PPU(uint32_t _INPUTS, uint32_t _DELAY) : INPUTS(_INPUTS), DELAY(_DELAY) {}

        /**
         * Return stats header for the Post-Processing Unit
         * @return Header
         */
        std::string header();

        /**
         * Set shared global cycle
         * @param globalCycle
         */
        void setGlobalCycle(const std::shared_ptr<uint64_t> &globalCycle) {
            global_cycle = globalCycle;
        }

        /**
         * Calculate the processing delay depending on the total outputs to compute
         * @param outputs Total outputs
         */
        void calculate_delay(uint64_t outputs);

    };

}

#endif //DNNSIM_PPU_H
