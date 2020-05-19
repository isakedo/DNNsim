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

        /** Global cycle */
        std::shared_ptr<uint64_t> global_cycle;

    public:

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

        void calculate_delay(uint64_t outputs);

    };

}

#endif //DNNSIM_PPU_H
