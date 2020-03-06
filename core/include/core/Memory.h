#ifndef DNNSIM_MEMORY_H
#define DNNSIM_MEMORY_H

#include "Utils.h"

namespace core {

    const uint64_t NULL_ADDR = UINT64_MAX;

    const uint64_t NULL_TIME = UINT64_MAX;

    const uint64_t BLOCK_SIZE = 0x40; // Align to 64 bits

    template <typename T>
    class Memory {

    protected:

        std::shared_ptr<std::map<uint64_t, uint64_t>> tracked_data;

        /** Global cycle */
        std::shared_ptr<uint64_t> global_cycle;

    public:

        explicit Memory(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data) :
                tracked_data(_tracked_data) {}

        /**
         * Set shared global cycle
         * @param globalCycle
         */
        void setGlobalCycle(const std::shared_ptr<uint64_t> &globalCycle) {
            global_cycle = globalCycle;
        }

        virtual void configure_layer() = 0;

    };

}

#endif //DNNSIM_MEMORY_H
