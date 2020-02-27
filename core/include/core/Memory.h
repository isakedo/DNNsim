#ifndef DNNSIM_MEMORY_H
#define DNNSIM_MEMORY_H

#include "Utils.h"

namespace core {

    template <typename T>
    class Memory {

    protected:

        std::shared_ptr<std::map<uint64_t, bool>> tracked_data;

        /** Global cycle */
        std::shared_ptr<uint64_t> global_cycle;

    public:

        Memory(const std::shared_ptr<std::map<uint64_t, bool>> &_tracked_data) : tracked_data(_tracked_data) {}

        /**
         * Set shared global cycle
         * @param globalCycle
         */
        void setGlobalCycle(const std::shared_ptr<uint64_t> &globalCycle) {
            global_cycle = globalCycle;
        }

    };

}

#endif //DNNSIM_MEMORY_H
