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

        std::shared_ptr<AddressRange> act_addresses;

        std::shared_ptr<AddressRange> wgt_addresses;

        /** Global cycle */
        std::shared_ptr<uint64_t> global_cycle;

    public:

        Memory(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_wgt_addresses)
                : tracked_data(_tracked_data), act_addresses(_act_addresses), wgt_addresses(_wgt_addresses) {}

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
