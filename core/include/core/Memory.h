#ifndef DNNSIM_MEMORY_H
#define DNNSIM_MEMORY_H

#include "Utils.h"

namespace core {

    /** Not valid address */
    const uint64_t NULL_ADDR = UINT64_MAX;

    /** Not valid delay */
    const uint32_t NULL_DELAY = UINT32_MAX;

    template <typename T>
    class Memory {

    protected:

        /** Current tracked data on-chip: Tuple <Address, on-chip,Hierarchy level> */
        std::shared_ptr<std::map<uint64_t, uint32_t>> tracked_data;

        /** Address range for activations */
        std::shared_ptr<AddressRange> act_addresses;

        /** Address range for output activations */
        std::shared_ptr<AddressRange> out_addresses;

        /** Address range for weights */
        std::shared_ptr<AddressRange> wgt_addresses;

        /** Global cycle */
        std::shared_ptr<uint64_t> global_cycle;

    public:

        /**
         * Constructor
         * @param _tracked_data     Current tracked data on-chip
         * @param _act_addresses    Activations addresses range
         * @param _out_addresses    Output activation addresses range
         * @param _wgt_addresses    Weight addresses range
         */
        Memory(const std::shared_ptr<std::map<uint64_t, uint32_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_out_addresses,
                const std::shared_ptr<AddressRange> &_wgt_addresses) {
            tracked_data = _tracked_data;
            act_addresses = _act_addresses;
            out_addresses = _out_addresses;
            wgt_addresses = _wgt_addresses;
        }

        /**
         * Return stats header for the memory modules
         * @return Header
         */
        virtual std::string header() = 0;

        /**
         * Set shared global cycle
         * @param globalCycle
         */
        void setGlobalCycle(const std::shared_ptr<uint64_t> &globalCycle) {
            global_cycle = globalCycle;
        }

        /** Configure memory for current layer parameters */
        virtual void configure_layer() = 0;

    };

}

#endif //DNNSIM_MEMORY_H
