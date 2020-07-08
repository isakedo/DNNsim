#ifndef DNNSIM_EVICTION_POLICY_H
#define DNNSIM_EVICTION_POLICY_H

#include "Utils.h"

namespace core {

    /**
     * Eviction Policy Interface
     */
    class EvictionPolicy {

    protected:

        /** Maximum number of addresses */
        const uint64_t MAX_SIZE = 0;

    public:

        /**
         * Constructor
         * @param _MAX_SIZE Maximum number of addresses
         */
        explicit EvictionPolicy(uint64_t _MAX_SIZE) : MAX_SIZE(_MAX_SIZE) {}

        /**
         * Flush all the addresses
         */
        virtual void flush() = 0;

        /**
         * Check if the bank is full
         * @return False if full, True otherwise
         */
        virtual bool free_entry() = 0;

        /**
         * Insert a new address in the eviction policy tracking
         * @param addr Memory address
         */
        virtual void insert_addr(uint64_t addr) = 0;

        /**
         * Remove an address from the eviction policy tracking
         * @return Memory address evicted
         */
        virtual uint64_t evict_addr() = 0;

        /**
         * Update the eviction policy tracking for the accessed address
         * @param addr Memory address accessed
         */
        virtual void update_status(uint64_t addr) = 0;

    };

}

#endif //DNNSIM_EVICTION_POLICY_H
