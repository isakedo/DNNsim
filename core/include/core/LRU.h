#ifndef DNNSIM_LRU_H
#define DNNSIM_LRU_H

#include "EvictionPolicy.h"

namespace core {

    class LRU : public EvictionPolicy {

    private:

        /** Addresses eviction list */
        std::list<uint64_t> data;

        /** Least Recently Used tracking tree */
        std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru;

        /**
         * Flush all the addresses
         */
        void flush() override;

        /**
         * Check if the bank is full
         * @return False if full, True otherwise
         */
        bool free_entry() override;

        /**
         * Insert a new address in the eviction policy tracking
         * @param addr Memory address
         */
        void insert_addr(uint64_t addr) override;

        /**
         * Remove an address from the eviction policy tracking
         * @return Memory address evicted
         */
        uint64_t evict_addr() override;

        /**
         * Update the eviction policy tracking for the accessed address
         * @param addr Memory address accessed
         */
        void update_status(uint64_t addr) override;

    public:

        /**
         * Constructor
         * @param _MAX_SIZE Maximum number of addresses
         */
        explicit LRU(uint64_t _MAX_SIZE) : EvictionPolicy(_MAX_SIZE) {}

    };

}

#endif //DNNSIM_LRU_H
