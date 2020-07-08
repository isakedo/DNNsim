#ifndef DNNSIM_FIFO_H
#define DNNSIM_FIFO_H

#include "EvictionPolicy.h"

namespace core {

    class FIFO : public EvictionPolicy {

    private:

        /** FIFO eviction queue */
        std::queue<uint64_t> fifo;

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
        explicit FIFO(uint64_t _MAX_SIZE) : EvictionPolicy(_MAX_SIZE) {}

    };

}

#endif //DNNSIM_FIFO_H
