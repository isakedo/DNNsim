#ifndef DNNSIM_EVICTION_POLICY_H
#define DNNSIM_EVICTION_POLICY_H

#include "Utils.h"

namespace core {

    class EvictionPolicy {

    protected:

        const uint64_t MAX_SIZE = 0;

    public:

        explicit EvictionPolicy(uint64_t _MAX_SIZE) : MAX_SIZE(_MAX_SIZE) {}

        virtual void flush() = 0;

        virtual bool free_entry() = 0;

        virtual void insert_addr(uint64_t addr) = 0;

        virtual uint64_t evict_addr() = 0;

        virtual void update_policy(uint64_t addr) = 0;

    };

}

#endif //DNNSIM_EVICTION_POLICY_H
