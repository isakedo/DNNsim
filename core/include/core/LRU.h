#ifndef DNNSIM_LRU_H
#define DNNSIM_LRU_H

#include "EvictionPolicy.h"

namespace core {

    class LRU : public EvictionPolicy {

    private:

        std::list<uint64_t> data;

        std::unordered_map<uint64_t, std::list<uint64_t>::iterator> lru;

        void flush() override;

        bool free_entry() override;

        void insert_addr(uint64_t addr) override;

        uint64_t evict_addr() override;

        void update_status(uint64_t addr) override;

    public:

        explicit LRU(uint64_t _MAX_SIZE) : EvictionPolicy(_MAX_SIZE) {}

    };

}

#endif //DNNSIM_MRU_H
