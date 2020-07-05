#ifndef DNNSIM_FIFO_H
#define DNNSIM_FIFO_H

#include "EvictionPolicy.h"

namespace core {

    class FIFO : public EvictionPolicy {

    private:

        std::queue<uint64_t> fifo;

    public:

        explicit FIFO(uint64_t _MAX_SIZE) : EvictionPolicy(_MAX_SIZE) {}

        void flush() override;

        bool free_entry() override;

        void insert_addr(uint64_t addr) override;

        uint64_t evict_addr() override;

        void update_policy(uint64_t addr) override;

    };

}

#endif //DNNSIM_FIFO_H
