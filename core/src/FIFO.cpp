
#include <core/FIFO.h>

namespace core {

    void FIFO::flush() {
        fifo = std::queue<uint64_t>();
    }

    bool FIFO::free_entry() {
        return fifo.size() < this->MAX_SIZE;
    }

    void FIFO::insert_addr(uint64_t addr) {
        fifo.emplace(addr);
    }

    uint64_t FIFO::evict_addr() {
        assert(!fifo.empty());
        auto addr = fifo.front();
        fifo.pop();
        return addr;
    }

    void FIFO::update_policy(uint64_t addr) {
        // Nothing
    }

}
