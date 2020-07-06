
#include <core/LRU.h>

namespace core {

    void LRU::flush() {
        data.clear();
        lru.clear();
    }

    bool LRU::free_entry() {
        return data.size() < this->MAX_SIZE;
    }

    void LRU::insert_addr(uint64_t addr) {
        data.push_front(addr);
        lru[addr] = data.begin();
    }

    uint64_t LRU::evict_addr() {
        assert(!data.empty());
        assert(!lru.empty());
        auto last = data.back();
        data.pop_back();
        lru.erase(last);
        return last;
    }

    void LRU::update_status(uint64_t addr) {
        data.erase(lru[addr]);
        insert_addr(addr);
    }

}
