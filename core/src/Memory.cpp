
#include <core/Memory.h>

namespace core {

    void Memory::read_done(unsigned id, uint64_t address, uint64_t clock_cycle)
    {
        printf("[Callback] read complete: %d 0x%lx cycle=%lu\n", id, address, clock_cycle);
        read_requests[address] = true;

        if (!read_queue.empty()) {
            request_address(read_queue.front());
            read_queue.pop();
        }
    }

    void Memory::request_address(uint64_t address) {
        if (memory->willAcceptTransaction()) {
            memory->addTransaction(false, address);
        } else {
            read_queue.push(address);
        }
    }

    void Memory::wait_until(uint64_t clock_cycle) {
        for (int c = mem_cycle; c <= clock_cycle; ++c) {
            memory->update();
            mem_cycle++;
        }
    }

    void Memory::wait_for(uint64_t address) {
        while(!read_requests[address]) {
            memory->update();
            mem_cycle++;
        }
    }

}
