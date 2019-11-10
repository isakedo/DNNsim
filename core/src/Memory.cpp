
#include <core/Memory.h>

namespace core {

    void Memory::transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle) {
        requests[address] = true;
        if (!queue.empty()) {
            auto tuple = queue.front();
            request_address(std::get<0>(tuple), std::get<1>(tuple));
            queue.pop();
        }
    }

    void Memory::request_address(uint64_t address, bool isWrite) {
        if (memory->willAcceptTransaction()) {
            memory->addTransaction(isWrite, address);
            requests[address] = false;
        } else {
            queue.push(std::make_tuple(address, isWrite));
        }
    }

    void Memory::wait_until(uint64_t _clock_cycle) {
        for (int c = _clock_cycle; c <= _clock_cycle; ++c) {
            memory->update();
            clock_cycle++;
        }
    }

    void Memory::wait_for(uint64_t address) {
        while(!requests[address]) {
            memory->update();
            clock_cycle++;
            mem_cycle++;
        }
    }

    void Memory::initialise() {
        memory = DRAMSim::getMemorySystemInstance(dram_conf, system_conf, "./DRAMSim2/", "DNNsim", size);

        DRAMSim::TransactionCompleteCB *read_cb =
                new DRAMSim::Callback<Memory, void, unsigned, uint64_t, uint64_t>(this, &Memory::transaction_done);
        DRAMSim::TransactionCompleteCB *write_cb =
                new DRAMSim::Callback<Memory, void, unsigned, uint64_t, uint64_t>(this, &Memory::transaction_done);

        memory->RegisterCallbacks(read_cb, write_cb, nullptr);
    }

    uint64_t Memory::getOnChipActSize() const {
        return on_chip_act_size;
    }

    uint64_t Memory::getOnChipWgtSize() const {
        return on_chip_wgt_size;
    }

    uint64_t Memory::getClockCycle() const {
        return clock_cycle;
    }

    void Memory::resetClockCycle() {
        clock_cycle = 0;
    }

    uint64_t Memory::getMemCycle() const {
        return mem_cycle;
    }

    void Memory::resetMemCycle() {
        mem_cycle = 0;
    }
}
