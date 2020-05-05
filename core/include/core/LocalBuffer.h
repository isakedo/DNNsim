#ifndef DNNSIM_LOCALBUFFER_H
#define DNNSIM_LOCALBUFFER_H

#include "Memory.h"

namespace core {

    /**
     *
     */
    template <typename T>
    class LocalBuffer : public Memory<T> {

    private:

        const uint32_t ROWS = 0;

        const uint32_t READ_DELAY = 0;

        const uint32_t WRITE_DELAY = 0;

        uint32_t idx = 0;

        std::vector<uint64_t> ready_cycle;

        std::vector<uint64_t> done_cycle;

        /* STATISTICS */

        uint64_t stall_cycles = 0;

    public:

        LocalBuffer(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_wgt_addresses,
                uint32_t _ROWS, uint32_t _READ_DELAY, uint32_t _WRITE_DELAY) : Memory<T>(_tracked_data, _act_addresses,
                _wgt_addresses), ROWS(_ROWS), READ_DELAY(_READ_DELAY), WRITE_DELAY(_WRITE_DELAY) {}

        uint64_t getStallCycles() const;

        // GENERAL

        void configure_layer() override;

        uint64_t getFifoReadyCycle() const;

        uint64_t getFifoDoneCycle() const;

        void update_fifo();

        // READ

        bool data_ready();

        void read_request(uint64_t global_buffer_ready_cycle);

        void evict_data();

        // WRITE

        bool write_ready();

        void write_request(uint64_t extra_delay);

        void update_done_cycle(uint64_t global_buffer_ready_cycle);

    };

}

#endif //DNNSIM_LOCALBUFFER_H
