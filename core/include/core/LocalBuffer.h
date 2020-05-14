#ifndef DNNSIM_LOCALBUFFER_H
#define DNNSIM_LOCALBUFFER_H

#include "Memory.h"

namespace core {

    /**
     * Local Buffer model
     * @tparam T Data type values
     */
    template <typename T>
    class LocalBuffer : public Memory<T> {

    private:

        /** Number of local buffer row slots */
        const uint32_t ROWS = 0;

        /** Read delay in cycles */
        const uint32_t READ_DELAY = 0;

        /** Write delay in cycles */
        const uint32_t WRITE_DELAY = 0;

        /** Row index */
        uint32_t idx = 0;

        /** Ready cycle per row */
        std::vector<uint64_t> ready_cycle;

        /** Ready cycle per row */
        std::vector<uint64_t> done_cycle;

        /* STATISTICS */

        uint64_t stall_cycles = 0;

    public:

        LocalBuffer(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_wgt_addresses,
                uint32_t _ROWS, uint32_t _READ_DELAY, uint32_t _WRITE_DELAY) : Memory<T>(_tracked_data, _act_addresses,
                _wgt_addresses), ROWS(_ROWS), READ_DELAY(_READ_DELAY), WRITE_DELAY(_WRITE_DELAY) {}

        /**
         * Return stats header for the Local Buffer
         * @return Header
         */
        std::string header() override;

        /**
         * Return stall cycles
         * @return Stall cycles
         */
        uint64_t getStallCycles() const;

        // GENERAL

        /** Configure memory for current layer parameters */
        void configure_layer() override;

        /**
         * Return ready cycle for current fifo position
         * @return Ready cycle for current fifo position
         */
        uint64_t getFifoReadyCycle() const;

        /**
         * Return done cycle for current fifo position
         * @return Done cycle for current fifo position
         */
        uint64_t getFifoDoneCycle() const;

        /** Update fifo position */
        void update_fifo();

        // READ

        /**
         * Return if the data is ready to be read
         * @return True if data is ready
         */
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
