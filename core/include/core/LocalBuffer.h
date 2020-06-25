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

        // GENERAL

        /** Configure memory for current layer parameters */
        void configure_layer() override;

        /**
         * Return if the data is ready to be read
         * @return True if data is ready
         */
        bool data_ready();

        void read_request(bool read = true);;

        void write_request();

    };

}

#endif //DNNSIM_LOCALBUFFER_H
