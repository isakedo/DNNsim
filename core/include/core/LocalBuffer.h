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

        /** Current occupied rows in the local buffer */
        uint32_t size = 0;

        /** Data Ready cycle */
        uint64_t ready_cycle = 0;

    public:

        /**
         * Constructor
         * @param _tracked_data         Current tracked data on-chip
         * @param _act_addresses        Address range for activations
         * @param _out_addresses        Output activation addresses range
         * @param _wgt_addresses        Address range for weights
         * @param _ROWS                 Number of local buffer row slots
         * @param _READ_DELAY           Read delay in cycles
         * @param _WRITE_DELAY          Write delay in cycles
         */
        LocalBuffer(const std::shared_ptr<std::map<uint64_t, uint32_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_out_addresses,
                const std::shared_ptr<AddressRange> &_wgt_addresses, uint32_t _ROWS, uint32_t _READ_DELAY,
                uint32_t _WRITE_DELAY) : Memory<T>(_tracked_data, _act_addresses, _out_addresses, _wgt_addresses),
                ROWS(_ROWS), READ_DELAY(_READ_DELAY), WRITE_DELAY(_WRITE_DELAY) {}

        /**
         * Return stats header for the Local Buffer
         * @return Header
         */
        std::string header() override;

        // GENERAL

        /** Configure memory for current layer parameters */
        void configure_layer() override;

        /**
         * Add new data to the local buffer
         * @param read True if add new data, false if not
         */
        void insert(bool read = true);

        /** Remove oldest value from the local buffer
         * @param read True if remove data, false if not
         */
        void erase(bool read = true);

        /**
         * Return if there is space in the local buffers
         * @return True if there is space
         */
        bool isFree();

        /**
         * Return if the data is ready to be read
         * @return True if data is ready
         */
        bool data_ready();

        /**
         * Return if the data is already written
         * @return True if data is written
         */
        bool write_done();

        /**
         * Calculate read ready cycle
         * @param read True if data is read
         */
        void read_request(bool read = true);

        /**
         * Calculate write ready cycle
         * @param delay Additional write delays
         */
        void write_request(uint64_t delay);

    };

}

#endif //DNNSIM_LOCALBUFFER_H
