#ifndef DNNSIM_GLOBALBUFFER_H
#define DNNSIM_GLOBALBUFFER_H

#include "Memory.h"

namespace core {

    /**
     *
     */
    template <typename T>
    class GlobalBuffer : public Memory<T> {

    private:

        const uint64_t SIZE = 0;

        const uint32_t ACT_BANKS = 0;

        const uint32_t WGT_BANKS = 0;

        const uint32_t BANK_WIDTH = 0;

        const uint32_t READ_DELAY = 0;

        const uint32_t WRITE_DELAY = 0;

        uint64_t ready_cycle = 0; // Restart for new layers

    public:

        GlobalBuffer(const std::shared_ptr<std::map<uint64_t, bool>> &_tracked_data, uint64_t _SIZE,
                uint32_t _ACT_BANKS, uint32_t _WGT_BANKS, uint32_t _BANK_WIDTH, uint32_t _READ_DELAY,
                uint32_t _WRITE_DELAY) : Memory<T>(_tracked_data), SIZE(_SIZE), ACT_BANKS(_ACT_BANKS),
                WGT_BANKS(_WGT_BANKS), BANK_WIDTH(_BANK_WIDTH), READ_DELAY(_READ_DELAY), WRITE_DELAY(_WRITE_DELAY) {}

        const uint64_t getSize() const;

        const uint32_t getActBanks() const;

        const uint32_t getWgtBanks() const;

        const uint32_t getBankWidth() const;

        const uint32_t getReadDelay() const;

        const uint32_t getWriteDelay() const;

        void configure_layer();

        /**
         * Return true if ready to feed need data
         * @return True if ready to process data
         */
        bool data_ready();

        void read_request(const std::vector<TileData<T>> &tiles_data);

        void evict_data(const std::vector<AddressRange> &addresses);

    };

}

#endif //DNNSIM_GLOBALBUFFER_H
