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

        const uint64_t ACT_SIZE = 0;

        const uint64_t WGT_SIZE = 0;

        const uint32_t ACT_BANKS = 0;

        const uint32_t WGT_BANKS = 0;

        const uint32_t OUT_BANKS = 0;

        const uint32_t BANK_WIDTH = 0;

        const uint32_t READ_DELAY = 0;

        const uint32_t WRITE_DELAY = 0;

        uint64_t act_read_ready_cycle = 0;

        uint64_t wgt_read_ready_cycle = 0;

        uint64_t write_ready_cycle = 0;

    public:

        GlobalBuffer(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data, uint64_t _ACT_SIZE,
                uint64_t _WGT_SIZE, uint32_t _ACT_BANKS, uint32_t _WGT_BANKS, uint32_t _OUT_BANKS, uint32_t _BANK_WIDTH,
                uint32_t _READ_DELAY, uint32_t _WRITE_DELAY) : Memory<T>(_tracked_data), ACT_SIZE(_ACT_SIZE),
                WGT_SIZE(_WGT_SIZE), ACT_BANKS(_ACT_BANKS), WGT_BANKS(_WGT_BANKS), OUT_BANKS(_OUT_BANKS),
                BANK_WIDTH(_BANK_WIDTH), READ_DELAY(_READ_DELAY), WRITE_DELAY(_WRITE_DELAY) {}

        const uint64_t getActSize() const;

        const uint64_t getWgtSize() const;

        const uint32_t getActBanks() const;

        const uint32_t getWgtBanks() const;

        const uint32_t getBankWidth() const;

        const uint32_t getReadDelay() const;

        const uint32_t getWriteDelay() const;

        const uint32_t getOutBanks() const;

        uint64_t getActReadReadyCycle() const;

        uint64_t getWgtReadReadyCycle() const;

        void configure_layer() override;

        bool write_done();

        void act_read_request(const std::vector<TileData<T>> &tiles_data);

        void wgt_read_request(const std::vector<TileData<T>> &tiles_data);

        void write_request(const std::vector<TileData<T>> &tiles_data);

        void evict_data(const std::vector<AddressRange> &addresses);

    };

}

#endif //DNNSIM_GLOBALBUFFER_H
