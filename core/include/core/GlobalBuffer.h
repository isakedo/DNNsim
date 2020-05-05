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

        /* SIMULATION PARAMETERS */

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

        /* STATISTICS */

        uint64_t act_reads = 0;

        uint64_t wgt_reads = 0;

        uint64_t out_writes = 0;

        uint64_t act_bank_conflicts = 0;

        uint64_t wgt_bank_conflicts = 0;

        uint64_t out_bank_conflicts = 0;

        uint64_t stall_cycles = 0;

    public:

        GlobalBuffer(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_wgt_addresses,
                uint64_t _ACT_SIZE, uint64_t _WGT_SIZE, uint32_t _ACT_BANKS, uint32_t _WGT_BANKS, uint32_t _OUT_BANKS,
                uint32_t _BANK_WIDTH, uint32_t _READ_DELAY, uint32_t _WRITE_DELAY) : Memory<T>(_tracked_data,
                _act_addresses, _wgt_addresses), ACT_SIZE(_ACT_SIZE), WGT_SIZE(_WGT_SIZE), ACT_BANKS(_ACT_BANKS),
                WGT_BANKS(_WGT_BANKS), OUT_BANKS(_OUT_BANKS), BANK_WIDTH(_BANK_WIDTH), READ_DELAY(_READ_DELAY),
                WRITE_DELAY(_WRITE_DELAY) {}

        uint64_t getActReads() const;

        uint64_t getWgtReads() const;

        uint64_t getOutWrites() const;

        uint64_t getActBankConflicts() const;

        uint64_t getWgtBankConflicts() const;

        uint64_t getOutBankConflicts() const;

        uint64_t getActSize() const;

        uint64_t getWgtSize() const;

        uint32_t getActBanks() const;

        uint32_t getWgtBanks() const;

        uint32_t getBankWidth() const;

        uint32_t getOutBanks() const;

        uint64_t getStallCycles() const;

        uint64_t getActReadReadyCycle() const;

        uint64_t getWgtReadReadyCycle() const;

        uint64_t getWriteReadyCycle() const;

        void configure_layer() override;

        bool write_done();

        void act_read_request(const std::vector<TileData<T>> &tiles_data, uint64_t fifo_ready_cycle);

        void wgt_read_request(const std::vector<TileData<T>> &tiles_data, uint64_t fifo_ready_cycle);

        void write_request(const std::vector<TileData<T>> &tiles_data, uint64_t fifo_ready_cycle,
                uint64_t ppu_delay);

        void evict_data(bool evict_act, bool evict_wgt);

    };

}

#endif //DNNSIM_GLOBALBUFFER_H
