#ifndef DNNSIM_DRAM_H
#define DNNSIM_DRAM_H

#include "Memory.h"
#include <DRAMSim.h>

namespace core {

    /**
     *
     */
    template <typename T>
    class DRAM : public Memory<T> {

    private:

        /* SIMULATION PARAMETERS */

        const uint64_t START_ACT_ADDRESS;

        const uint64_t START_WGT_ADDRESS;

        uint32_t BASE_DATA_SIZE;

        uint32_t BASE_VALUES_PER_BLOCK;

        uint32_t ACT_VALUES_PER_BLOCK;

        uint32_t ACT_DATA_SIZE;

        uint32_t WGT_VALUES_PER_BLOCK;

        uint32_t WGT_DATA_SIZE;

        /** Memory system */
        DRAMSim::MultiChannelMemorySystem *dram_interface;

        /** Transactions queue */
        std::queue<std::tuple<uint64_t, bool>> request_queue;

        std::map<uint64_t, nullptr_t> waiting_addresses;

        /* STATISTICS */

        uint64_t act_reads = 0;

        uint64_t wgt_reads = 0;

        uint64_t out_writes = 0;

        uint64_t stall_cycles = 0;

        void transaction_request(uint64_t address, bool isWrite);

    public:

        DRAM(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_wgt_addresses,
                uint32_t _SIZE, uint32_t _BASE_DATA_SIZE, uint64_t clock_freq, uint64_t _START_ACT_ADDRESS,
                uint64_t _START_WGT_ADDRESS, const std::string &_dram_conf, const std::string &_system_conf,
                const std::string &_network) : Memory<T>(_tracked_data, _act_addresses, _wgt_addresses),
                START_ACT_ADDRESS(_START_ACT_ADDRESS), START_WGT_ADDRESS(_START_WGT_ADDRESS),
                BASE_VALUES_PER_BLOCK(64 / _BASE_DATA_SIZE), BASE_DATA_SIZE(_BASE_DATA_SIZE), ACT_VALUES_PER_BLOCK(0),
                ACT_DATA_SIZE(0), WGT_VALUES_PER_BLOCK(0), WGT_DATA_SIZE(0) {

            dram_interface = DRAMSim::getMemorySystemInstance(_dram_conf, _system_conf, "./DRAMSim2/",
                    "DNNsim_" + _network, _SIZE);

            DRAMSim::TransactionCompleteCB *read_cb =
                    new DRAMSim::Callback<DRAM, void, unsigned, uint64_t, uint64_t>(this, &DRAM::read_transaction_done);

            DRAMSim::TransactionCompleteCB *write_cb =
                    new DRAMSim::Callback<DRAM, void, unsigned, uint64_t, uint64_t>(this, &DRAM::write_transaction_done);

            dram_interface->RegisterCallbacks(read_cb, write_cb, nullptr);

            dram_interface->setCPUClockSpeed(clock_freq);
        }

        uint64_t getActReads() const;

        uint64_t getWgtReads() const;

        uint64_t getOutWrites() const;

        uint64_t getStallCycles() const;

        uint64_t getStartActAddress() const;

        uint64_t getStartWgtAddress() const;

        uint32_t getBaseValuesPerBlock() const;

        uint32_t getBaseDataSize() const;

        uint32_t getActValuesPerBlock() const;

        uint32_t getActDataSize() const;

        uint32_t getWgtValuesPerBlock() const;

        uint32_t getWgtDataSize() const;

        /**
         * Return stats header for the DRAM
         * @return Header
         */
        std::string header() override;

        void cycle();

        void configure_layer() override {}; // Unused

        void configure_layer(uint32_t _ACT_DATA_SIZE, uint32_t _WGT_DATA_SIZE);

        std::vector<AddressRange> compress_addresses(const std::vector<uint64_t> &addresses);

        bool data_ready(const std::vector<TileData<T>> &tiles_data);

        void read_request(const std::vector<TileData<T>> &tiles_data);

        void read_transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle);

        void read_data(const std::vector<AddressRange> &act_addresses, const std::vector<AddressRange> &wgt_addresses);

        void write_transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle);

        void write_data(const std::vector<AddressRange> &write_addresses);

    };

}

#endif //DNNSIM_DRAM_H
