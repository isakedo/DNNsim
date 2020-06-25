#ifndef DNNSIM_DRAM_H
#define DNNSIM_DRAM_H

#include "Memory.h"
#include <DRAMSim.h>

namespace core {

    /**
     * DRAM model
     * @tparam T Data type values
     */
    template <typename T>
    class DRAM : public Memory<T> {

    private:

        /* SIMULATION PARAMETERS */

        /** Start activation address */
        const uint64_t START_ACT_ADDRESS;

        /** Start weight address */
        const uint64_t START_WGT_ADDRESS;

        /** Size in MiB */
        const uint32_t SIZE;

        /** Baseline values per DRAM block */
        uint32_t BASE_VALUES_PER_BLOCK;

        /** Baseline data size in bits */
        uint32_t BASE_DATA_SIZE;

        /** Activations values per DRAM block */
        uint32_t ACT_VALUES_PER_BLOCK;

        /** Activations data size in bits */
        uint32_t ACT_DATA_SIZE;

        /** Weights values per DRAM block */
        uint32_t WGT_VALUES_PER_BLOCK;

        /** Weights data size in bits */
        uint32_t WGT_DATA_SIZE;

        /** Memory system */
        DRAMSim::MultiChannelMemorySystem *dram_interface;

        /** Transactions queue */
        std::queue<std::tuple<uint64_t, bool>> request_queue;

        /** List of required addresses waiting to be transferred to on-chip */
        std::set<uint64_t> waiting_addresses;

        /* STATISTICS */

        /** Activation off-chip reads */
        uint64_t act_reads = 0;

        /** Weight off-chip reads */
        uint64_t wgt_reads = 0;

        /** Output Activation off-chip writes */
        uint64_t out_writes = 0;

        /**
         * Request an address to the memory system
         * @param address Address to request
         * @param isWrite Transaction type: True = Write, False = Read
         */
        void transaction_request(uint64_t address, bool isWrite);

    public:

        /**
         * Constructor
         * @param _tracked_data         Current tracked data on-chip
         * @param _act_addresses        Address range for activations
         * @param _wgt_addresses        Address range for weights
         * @param _SIZE                 Size in MiB
         * @param _BASE_DATA_SIZE       Baseline data size in bits
         * @param _clock_freq           Compute frequency
         * @param _START_ACT_ADDRESS    Start activation address
         * @param _START_WGT_ADDRESS    Start weight address
         * @param _dram_conf            DRAM configuration file
         * @param _system_conf          System configuration file
         * @param _network              Network name
         */
        DRAM(const std::shared_ptr<std::map<uint64_t, uint64_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_wgt_addresses,
                uint32_t _SIZE, uint32_t _BASE_DATA_SIZE, uint64_t _clock_freq, uint64_t _START_ACT_ADDRESS,
                uint64_t _START_WGT_ADDRESS, const std::string &_dram_conf, const std::string &_system_conf,
                const std::string &_network) : Memory<T>(_tracked_data, _act_addresses, _wgt_addresses),
                START_ACT_ADDRESS(_START_ACT_ADDRESS), START_WGT_ADDRESS(_START_WGT_ADDRESS), SIZE(_SIZE),
                BASE_VALUES_PER_BLOCK(64 / _BASE_DATA_SIZE), BASE_DATA_SIZE(_BASE_DATA_SIZE), ACT_VALUES_PER_BLOCK(0),
                ACT_DATA_SIZE(0), WGT_VALUES_PER_BLOCK(0), WGT_DATA_SIZE(0) {

            dram_interface = DRAMSim::getMemorySystemInstance(_dram_conf, _system_conf, "./DRAMSim2/",
                    "DNNsim_" + _network, _SIZE);

            DRAMSim::TransactionCompleteCB *read_cb =
                    new DRAMSim::Callback<DRAM, void, unsigned, uint64_t, uint64_t>(this, &DRAM::read_transaction_done);

            DRAMSim::TransactionCompleteCB *write_cb =
                    new DRAMSim::Callback<DRAM, void, unsigned, uint64_t, uint64_t>(this, &DRAM::write_transaction_done);

            dram_interface->RegisterCallbacks(read_cb, write_cb, nullptr);

            dram_interface->setCPUClockSpeed(_clock_freq);
        }

        /**
         * Return the number of activation reads
         * @return Activation reads
         */
        uint64_t getActReads() const;

        /**
         * Return the number of weight reads
         * @return Weight reads
         */
        uint64_t getWgtReads() const;

        /**
         * Return the number of output activation writes
         * @return Output activation writes
         */
        uint64_t getOutWrites() const;

        /**
         * Return the start activation address
         * @return Start activation address
         */
        uint64_t getStartActAddress() const;

        /**
         * Return the start weight address
         * @return Start weight address
         */
        uint64_t getStartWgtAddress() const;

        /**
         * Return the baseline values per block
         * @return Baseline values per block
         */
        uint32_t getBaseValuesPerBlock() const;

        /**
         * Return the baseline data size in bits
         * @return Baseline data size in bits
         */
        uint32_t getBaseDataSize() const;

        /**
         * Return the activation values per block
         * @return Activation values per block
         */
        uint32_t getActValuesPerBlock() const;

        /**
         * Return the activation data size in bits
         * @return Activation data size in bits
         */
        uint32_t getActDataSize() const;

        /**
         * Return the weight values per block
         * @return Weight values per block
         */
        uint32_t getWgtValuesPerBlock() const;

        /**
         * Return the weight data size in bits
         * @return Weight data size in bits
         */
        uint32_t getWgtDataSize() const;

        /**
         * Return stats header for the DRAM
         * @return Header
         */
        std::string header() override;

        /** Update memory interface one cycle */
        void cycle();

        /** Configure memory for current layer parameters */
        void configure_layer() override {}; // Unused

        /**
         * Configure memory for current layer parameters
         * @param _ACT_DATA_SIZE Activations data size in bits
         * @param _WGT_DATA_SIZE Weight data size in bits
         */
        void configure_layer(uint32_t _ACT_DATA_SIZE, uint32_t _WGT_DATA_SIZE);

        /**
         * Take a list of addresses and compress them in ranges
         * @param addresses List of addresses
         * @return Compressed addresses
         */
        std::vector<AddressRange> compress_addresses(const std::vector<uint64_t> &addresses);

        /**
         * Check if all the requested data is on-chip
         * @return True if all values are on-chip
         */
        bool data_ready();

        /**
         * Add requested data on-chip to the waiting list if still off-chip
         * @param tiles_data Data to be read
         */
        void read_request(const TilesData<T> &tiles_data, bool layer_act_on_chip);

        /**
         * Callback function for read address from DRAM
         * @param id            Channel id
         * @param address       Address requested
         * @param _clock_cycle  Arrival clock cycle
         */
        void read_transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle);

        /**
         * Read memory addresses from off-chip alternating blocks of 16 addresses
         * @param act_addresses     Activation read addresses
         * @param wgt_addresses     Weight read addresses
         */
        void read_data(const std::vector<AddressRange> &act_addresses, const std::vector<AddressRange> &wgt_addresses);

        /**
         * Callback function for write address from DRAM
         * @param id            Channel id
         * @param address       Address requested
         * @param _clock_cycle  Arrival clock cycle
         */
        void write_transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle);

        /** Write memory addresses to off-chip
         * @param write_addresses   Output activation addresses
         */
        void write_data(const std::vector<AddressRange> &write_addresses);

    };

}

#endif //DNNSIM_DRAM_H
