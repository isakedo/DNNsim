#ifndef DNNSIM_GLOBALBUFFER_H
#define DNNSIM_GLOBALBUFFER_H

#include <cstdint>
#include "Memory.h"

namespace core {

    /**
     * Global Buffer model
     * @tparam T Data type values
     */
    template <typename T>
    class GlobalBuffer : public Memory<T> {

    private:

        /* SIMULATION PARAMETERS */

        const uint32_t LEVELS = 0;

        /** Activation memory size */
        std::vector<uint64_t> ACT_SIZE;

        /** Weight memory size */
        std::vector<uint64_t> WGT_SIZE;

        /** Activation banks */
        const uint32_t ACT_BANKS = 0;

        /** Weight banks */
        const uint32_t WGT_BANKS = 0;

        /** Output Activation banks */
        const uint32_t OUT_BANKS = 0;

        /** Bank read delay */
        std::vector<uint32_t> READ_DELAY;

        /** Write read delay */
        std::vector<uint32_t> WRITE_DELAY;

        /** Bank interface datawidth */
        const uint32_t BANK_WIDTH = 0;

        /** Addresses per access */
        const uint32_t ADDRS_PER_ACCESS = 0;

        /** Partial sum banks ready cycle */
        uint64_t psum_read_ready_cycle = 0;

        /** Input banks ready cycle */
        uint64_t read_ready_cycle = 0;

        /** Output Activations banks ready cycle */
        uint64_t write_ready_cycle = 0;

        /* STATISTICS */

        /** Activation bank reads */
        uint64_t act_reads = 0;

        /** Partial sum bank reads */
        uint64_t psum_reads = 0;

        /** Weight bank reads */
        uint64_t wgt_reads = 0;

        /** Output bank writes */
        uint64_t out_writes = 0;

        /** Activation bank conflicts */
        uint64_t act_bank_conflicts = 0;

        /** Partial sum bank conflicts */
        uint64_t psum_bank_conflicts = 0;

        /** Weight bank conflicts */
        uint64_t wgt_bank_conflicts = 0;

        /** Output Activation bank conflicts */
        uint64_t out_bank_conflicts = 0;

    public:

        /**
         * Constructor
         * @param _tracked_data         Current tracked data on-chip
         * @param _act_addresses        Address range for activations
         * @param _wgt_addresses        Address range for weights
         * @param _LEVELS               Hierarchy levels
         * @param _ACT_SIZE             Activation size in Bytes
         * @param _WGT_SIZE             Weights size in Bytes
         * @param _ACT_OUT_BANKS        Activations banks
         * @param _WGT_BANKS            Weight banks
         * @param _BANK_WIDTH           On-chip bank width
         * @param _DRAM_WIDTH           Dram width
         * @param _READ_DELAY           Bank read delay
         * @param _WRITE_DELAY          Bank write delay
         */
        GlobalBuffer(const std::shared_ptr<std::map<uint64_t, uint32_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_wgt_addresses,
                uint32_t _LEVELS, const std::vector<uint64_t> &_ACT_SIZE, const std::vector<uint64_t> &_WGT_SIZE,
                uint32_t _ACT_OUT_BANKS, uint32_t _WGT_BANKS, uint32_t _BANK_WIDTH, uint32_t _DRAM_WIDTH,
                const std::vector<uint32_t> & _READ_DELAY, const std::vector<uint32_t> & _WRITE_DELAY) :
                Memory<T>(_tracked_data, _act_addresses, _wgt_addresses), LEVELS(_LEVELS), ACT_BANKS(_ACT_OUT_BANKS/2),
                WGT_BANKS(_WGT_BANKS), OUT_BANKS(_ACT_OUT_BANKS/2), BANK_WIDTH(_BANK_WIDTH),
                ADDRS_PER_ACCESS(ceil(BANK_WIDTH / (double)_DRAM_WIDTH)) {
            ACT_SIZE = _ACT_SIZE;
            WGT_SIZE = _WGT_SIZE;
            READ_DELAY = _READ_DELAY;
            WRITE_DELAY = _WRITE_DELAY;
        }

        /**
         * Return the number of activation bank reads
         * @return Activation bank reads
         */
        uint64_t getActReads() const;

        /**
         * Return the number of output bank reads
         * @return Output bank reads
         */
        uint64_t getPsumReads() const;

        /**
         * Return the number of weight bank reads
         * @return Weight bank reads
         */
        uint64_t getWgtReads() const;

        /**
         * Return the number of output bank writes
         * @return Output bank writes
         */
        uint64_t getOutWrites() const;

        /**
         * Return the number of activation bank conflicts
         * @return Activation bank conflicts
         */
        uint64_t getActBankConflicts() const;

        /**
         * Return the number of partial sum bank conflicts
         * @return Partial sum bank conflicts
         */
        uint64_t getPsumBankConflicts() const;

        /**
         * Return the number of weight bank conflicts
         * @return Weight bank conflicts
         */
        uint64_t getWgtBankConflicts() const;

        /**
         * Return the number of output activations bank conflicts
         * @return Output activations bank conflicts
         */
        uint64_t getOutBankConflicts() const;

        /**
         * Return activation memory size
         * @return Activation memory size
         */
        uint64_t getActSize() const;

        /**
         * Return weight memory size
         * @return Weight memory size
         */
        uint64_t getWgtSize() const;

        /**
         * Return activation memory banks
         * @return Activation memory banks
         */
        uint32_t getActBanks() const;

        /**
         * Return weight memory banks
         * @return Weight memory banks
         */
        uint32_t getWgtBanks() const;

        /**
         * Return output memory banks
         * @return Output memory banks
         */
        uint32_t getOutBanks() const;

        /**
         * Return number of addresses per access
         * @return Number of addresses per access
         */
        uint32_t getAddrsPerAccess() const;

        /**
         * Returns true when the read data is ready
         * @return True if data ready
         */
        bool data_ready() const;

        /**
         * Return activation and weight memory size for the file name
         * @return Activation and weight memory size
         */
        std::string filename();

        /**
         * Return stats header for the Global Buffer
         * @return Header
         */
        std::string header() override;

        /** Configure memory for current layer parameters */
        void configure_layer() override;

        /**
         * Check if all the writes are done
         * @return True if writes done
         */
        bool write_done();

        /**
         * Read request to the activation banks
         * @param tiles_data        Data to be read from the banks
         * @param layer_act_on_chip Layer activation on-chip flag
         */
        void act_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool layer_act_on_chip);

        /**
         * Read request to the output banks
         * @param tiles_data        Data to be read from the banks
         * @param read_psum         Update to True if psums to red (Overwritten)
         */
        void psum_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool &read_psum);

        /**
         * Read request to the weight banks
         * @param tiles_data        Data to be read from the banks
         */
        void wgt_read_request(const std::shared_ptr<TilesData<T>> &tiles_data);

        /**
         * Write request to the output banks
         * @param tiles_data        Data to be written to the banks
         */
        void write_request(const std::shared_ptr<TilesData<T>> &tiles_data);

        /**
         * Evict activations and/or weights from on-chip
         * @param evict_act If True evict activations
         * @param evict_wgt If True evict weight
         */
        void evict_data(bool evict_act, bool evict_wgt);

    };

}

#endif //DNNSIM_GLOBALBUFFER_H
