#ifndef DNNSIM_GLOBALBUFFER_H
#define DNNSIM_GLOBALBUFFER_H

#include <cstdint>
#include "Memory.h"
#include "FIFO.h"

namespace core {

    /**
     * Global Buffer model
     * @tparam T Data type values
     */
    template <typename T>
    class GlobalBuffer : public Memory<T> {

    private:

        /* SIMULATION PARAMETERS */

        const uint32_t ACT_LEVELS = 0;

        const uint32_t WGT_LEVELS = 0;

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

        /** Activation bank read delay */
        std::vector<uint32_t> ACT_READ_DELAY;

        /** Activation Write read delay */
        std::vector<uint32_t> ACT_WRITE_DELAY;

        /** Weight Bank read delay */
        std::vector<uint32_t> WGT_READ_DELAY;

        /** Activation Bank interface datawidth */
        const uint32_t ACT_BANK_WIDTH = 0;

        /** Weights Bank interface datawidth */
        const uint32_t WGT_BANK_WIDTH = 0;

        /** Activations addresses per access */
        const uint32_t ACT_ADDRS_PER_ACCESS = 0;

        /** Weights addresses per access */
        const uint32_t WGT_ADDRS_PER_ACCESS = 0;

        std::vector<std::vector<std::shared_ptr<EvictionPolicy>>> act_eviction_policy;

        std::vector<std::vector<std::shared_ptr<EvictionPolicy>>> out_eviction_policy;

        std::vector<std::vector<std::shared_ptr<EvictionPolicy>>> wgt_eviction_policy;

        /** Partial sum banks ready cycle */
        uint64_t psum_read_ready_cycle = 0;

        /** Input banks ready cycle */
        uint64_t read_ready_cycle = 0;

        /** Output Activations banks ready cycle */
        uint64_t write_ready_cycle = 0;

        /* STATISTICS */

        /** Activation bank reads */
        std::vector<uint64_t> act_reads;

        /** Partial sum bank reads */
        std::vector<uint64_t> psum_reads;

        /** Weight bank reads */
        std::vector<uint64_t> wgt_reads;

        /** Output bank writes */
        std::vector<uint64_t> out_writes;

        /** Activation bank conflicts */
        std::vector<uint64_t> act_bank_conflicts;

        /** Partial sum bank conflicts */
        std::vector<uint64_t> psum_bank_conflicts;

        /** Weight bank conflicts */
        std::vector<uint64_t> wgt_bank_conflicts;

        /** Output Activation bank conflicts */
        std::vector<uint64_t> out_bank_conflicts;

    public:

        /**
         * Constructor
         * @param _tracked_data         Current tracked data on-chip
         * @param _act_addresses        Address range for activations
         * @param _out_addresses        Output activation addresses range
         * @param _wgt_addresses        Address range for weights
         * @param _ACT_LEVELS           Activation Hierarchy levels
         * @param _WGT_LEVELS           Weight Hierarchy levels
         * @param _ACT_SIZE             Activation size in Bytes
         * @param _WGT_SIZE             Weights size in Bytes
         * @param _ACT_OUT_BANKS        Activations banks
         * @param _WGT_BANKS            Weight banks
         * @param _ACT_BANK_WIDTH       Activations on-chip bank width
         * @param _WGT_BANK_WIDTH       Weights on-chip bank width
         * @param _DRAM_WIDTH           Dram width
         * @param _ACT_READ_DELAY       Activation bank read delay
         * @param _ACT_WRITE_DELAY      Activation bank write delay
         * @param _WGT_READ_DELAY       Weight bank read delay
         */
        GlobalBuffer(const std::shared_ptr<std::map<uint64_t, uint32_t>> &_tracked_data,
                const std::shared_ptr<AddressRange> &_act_addresses, const std::shared_ptr<AddressRange> &_out_addresses,
                const std::shared_ptr<AddressRange> &_wgt_addresses, uint32_t _ACT_LEVELS, uint32_t _WGT_LEVELS,
                const std::vector<uint64_t> &_ACT_SIZE, const std::vector<uint64_t> &_WGT_SIZE, uint32_t _ACT_OUT_BANKS,
                uint32_t _WGT_BANKS, uint32_t _ACT_BANK_WIDTH, uint32_t _WGT_BANK_WIDTH, uint32_t _DRAM_WIDTH,
                const std::vector<uint32_t> & _ACT_READ_DELAY, const std::vector<uint32_t> & _ACT_WRITE_DELAY,
                const std::vector<uint32_t> & _WGT_READ_DELAY) : Memory<T>(_tracked_data, _act_addresses, _out_addresses,
                _wgt_addresses), ACT_LEVELS(_ACT_LEVELS), WGT_LEVELS(_WGT_LEVELS), ACT_BANKS(_ACT_OUT_BANKS/2),
                WGT_BANKS(_WGT_BANKS), OUT_BANKS(_ACT_OUT_BANKS/2), ACT_BANK_WIDTH(_ACT_BANK_WIDTH),
                WGT_BANK_WIDTH(_WGT_BANK_WIDTH), ACT_ADDRS_PER_ACCESS(ceil(ACT_BANK_WIDTH / (double)_DRAM_WIDTH)),
                WGT_ADDRS_PER_ACCESS(ceil(WGT_BANK_WIDTH / (double)_DRAM_WIDTH)) {

            ACT_SIZE = _ACT_SIZE;
            WGT_SIZE = _WGT_SIZE;
            ACT_READ_DELAY = _ACT_READ_DELAY;
            ACT_WRITE_DELAY = _ACT_WRITE_DELAY;
            WGT_READ_DELAY = _WGT_READ_DELAY;

            act_eviction_policy = std::vector<std::vector<std::shared_ptr<EvictionPolicy>>>(ACT_LEVELS,
                    std::vector<std::shared_ptr<EvictionPolicy>>(ACT_BANKS, std::shared_ptr<EvictionPolicy>()));

            for (int lvl = 1; lvl < ACT_LEVELS; ++lvl) {
                auto bank_size = ACT_SIZE[lvl] / _DRAM_WIDTH / _ACT_OUT_BANKS;
                for (int bank = 0; bank < ACT_BANKS; ++bank) {
                    assert(bank_size > 0);
                    act_eviction_policy[lvl][bank] = std::make_shared<FIFO>(bank_size);
                }
            }

            out_eviction_policy = std::vector<std::vector<std::shared_ptr<EvictionPolicy>>>(ACT_LEVELS,
                    std::vector<std::shared_ptr<EvictionPolicy>>(OUT_BANKS, std::shared_ptr<EvictionPolicy>()));

            for (int lvl = 1; lvl < ACT_LEVELS; ++lvl) {
                auto bank_size = ACT_SIZE[lvl] / _DRAM_WIDTH / _ACT_OUT_BANKS;
                for (int bank = 0; bank < OUT_BANKS; ++bank) {
                    assert(bank_size > 0);
                    out_eviction_policy[lvl][bank] = std::make_shared<FIFO>(bank_size);
                }
            }

            wgt_eviction_policy = std::vector<std::vector<std::shared_ptr<EvictionPolicy>>>(WGT_LEVELS,
                    std::vector<std::shared_ptr<EvictionPolicy>>(WGT_BANKS, std::shared_ptr<EvictionPolicy>()));

            for (int lvl = 1; lvl < WGT_LEVELS; ++lvl) {
                auto bank_size = WGT_SIZE[lvl] / _DRAM_WIDTH / _WGT_BANKS;
                for (int bank = 0; bank < WGT_BANKS; ++bank) {
                    assert(bank_size > 0);
                    wgt_eviction_policy[lvl][bank] = std::make_shared<FIFO>(bank_size);
                }
            }
        }

        /**
         * Return the number of activation bank reads
         * @param idx  Hierarchy level
         * @return Activation bank reads
         */
        uint64_t getActReads(uint32_t idx) const;

        /**
         * Return the number of output bank reads
         * @param idx  Hierarchy level
         * @return Output bank reads
         */
        uint64_t getPsumReads(uint32_t idx) const;

        /**
         * Return the number of weight bank reads
         * @param idx  Hierarchy level
         * @return Weight bank reads
         */
        uint64_t getWgtReads(uint32_t idx) const;

        /**
         * Return the number of output bank writes
         * @param idx  Hierachy level
         * @return Output bank writes
         */
        uint64_t getOutWrites(uint32_t idx) const;

        /**
         * Return the number of activation bank conflicts
         * @param idx  Hierarchy level
         * @return Activation bank conflicts
         */
        uint64_t getActBankConflicts(uint32_t idx) const;

        /**
         * Return the number of partial sum bank conflicts
         * @param idx  Hierarchy level
         * @return Partial sum bank conflicts
         */
        uint64_t getPsumBankConflicts(uint32_t idx) const;

        /**
         * Return the number of weight bank conflicts
         * @param idx  Hierarchy level
         * @return Weight bank conflicts
         */
        uint64_t getWgtBankConflicts(uint32_t idx) const;

        /**
         * Return the number of output activations bank conflicts
         * @param idx  Hierarchy level
         * @return Output activations bank conflicts
         */
        uint64_t getOutBankConflicts(uint32_t idx) const;

        /**
         * Return activation hierarchy levels
         * @return Activation hierarchy levels
         */
        uint64_t getActLevels() const;

        /**
         * Return weight hierarchy levels
         * @return Weight hierarchy levels
         */
        uint64_t getWgtLevels() const;

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
         * Return number of activation addresses per access
         * @return Number of activation addresses per access
         */
        uint32_t getActAddrsPerAccess() const;

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
         * @param read_act          Update to True if activations to be read (Overwritten)
         * @param layer_act_on_chip Layer activation on-chip flag
         */
        void act_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool layer_act_on_chip, bool &read_act);

        /**
         * Read request to the output banks
         * @param tiles_data        Data to be read from the banks
         * @param read_psum         Update to True if partial sums to be read (Overwritten)
         */
        void psum_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool &read_psum);

        /**
         * Read request to the weight banks
         * @param tiles_data        Data to be read from the banks
         * @param read_wgt          Update to True if weights to be read (Overwritten)
         */
        void wgt_read_request(const std::shared_ptr<TilesData<T>> &tiles_data, bool &read_wgt);

        /**
         * Write request to the output banks
         * @param tiles_data        Data to be written to the banks
         */
        void write_request(const std::shared_ptr<TilesData<T>> &tiles_data);

        /**
         * Evict activations and/or weights from on-chip
         * @param evict_act If True evict activations
         * @param evict_out If True evict outputs and partial sums
         * @param evict_wgt If True evict weight
         */
        void evict_data(bool evict_act, bool evict_out, bool evict_wgt);

    };

}

#endif //DNNSIM_GLOBALBUFFER_H
