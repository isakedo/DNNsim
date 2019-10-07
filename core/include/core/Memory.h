#ifndef DNNSIM_MEMORY_H
#define DNNSIM_MEMORY_H

#include <sys/common.h>
#include <DRAMSim.h>

namespace core {

    /**
     * Container for the memory behaviour of the simulator
     * @tparam T Data type of the layer
     */
    class Memory {

    private:

        /** Memory system */
        DRAMSim::MultiChannelMemorySystem *memory;

        /** Transactions queue */
        std::queue<std::tuple<uint64_t, bool>> queue;

        /** Size for the on-chip activations memory */
        uint64_t on_chip_act_size;

        /** Size for the on-chip weights memory */
        uint64_t on_chip_wgt_size;

    public:

        /** Memory clock cycle */
        uint64_t mem_cycle;

        /** Requests */
        std::map<uint64_t, bool> requests;

        /** Called when read is ready. Update requests map
         * @param id Identifier of the request
         * @param address Address of the read request
         * @param clock_cycle Clock cycle at what the request was served
         */
        void transaction_done(unsigned id, uint64_t address, uint64_t clock_cycle);

        /** Request read from the off-chip dram
         * @param address Address of the read request
         * @param isWrite Type of transaction, True write, False read
         */
        void request_address(uint64_t address, bool isWrite);

        /** Pass memory cycles until the given clock cycle
         * @param clock_cycle Target clock cycle
         */
        void wait_until(uint64_t clock_cycle);

        /** Wait for the address to be served
         * @param address Address of the read request
         */
        void wait_for(uint64_t address);

        Memory() {
            memory = DRAMSim::getMemorySystemInstance("ini/DDR2_micron_16M_8b_x8_sg3E.ini", "system.ini",
                    "./DRAMSim2/", "DNNsim", 16384);

            DRAMSim::TransactionCompleteCB *read_cb =
                    new DRAMSim::Callback<Memory, void, unsigned, uint64_t, uint64_t>(this, &Memory::transaction_done);
            DRAMSim::TransactionCompleteCB *write_cb =
                    new DRAMSim::Callback<Memory, void, unsigned, uint64_t, uint64_t>(this, &Memory::transaction_done);

            memory->RegisterCallbacks(read_cb, write_cb, nullptr);
            mem_cycle = 0;
            on_chip_act_size = 1048576/8; //1MiB
            on_chip_wgt_size = 1048576/8; //1MiB
        }

        Memory(const std::string &dram_conf, const std::string &system_conf, uint64_t size) {
            memory = DRAMSim::getMemorySystemInstance(dram_conf, system_conf, "./DRAMSim2/", "DNNsim", size);

            DRAMSim::TransactionCompleteCB *read_cb =
                    new DRAMSim::Callback<Memory, void, unsigned, uint64_t, uint64_t>(this, &Memory::transaction_done);
            DRAMSim::TransactionCompleteCB *write_cb =
                    new DRAMSim::Callback<Memory, void, unsigned, uint64_t, uint64_t>(this, &Memory::transaction_done);

            memory->RegisterCallbacks(read_cb, write_cb, nullptr);
            mem_cycle = 0;
            on_chip_act_size = 1048576; //1MiB
            on_chip_wgt_size = 1048576; //1MiB
        }

        /**
         * Get On-Chip activations size
         * @return On-Chip activations size
         */
        uint64_t getOnChipActSize() const;

        /**
         * Get On-Chip weights size
         * @return On-Chip weights size
         */
        uint64_t getOnChipWgtSize() const;

    };

}

#endif //DNNSIM_MEMORY_H
