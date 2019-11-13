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

    protected:

        /** Memory system */
        DRAMSim::MultiChannelMemorySystem *memory;

        /** Transactions queue */
        std::queue<std::tuple<uint64_t, bool>> queue;

        /** Size for the on-chip activations memory */
        uint64_t on_chip_act_size;

        /** Size for the on-chip weights memory */
        uint64_t on_chip_wgt_size;

        /** Simulation memory clock cycle */
        uint64_t clock_cycle;

        /** Memory cycles stats */
        uint64_t mem_cycle;

        std::string dram_conf;

        std::string system_conf;

        uint64_t size;

        std::string network;

    public:

        /** Requests */
        std::map<uint64_t, bool> requests;

        /** Called when read is ready. Update requests map
         * @param id Identifier of the request
         * @param address Address of the read request
         * @param _clock_cycle Clock cycle at what the request was served
         */
        void transaction_done(unsigned id, uint64_t address, uint64_t _clock_cycle);

        /** Request read from the off-chip dram
         * @param address Address of the read request
         * @param isWrite Type of transaction, True write, False read
         */
        void request_address(uint64_t address, bool isWrite);

        /** Pass memory cycles until the given clock cycle
         * @param _clock_cycle Target clock cycle
         */
        void wait_until(uint64_t _clock_cycle);

        /** Wait for the address to be served
         * @param address Address of the read request
         */
        void wait_for(uint64_t address);

        void initialise();

        Memory() {

            dram_conf = "ini/DDR2_micron_16M_8b_x8_sg3E.ini";
            system_conf = "system.ini";
            size = 16384;

            clock_cycle = 0;
            mem_cycle = 0;

            on_chip_act_size = 1048576/80; //1MiB
            on_chip_wgt_size = 1048576/80; //1MiB

            network = "crap";
        }

        Memory(const std::string &_dram_conf, const std::string &_system_conf, uint64_t _size, uint64_t act_size,
                uint64_t wgt_size, const std::string &_network) {

            dram_conf = _dram_conf;
            system_conf = _system_conf;
            size = _size;

            clock_cycle = 0;
            mem_cycle = 0;

            on_chip_act_size = act_size;
            on_chip_wgt_size = wgt_size;

            network = _network;
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

        /**
         * Get current clock cycle in the simulation
         * @return Clock cycle
         */
        uint64_t getClockCycle() const;

        /**
         * Reset simulation clock cycle to 0
         */
        void resetClockCycle();

        /**
         * Get current memory stat clock cycle
         * @return Clock cycle
         */
        uint64_t getMemCycle() const;

        /**
         * Reset memory stat clock cycle to 0
         */
        void resetMemCycle();

    };

}

#endif //DNNSIM_MEMORY_H
